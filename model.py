from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, PReLU, InputLayer
from tensorflow.keras import initializers
from tensorflow.keras import callbacks
from dataset import Dataset
import yaml
from constants import *

class Model:
    def __init__(self, config_fn: str):
        with open(config_fn, 'r') as yaml_file:
            self.config = yaml.safe_load(yaml_file)
    def create_model(self,
                     input_size: tuple = LR_IMG_SIZE,
                     upscaling_factor: tuple = UPSCALING_FACTOR,
                     color_channels: int = COLOR_CHANNELS):
        d = self.config['model_d']
        s = self.config['model_s']
        m = self.config['model_m']
        model = Sequential()
        # Input Layer
        model.add(InputLayer(input_shape=(input_size[0], input_size[1], color_channels)))
        # feature extraction
        model.add(Conv2D(
            kernel_size=5,
            filters=d,
            padding="same",
            kernel_initializer=initializers.HeNormal(),
        ))
        model.add(PReLU(alpha_initializer='zeros', shared_axes=[1, 2]))
        # shrinking
        model.add(Conv2D(
            kernel_size=1,
            filters=s,
            padding='same',
            kernel_initializer=initializers.HeNormal(),
        ))
        model.add(PReLU(alpha_initializer='zeros', shared_axes=[1, 2]))
        # non-linear mapping
        for _ in range(m):
            model.add(Conv2D(
                kernel_size=3,
                filters=s,
                padding="same",
                kernel_initializer=initializers.HeNormal(),
            ))
        model.add(PReLU(alpha_initializer='zeros', shared_axes=[1, 2]))
        # Expanding
        model.add(Conv2D(kernel_size=1, filters=d, padding="same"))
        model.add(PReLU(alpha_initializer='zeros', shared_axes=[1, 2]))
        # Deconvolution
        model.add(Conv2DTranspose(
            kernel_size=9,
            filters=color_channels,
            strides=upscaling_factor,
            padding='same',
            kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.001),
        ))

        return model

    def train(self):
        train_dataset = Dataset(
            hr_image_folder=self.config['data_path'],
            batch_size=self.config["batch_size"],
            set_type="train",
        )
        val_dataset = Dataset(
            hr_image_folder=self.config["data_path"],
            batch_size=self.config["val_batch_size"],
            set_type="val",
        )
        model = self.create_model()
        model.compile(
            optimizer='rmsprop',
            loss="mean_squared_error",
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=20, min_lr=10e-6, verbose=1
        )
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=10e-6,
            patience=40,
            verbose=0,
            restore_best_weights=True,
        )
        save = callbacks.ModelCheckpoint(
            filepath=self.config["weights_fn"],
            monitor="loss",
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
        )

        history = model.fit(
            train_dataset,
            epochs=self.config["epochs"],
            steps_per_epoch=self.config["steps_per_epoch"],
            callbacks=[reduce_lr, early_stopping, save],
            validation_data=val_dataset,
            validation_steps=self.config["validation_steps"],
        )



