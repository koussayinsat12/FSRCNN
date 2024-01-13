import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import yaml
from dataset import Dataset
from constants import HR_IMG_SIZE, DOWNSAMPLE_MODE
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
model = keras.models.load_model("weights\model_00159.h5")
test_dataset = Dataset(
    hr_image_folder=config["data_path"],
    batch_size=config["val_batch_size"],
    set_type="test",
)
#Evaluation
n_runs = 5
psnrs = []
for _ in range(n_runs):
    for batch in test_dataset:
        preds = model.predict(batch[0])
        psnr = tf.image.psnr(batch[1], preds, max_val=1.0)
        psnr = psnr.numpy().tolist()
        psnrs.extend(psnr)

print("Mean PSNR: {:.3f}".format(np.mean(psnrs)))
#Visualization
batch_id = 5
batch = test_dataset.__getitem__(batch_id)
preds = model.predict(batch[0])
img_id = 1
plt.figure(figsize=[15, 15])
plt.subplot(2, 2, 1)
plt.imshow(batch[0][img_id])
plt.axis("off")
plt.title("LR Image")
plt.subplot(2, 2, 2)
plt.imshow(batch[1][img_id])
plt.axis("off")
plt.title("HR Image")
plt.subplot(2, 2, 3)
plt.imshow(preds[img_id])
plt.axis("off")
plt.title("Restored Image")
plt.subplot(2, 2, 4)
lr_image = Image.fromarray(np.array(batch[0][img_id] * 255, dtype="uint8"))
lr_image_resized = lr_image.resize(HR_IMG_SIZE, resample=DOWNSAMPLE_MODE)
plt.imshow(lr_image_resized)
plt.axis("off")
plt.title("Bilinear Upsampling")
plt.tight_layout()
plt.show()