# Fast Super-Resolution Convolutional Neural Network (FSRCNN)

## Overview

This project implements the Fast Super-Resolution Convolutional Neural Network (FSRCNN) for image super-resolution. FSRCNN is a deep learning model designed to enhance the resolution of low-resolution images. The project includes components such as data processing, model architecture, training, and inference.

## Project Structure

1. **config.yaml**: Configuration file specifying parameters such as data paths, model architecture, learning rate, and training settings.

2. **constants.py**: Defines constants used throughout the project, such as image formats, downsample mode, color channels, and image sizes.

3. **dataset.py**: Implements the `Dataset` class, a custom data generator using Keras `Sequence`. It loads and augments high-resolution images for training and validation.

4. **inference.py**: Conducts inference on a trained model using a test dataset. Evaluates the model's performance in terms of Peak Signal-to-Noise Ratio (PSNR) and provides visualizations.

5. **model.py**: Contains the `Model` class, responsible for defining the SRCNN architecture, training the model, and saving the weights.

6. **train.py**: Orchestrates the training process by creating an instance of the `Model` class and calling the `train` method.

## Configuration

Modify the `config.yaml` file to customize the project settings. Key parameters include the data path, model dimensions, learning rate, training epochs, and batch sizes.

## Model Architecture

The SRCNN model consists of a series of convolutional layers for feature extraction, shrinking, non-linear mapping, expanding, and deconvolution. The architecture is configured based on parameters specified in the `config.yaml` file.

## Training

Run the `train.py` script to initiate the training process. The script loads the dataset, creates an instance of the SRCNN model, and trains the model using specified configurations. Training progress and model weights are saved to the specified file paths.

## Inference

Use the `inference.py` script to evaluate the trained model on a test dataset. The PSNR metric is calculated, and visualizations comparing low-resolution, high-resolution, and restored images are displayed.

## Dependencies

- TensorFlow
- Keras
- PIL (Python Imaging Library)
- NumPy
- Matplotlib
- Albumentations

Ensure these dependencies are installed before running the scripts.

## How to Run

1. Install dependencies using `pip install -r requirements.txt`.
2. Adjust configurations in `config.yaml`.
3. Run `train.py` to train the model.
4. Run `inference.py` to evaluate the model and visualize results.

Feel free to explore and modify the project to suit your specific requirements.
