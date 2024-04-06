# FoodVision_101

This repository contains Python code for performing transfer learning using the EfficientNetB1 model on the Food-101 dataset. Transfer learning enables us to leverage pre-trained models and fine-tune them for specific tasks, such as image classification on a custom dataset.

## Dataset
The Food-101 dataset is a large-scale dataset consisting of 101 food categories, with 101,000 images in total. Each category contains 1,000 images, split into training and testing sets. This dataset is commonly used for food image classification tasks.

Note: Please download the Food-101 dataset from [source_link] and place it in the data directory before running the code.

## Requirements
* Python 3.x
* TensorFlow
* TensorFlow Hub
* scikit-learn
* matplotlib
* numpy

## Model Architecture
We utilize EfficientNetB1, a state-of-the-art convolutional neural network (CNN) architecture, as the base model for transfer learning. EfficientNetB1 achieves excellent performance while being computationally efficient, making it suitable for various computer vision tasks.

## Results
The script fine-tunes the pre-trained EfficientNetB1 model on the Food-101 dataset. It evaluates the performance of the model on the test dataset and plots the training/validation loss and accuracy curves. Additionally, it provides metrics such as accuracy for each food category.
  
