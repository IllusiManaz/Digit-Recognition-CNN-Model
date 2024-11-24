# Digit-Recognition-CNN-Model
This project implements a Digit Recognition Model using Convolutional Neural Networks (CNNs), built on the MNIST dataset, a large collection of handwritten digits commonly used for training machine learning models. The model is trained to classify images of digits (0-9) and can be used to predict the digit in new input images.

# Overview
This repository contains code for:

# Loading and preprocessing the MNIST dataset.
1. Building a Convolutional Neural Network (CNN) model.
2. Training the model on the MNIST dataset.
3. Saving the trained model to a file.
4. Loading a saved model and making predictions on new images.

# Prerequisites
Before you begin, ensure that you have the following software installed:

1. Python (3.x)
2. TensorFlow (>=2.0)
3. Matplotlib
4. NumPy
5. PIL or Pillow (for image manipulation)

You can install the necessary dependencies using pip by running:

pip install -r requirements.txt
Or you can manually install the dependencies:

pip install tensorflow matplotlib numpy pillow

# Dataset
The model uses the MNIST dataset, which contains 60,000 28x28 grayscale images of handwritten digits (0-9) for training and 10,000 images for testing. This dataset is readily available through TensorFlow.

# Files in the Repository
1. model.py: Contains the CNN model architecture and training code.
2. train.py: Script to load the MNIST dataset, train the model, and save it.
3. predict.py: Script to load the pre-trained model and make predictions on new images.
4. lenet5.h5: The trained CNN model saved as an H5 file after training.
5. requirements.txt: List of dependencies required to run the project.

# Steps to Run the Project
**Step 1:** Train the Model
Run the train.py script to train the CNN model on the MNIST dataset.

--> python train.py

This script will:

1. Load the MNIST dataset.
2. Preprocess the images (normalize and reshape).
3. Define and compile a CNN model.
4. Train the model using the training data and validate it using the test data.
5. Save the trained model to the file lenet5.h5.

**Step 2:** Make Predictions with the Trained Model
Once the model is trained, you can use the predict.py script to make predictions on new images.

-->python predict.py

In this script, an external image (such as a number from the web) is loaded, preprocessed (cropped and resized to 28x28 pixels), and passed through the trained model to predict the digit. The result is then printed out.

**Step 3:** Use the Pre-trained Model
If you just want to test the pre-trained model without training it yourself, you can download the lenet5.h5 file and run the predict.py script directly.

1. Download the pre-trained model: lenet5.h5
2. Run the predict.py script as mentioned in Step 2.
Example Image Prediction
The predict.py script will load an image from a URL (e.g., a digit or a number) and make a prediction. Here's an example of how to make a prediction:

--> image=urlopen("https://www.shutterstock.com/image-photo/single-number-fire-flames-alphabet-260nw-2099710750.jpg")

You can replace the image URL with any valid image link that contains a handwritten digit or a number.

# Model Evaluation
Once the model is trained, you can evaluate its performance on the test dataset by running the following:

--> python evaluate.py

This will display the accuracy and loss of the model on the test data, helping you understand how well it generalizes to unseen examples.

# Model Architecture
The architecture of the CNN model is as follows:

1. Input Layer: Takes in 28x28 grayscale images.
2. Convolution Layer (Conv2D): First convolutional layer with 12 filters and a kernel size of 5x5.
3. MaxPooling Layer (MaxPool2D): A max-pooling layer with a pool size of 2x2.
4. Convolution Layer (Conv2D): Second convolutional layer with 32 filters and a kernel size of 5x5.
5. MaxPooling Layer (MaxPool2D): Another max-pooling layer.
6. Convolution Layer (Conv2D): Third convolutional layer with 240 filters and a kernel size of 5x5.
7. Flatten Layer: Flatten the 3D feature maps to 1D vector.
8. Dense Layer (Fully Connected): Fully connected layer with 128 units.
9. Output Layer: Dense layer with 10 units (corresponding to digits 0-9) and a softmax activation function.

# Performance
The model achieves a validation accuracy of around 99.5% on the MNIST dataset after training for several epochs. The accuracy on the test set is approximately 99%.

# Future Improvements
1. Data Augmentation: You could implement data augmentation (e.g., rotation, scaling, translation) to make the model more robust.
2. Hyperparameter Tuning: Experiment with different hyperparameters (e.g., learning rate, batch size, number of filters).
3. Deeper Model: Try using a deeper CNN model for potentially higher accuracy.
4. Real-time Digit Recognition: Extend this project to recognize digits from a live camera feed.

# Application:
1. **Healthcare and Medical Imaging:**

Use: Recognizing and extracting handwritten numbers in medical records, prescriptions, or diagnostic forms.
Example: Recognizing prescription dosages or patient ID numbers from handwritten notes in medical records.

2. **Captcha Solving and Anti-Bot Systems:**

Use: Digit recognition is used to solve CAPTCHA challenges that require recognizing and typing digits from distorted images, helping to distinguish humans from bots.
Example: Online login forms, registration processes, and financial transactions may use CAPTCHA with digit recognition.
