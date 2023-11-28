# Banana Grade Classifier
Banana Grade Classifier is an AI-based system that classifies bananas into different grades based on their ripeness and quality. This system helps farmers and distributors to sort bananas efficiently and accurately, which can increase their productivity and reduce waste. Additionally, consumers can benefit from the system by being able to choose bananas with their desired ripeness and quality.

# Dataset
The dataset for this project has been taken from Kaggle. It contains a total of 5,000 images of bananas in three different classes: A, B, and C. Class A represents high-quality bananas, class B represents medium-quality bananas, and class C represents low-quality bananas.

The dataset has been split into two parts: 80% for training and 20% for testing. The images have been preprocessed to remove any background noise and standardize their sizes.

# Model Architecture
The model for this project has been built using a convolutional neural network (CNN) in Keras. The CNN consists of three convolutional layers with increasing filters, followed by two fully connected layers. The model has been trained on the training dataset for 50 epochs using the Adam optimizer and categorical cross-entropy loss function.

# Usage
To use the Banana Grade Classifier, you can simply
1) install the requirement.txt file
Then run the command 
2) Uvicorn main:app --reload 
3) For frontend go to the frontend and use the readme.nd there to run the frontend part.

# Technologies Used
Python
TensorFlow
Keras
OpenCV
NumPy
Matplotlib
Fastapi


## Author: Vandit Tyagi
## License
This project is licensed under the MIT License. Feel free to use and modify this code as per your requirement
