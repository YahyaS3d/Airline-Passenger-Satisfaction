# Airline Passenger Satisfaction Classification

## Overview
This project develops a neural network model to classify airline passengers into three categories based on their satisfaction: dissatisfied, neutral, or satisfied. It utilizes a dataset containing various features related to passenger experience, including demographic information, travel details, and flight convenience.

## Dataset
The dataset is sourced from Kaggle and comprises features like Gender, Customer Type, Age, Type of Travel, Class, Flight Distance, Departure Delay, and Arrival Delay. The target variable is the overall customer satisfaction level.

Kaggle Dataset Link: [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

## Preprocessing
Data preprocessing steps include:
- Filling missing values in 'Arrival Delay in Minutes'.
- Encoding categorical features using OneHotEncoder.
- Normalizing numerical features using StandardScaler.
- Splitting the dataset into training and test sets.

## Model
The neural network model consists of:
- An input layer matching the number of features.
- Two hidden layers with ReLU activation.
- An output layer with softmax activation for multi-class classification.
- The model uses categorical cross-entropy as the loss function and Adam optimizer.

## Training
The model is trained with early stopping to prevent overfitting. Training involves:
- Using a batch size of 32.
- Validating on 20% of the training data.
- Monitoring validation loss for early stopping.

## Evaluation
Model performance is evaluated on the test set. Accuracy and loss metrics are calculated to assess how well the model classifies unseen data.

## Discussion
Performance analysis suggests the model generalizes well, with considerations for potential overfitting or underfitting discussed. Strategies for improvement include regularization, data augmentation, and model adjustments.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- TensorFlow/Keras
- matplotlib

## Usage
1. Clone the repository.
2. Ensure the dataset is placed in the correct directory.
3. Install the required libraries.
4. Run the model training and evaluation script.

## Conclusion
This project demonstrates the application of neural networks in classifying airline passenger satisfaction. With further tuning and analysis, the model's accuracy can be improved, providing valuable insights for enhancing customer experience.
