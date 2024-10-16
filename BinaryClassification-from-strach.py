from ucimlrepo import fetch_ucirepo
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import math

# Set of helper functions

def encode_data(dataset, encoding_map):
    for column, method in encoding_map.items():
        if method == 'ordinal':
            custom_order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th',
                            '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc',
                            'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate']
            mapping = {value: idx + 1 for idx, value in enumerate(custom_order)}
            dataset[column] = dataset[column].map(mapping)
        else:  # One-hot encoding case
            uniques = dataset[column].unique()
            for value in uniques:
                dataset.loc[:, f"{column}_{value}"] = (dataset[column] == value).astype(int)
            dataset.drop(column, axis=1, inplace=True)
    return dataset

# Custom scaling function

def min_max_scaler(X):
    """Scales the features of X to be in the range [0, 1]."""
    X_scaled = X.copy()
    for column in X_scaled.columns:
        min_value = X_scaled[column].min()
        max_value = X_scaled[column].max()
        X_scaled[column] = (X_scaled[column] - min_value) / (max_value - min_value)
    return X_scaled

def train_test_split(X, y):
    """ Takes features and target value,
        Returns Training, Validation and Test sets """ 
    ntrain = int(len(X) * 0.8)
    nval = int(len(X) * 0.9)
    Xtraining, Xval, Xtest = X[:ntrain], X[ntrain:nval], X[nval:]
    Ytraining, Yval, Ytest = y[:ntrain], y[ntrain:nval], y[nval:]
    return Xtraining, Xtest, Ytraining, Ytest, Xval, Yval

def calculate_gradient(X, y, w, l):
    factor = y * np.dot(X, w)
    gradient = - y * (math.exp(factor) / (1 + math.exp(factor))) * X + l*X
    return gradient

def gradient_descent(X, y, w, learning_rate, lambda_reg):
   
    for i in range(len(X)):

        X_i = X.iloc[i].to_numpy().reshape(1, -1)
        y_i = y.iloc[i]

        # Calculate gradient for the i-th training example
        grad = calculate_gradient(X_i, y_i, w, lambda_reg)

        # Update weights
        w -= learning_rate * grad.flatten()

    return w

def predict(X, w):
    # Calculate the probability using the logistic function
    probability = 1 / (1 + np.exp(-np.dot(X, w)))
    # Convert probabilities to class labels (-1 or 1)
    predictions = np.where(probability >= 0.5, 1, -1)
    return predictions

def calculate_accuracy(X, y, w):
    # Make predictions on the dataset
    predictions = predict(X, w)
    # Calculate accuracy
    accuracy = np.mean(predictions == y)
    return accuracy


def main():
        
    # Fetching dataset
    adult = fetch_ucirepo(id=2)

    X = adult.data.features
    y = adult.data.targets

    # Drop unnecessary columns
    X.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)

    # Remove rows with NaNs in either X or y
    X = X.dropna()
    y = y.loc[X.index].dropna()

    # Converting y values to -1 and 1 
    y = y['income'].str.strip().str.replace('.', '', regex=False)
    transform_y = {"<=50K": -1, ">50K": 1}
    y = y.replace(transform_y)


    encoding_map = {"workclass": "one-hot", "education": "ordinal", "marital-status": "one-hot", "occupation": "one-hot", "relationship": "one-hot","race": "one-hot", "sex": "one-hot", "native-country": "one-hot"}

    # Encode categorical variables
    X = encode_data(X, encoding_map)

    # Add intercept column
    X["intercept"] = 1 

    # Split data into training and test sets
    Xtraining, Xtest, Ytraining, Ytest, Xval, Yval = train_test_split(X, y)
    Xtraining_scaled = min_max_scaler(Xtraining)  
    Xtest_scaled = min_max_scaler(Xtest)

    # 2.1 Gradient on one sample
    first_row = Xtraining.iloc[0].to_numpy().reshape(1, -1)
    first_rowy = Ytraining.iloc[0]
    w = np.zeros(Xtraining.shape[1])

    grad = calculate_gradient(first_row, first_rowy, w, 1)

    ### Part 2.2 ####

    # Parameters
    learning_rate = 0.01
    lambda_reg = 1

    # Initialize weights to zeros
    w = np.zeros(Xtraining.shape[1])

    # Perform gradient descent
    w = gradient_descent(Xtraining_scaled, Ytraining, w, learning_rate, lambda_reg)

    # Test the accuracy on the training dataset
    training_accuracy = calculate_accuracy(Xtraining_scaled.to_numpy(), Ytraining.to_numpy(), w)
    print(f"Training Accuracy: {training_accuracy * 100:.2f}%")

    test_accuracy = calculate_accuracy(Xtest_scaled.to_numpy(), Ytest.to_numpy(), w)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
