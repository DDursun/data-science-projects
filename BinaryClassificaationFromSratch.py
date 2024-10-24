from ucimlrepo import fetch_ucirepo
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

# Set of helper functions
def encode_data(dataset, encoding_map):
    """
    Function that encodes the data based on input encoding map of columns.
    """
    for column, method in encoding_map.items():
        if method == 'ordinal':
            custom_order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th',
                            '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc',
                            'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate']
            mapping = {value: idx + 1 for idx, value in enumerate(custom_order)}
            dataset[column] = dataset[column].map(mapping)
        
        # One-hot encoding 
        else: 
            uniques = dataset[column].unique()
            for value in uniques:
                dataset.loc[:, f"{column}_{value}"] = (dataset[column] == value).astype(int)
            dataset.drop(column, axis=1, inplace=True)
    return dataset

def min_max_scaler(X):
    """Scales the features of X, except the intercept."""
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

def calculate_gradient(X, y, w, lamda):
    factor = y * np.dot(X, w)
    gradient = - y * (np.exp(factor) / (1 + np.exp(factor))) * X + lamda*w
    return gradient

def gradient_descent(X, y, w, learning_rate, lambda_reg, num_iterations):
    for epoch in range(num_iterations):
        for i in range(len(X)):
            X_i = X.iloc[i].to_numpy().reshape(1, -1)
            y_i = y.iloc[i]

            # Calculate gradient for the i-th training example
            grad = calculate_gradient(X_i, y_i, w, lambda_reg)

            # Update weights
            w -= learning_rate * grad.flatten()

        print(f"Epoch {epoch + 1}/{num_iterations} completed.")

    return w

def predict(X, w):
    probability = 1 / (1 + np.exp(-np.dot(X, w)))
    predictions = np.where(probability >= 0.5, 1, -1)
    return predictions

def calculate_accuracy(X, y, w):
    predictions = predict(X, w)
    accuracy = np.mean(predictions == y)
    return accuracy

def main():
    # Fetching dataset
    adult = fetch_ucirepo(id=2)

    X = adult.data.features
    y = adult.data.targets

    # Data prepatation #
     
    # Drop unnecessary columns
    X.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)
    # Removing rows with NaNs
    X = X.dropna()
    y = y.loc[X.index].dropna()

    # Converting y values to -1 and 1 
    y = y['income'].str.strip().str.replace('.', '', regex=False)
    transform_y = {"<=50K": -1, ">50K": 1}
    y = y.replace(transform_y)

    # Encode categorical variables
    encoding_map = {"workclass": "one-hot", "education": "ordinal", "marital-status": "one-hot", "occupation": "one-hot", 
                    "relationship": "one-hot","race": "one-hot", "sex": "one-hot", "native-country": "one-hot"}
    X = encode_data(X, encoding_map)

    # Split data into training and test sets
    Xtraining, Xtest, Ytraining, Ytest, Xval, Yval = train_test_split(X, y)
    Xtraining_scaled = min_max_scaler(Xtraining)  
    Xtest_scaled = min_max_scaler(Xtest)

    # Add intercept column
    Xtraining_scaled["intercept"] = 1 
    Xtest_scaled["intercept"] = 1

    # 2.1 Gradient on one sample
    first_row = Xtraining_scaled.iloc[0].to_numpy().reshape(1, -1)
    first_rowy = Ytraining.iloc[0]
    w = np.zeros(Xtraining_scaled.shape[1])

    grad = calculate_gradient(first_row, first_rowy, w, 1)

    ### Part 2.2 ####

    # Parameters
    learning_rate = 0.01
    lambda_reg = 1
    num_iterations = 5

    # Initialize weights with zero matrix
    w = np.zeros(Xtraining_scaled.shape[1])

    # Perform gradient descent
    w = gradient_descent(Xtraining_scaled, Ytraining, w, learning_rate, lambda_reg, num_iterations)

    # Test the accuracy values
    training_accuracy = calculate_accuracy(Xtraining_scaled.to_numpy(), Ytraining.to_numpy(), w)
    print(f"Training Accuracy: {training_accuracy * 100:.2f}%")

    test_accuracy = calculate_accuracy(Xtest_scaled.to_numpy(), Ytest.to_numpy(), w)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
