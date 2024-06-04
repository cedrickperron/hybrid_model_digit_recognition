import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


import joblib
from time import time


class RandomForestClassifierCustom:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.model = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.accuracy = None
        self.params = None
        
    def fit(self, max_depth, min_samples_split, criterion):
        """
        Fits the model with the training dataset.

        Parameters:
        max_depth: int
            Maximum depth of the tree.
        min_samples_split: int
            Minimum number of samples required to split an internal node.
        criterion: str
            Function to measure the quality of a split.

        Returns:
        None
        """
        self.model = RandomForestClassifier(random_state=1, criterion=criterion, min_samples_split=min_samples_split, max_depth=max_depth)
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test=None):
        """
        Predicts the target values for the input data.

        Parameters:
        X_test: array-like, shape (n_samples, n_features), optional (default=None)
            Test data.

        Returns:
        array-like, shape (n_samples,)
            Predicted target values.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call fit method first.")

        if X_test is None:
            X_test = self.X_test
        return self.model.predict(X_test)

    def score(self, y_true, y_pred):
        """
        Calculates the accuracy score based on true and predicted values.

        Parameters:
        y_true: array-like, shape (n_samples,)
            True target values.
        y_pred: array-like, shape (n_samples,)
            Predicted target values.

        Returns:
        float
            Accuracy score.
        """
        return accuracy_score(y_true, y_pred)

    def best_fit(self, max_depths = [1, 5, 10, 15, 20, 25, 30, 50, 100],
                 min_samples_splits = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                 criterions=["entropy", "gini"] ):
        """
        Finds the best model based on accuracy score.

        Parameters:
        max_depths: list of int
            Maximum depths of the trees to try.
        min_samples_splits: list of int
            Minimum number of samples required to split an internal node.
        criterions: list of str
            Functions to measure the quality of a split.

        Returns:
        sklearn.ensemble.RandomForestClassifier
            Best trained model.
        tuple
            Hyperparameters (max_depth, min_samples_split, criterion) for the best model.
        float
            Accuracy score of the best model.
        """
        max_accuracy = 0.0
        best_hyperparameters = None
        best_model = None
        for criterion in criterions:
            for max_depth in max_depths:
                for min_samples_split in min_samples_splits:
                    self.fit(max_depth, min_samples_split, criterion)
                    predictions = self.predict()
                    accuracy = self.score(self.y_test, predictions)
                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                        best_hyperparameters = (max_depth, min_samples_split, criterion)
                        best_model = self.model

        self.accuracy = max_accuracy
        self.params = best_hyperparameters
        self.model = best_model
        return self.model

    def parameter_plot(self):
        """
        Plots accuracy scores for different values of k.

        Parameters:
        None

        Returns:
        None
        """
        pass
    
    def print_best_params(self):
        """
        Prints the best hyperparameters and accuracy score.
        """
        print(f"Validation accuracy for Random Forest: {self.accuracy:.3f}")
        print(f"Best parameters: {self.params}")

    def save_model(self, directory = "../../save/", filename = "random_forest_model"):
        joblib.dump(self.model, directory + filename + ".pkl")
    
    def save_time(self, start_time, filename="../../result/time_data.txt"):
        class_name = self.__class__.__name__
        current_time = time() - start_time

        # Check if class_name is already in the file
        found = False
        with open(filename, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if class_name in line:
                    found = True
                    # Update the time for the existing class_name
                    lines[i] = f"{class_name}     {current_time}\n"
                    break

        # If class_name is not found, append the new data
        if not found:
            lines.append(f"{class_name}     {current_time}\n")

        # Write the updated data to the file
        with open(filename, "w") as f:
            f.writelines(lines)

    def save_accuracy(self, filename="../../result/accuracy_data.txt", max_cut=None):
        class_name = self.__class__.__name__

        # Get the testing accuracy
        if max_cut is None:
            # Predict on the training data
            predictions = self.predict(self.X_train)
            testing_accuracy = self.score(self.y_train, predictions)

            # Get the validation accuracy
            validation_accuracy = self.accuracy
        else:
            # Predict on the training data with max_cut
            predictions = self.predict(self.X_train[:max_cut])
            testing_accuracy = self.score(self.y_train[:max_cut], predictions)

            # Get the validation accuracy with max_cut
            val_predictions = self.predict(self.X_test[:max_cut])
            validation_accuracy = self.score(self.y_test[:max_cut], val_predictions)

        # Check if class_name is already in the file
        found = False
        with open(filename, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if class_name in line:
                    found = True
                    # Update the accuracy for the existing class_name
                    lines[i] = f"{class_name}     {testing_accuracy:.4f}     {validation_accuracy:.4f}     {str(max_cut)}\n"
                    break

        # If class_name is not found, append the new data
        if not found:
            lines.append(f"{class_name}     {testing_accuracy:.4f}     {validation_accuracy:.4f}     {str(max_cut)}\n")

        # Write the updated data to the file
        with open(filename, "w") as f:
            f.writelines(lines)

            

        

if __name__ == '__main__':
    from sys import path
    path.append("../../utils/")
    import pre_processing
    path.append("../")
    
    from pre_processing import DataProcessor
    dataset_df = pd.read_csv("../../Data/train.csv")
    dataset_df = DataProcessor(dataset_df)

    X_train, X_test, y_train, y_test = dataset_df.run_and_split()

    start_time = time()
    rf_classifier = RandomForestClassifierCustom(X_train, X_test, y_train, y_test)
    rf_classifier.best_fit()
    
    rf_classifier.print_best_params()

    rf_classifier.save_model()
    rf_classifier.save_time(start_time)
    rf_classifier.save_accuracy()
