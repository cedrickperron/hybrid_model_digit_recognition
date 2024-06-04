import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib
from time import time

class LogisticRegressionClassifier:
    def __init__(self, X_train, X_test, y_train, y_test, model = None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.accuracy = None
        self.params = None
    
    def fit(self, C, solver):
        """
        Fits the model with the training dataset.

        Parameters:
        C: float
            Inverse of regularization strength; smaller values specify stronger regularization.
        solver: str
            Algorithm to use in the optimization problem.

        Returns:
        None
        """
        self.model = LogisticRegression(C=C, random_state=1, max_iter=150, solver=solver)
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
    def best_fit(self, hyper_params=np.arange(0.25, 5, 0.25), solver=["newton-cg", "lbfgs", "liblinear", 'sag', "saga"]):
        """
        Finds the best model based on accuracy score.

        Parameters:
        hyper_params: array-like, optional (default=None)
            Values for regularization strength.
        solver: list of str, optional (default=None)
            Algorithms to use in the optimization problem.

        Returns:
        sklearn.linear_model.LogisticRegression
            Best trained model.
        float
            Regularization strength for the best model.
        float
            Accuracy score of the best model.
        """
        max_accuracy = 0.0
        max_acc_params = None
        best_model =  None
        for s in solver:
            for C in hyper_params:
                self.fit(C, s)
                LR_Model_predictions_test = self.predict()
                LR_Model_accuracy_test = self.score(self.y_test, LR_Model_predictions_test)
                if LR_Model_accuracy_test > max_accuracy:
                    max_accuracy = LR_Model_accuracy_test
                    max_acc_params = (s, C)
                    best_model = self.model
        self.accuracy = max_accuracy
        self.params = max_acc_params
        self.model = best_model
        return self.model

    def parameter_plot(self):
        """
        Plots accuracy scores for different values of regularization strength.

        Parameters:
        None

        Returns:
        None
        """
        pass
    
    def print_best_params(self):
        print(f"Validation accuracy for Logistic Regression: {self.accuracy:.3f}")
        print(f"Parameters of Logistic Regression: {self.params}")

    def save_model(self, directory = "../../save/"):
        joblib.dump(self.model, directory + "logistic_regression_model.pkl")

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
            val_predictions = self.predict(self.X_test)
            validation_accuracy = self.score(self.y_test, val_predictions)
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

    X_train, X_test, y_train, y_test = dataset_df.run_and_split() # Run the standard list of operations

    s_t = time()
    lr_classifier = LogisticRegressionClassifier(X_train, X_test, y_train, y_test)
    lr_classifier.best_fit()

    lr_classifier.print_best_params()

    lr_classifier.save_model()
    lr_classifier.save_time(s_t)
    lr_classifier.save_accuracy()

