import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import joblib
from time import time

class KNNClassifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.model = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test =y_test
        self.k_scores_test = []
        self.accuracy = None
        self.params = None
        
    def fit(self, n_neighbors, weight):
        """
        Fits the model with the training dataset.

        Parameters:
        X_train: array-like, shape (n_samples, n_features)
            Training data.
        y_train: array-like, shape (n_samples,)
            Target values.
        n_neighbors: int
            Number of neighbors to use.
        weights: str
            Weight function used in prediction.
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight)
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test= None):
        """
        Predicts the target values for the input data.

        Parameters:
        X_test: array-like, shape (n_samples, n_features)
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
        Takes as input the true values of y and the predicted values of y and returns the accuracy score
        """
        
        return accuracy_score(y_true, y_pred)

    def best_fit(self, max_neighbors=20, weights=['uniform', 'distance']):
        """
        Finds the best model based on accuracy score.

        Parameters:
        max_neighbors: int, optional (default=100)
            Maximum number of neighbors to consider.
        weights: list of str, optional (default=['uniform', 'distance'])
            Weight function used in prediction.

        Returns:
        sklearn.neighbors.KNeighborsClassifier
            Best trained model.
        int
            Number of neighbors for the best model.
        float
            Accuracy score of the best model.
        """
        best_score = 0
        best_k = 0
        best_model = None
        for weight in weights:
            for k in range(1, max_neighbors + 1):
                self.fit(k, weight)
                knn_predictions_test = self.predict()
                score = accuracy_score(self.y_test, knn_predictions_test)
                self.k_scores_test.append(score)
                if score > best_score:
                    best_score = score
                    best_params = [weight, k]
                    best_model = self.model

        self.model = best_model
        self.params = best_params
        self.accuracy = best_score


    def parameter_plot(self):
        """
        Plots accuracy scores for different values of k.

        Parameters:
        None

        Returns:
        None
        """
        k_list = list(range(1, len(self.k_scores_test) // 2 + 1))
        plt.plot(k_list, self.k_scores_test[:len(k_list)], label='Uniform Weight')
        plt.plot(k_list, self.k_scores_test[len(k_list):], label='Distance Weight')
        plt.xlabel("k")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    def print_best_params(self):
        print(f"Validation accuracy for KNN: {self.accuracy:.3f}")
        print(f"Parameters of KNN: {self.params}")

    def save_model(self, directory = "../../save/"):
        joblib.dump(self.model, directory + "knn_model.pkl")

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
    
    from pre_processing import Dataset
    data = Dataset()
    data.load_dataset()
    data.shuffle()
    data.flatten((28*28, ))
    #print(len(data.train_images), len(data.test_images))
    n_train = 5000; n_test = 2000
    s_t = time()
    knn_classifier = KNNClassifier(data.train_images[:n_train], data.test_images[:n_test], data.train_labels[:n_train], data.test_labels[:n_test])
    knn_classifier.best_fit()
    knn_classifier.print_best_params()
    
    knn_classifier.parameter_plot()

    knn_classifier.save_model()
    knn_classifier.save_time(s_t)
    knn_classifier.save_accuracy()
    
    
    

    
