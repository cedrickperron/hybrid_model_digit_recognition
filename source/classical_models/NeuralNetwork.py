import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers
from sklearn.metrics import accuracy_score
import pickle
from time import time
# ------------------------------------------------------------ #
import warnings
from tensorflow.keras.regularizers import L1, L2, L1L2



# Ignore all warnings
warnings.filterwarnings("ignore")
# ------------------------------------------------------------ #



def default_layer(X_train, m, regularizer=None, dropout_rate = 0.2):
    """
    Creates a default layer
    """
    input_shape = X_train.shape[1:] + (1,)
    return [
                # 1
                layers.Conv2D(16, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape=input_shape), # 16 filters, output: 26x26x16
                layers.Dropout(rate=dropout_rate),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)), # Max pooling, output: 14x14x16
                # 2
                layers.Conv2D(32, kernel_size = (3, 3), padding = "same", activation = 'relu'),
                layers.Dropout(rate=dropout_rate),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2,2)),
                # Flatten, Connected
                layers.Flatten(),
                layers.Dense(m, activation = 'relu', kernel_regularizer = regularizer),
                layers.Dropout(rate=dropout_rate),
                layers.BatchNormalization(),
                ## Intermediate Layers
                layers.Dense(m, activation = 'relu', kernel_regularizer = None),
                layers.Dropout(dropout_rate),
                layers.BatchNormalization(),
                # Final Layer
                layers.Dense(10, activation="softmax")
            ]


class NeuralNetworkClassifier:
    def __init__(self, X_train, X_test, y_train, y_test, model = None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.accuracy = None
        self.params = None
        self.model = model
        self.history = None

        self.callbacks = keras.callbacks.EarlyStopping(
                monitor = "val_accuracy",
                patience=50, # Specifies the number of epoch (number of forward steps) with no improvement after which the training stop (So if the validation percentage does not increase for 20 consecutive epochs, it's going to stop the training)
                min_delta=0.001, # This is the minimum change in validation loss (speaks about ability of the system to predict new data) that is required for an epoch to be consider successful
                restore_best_weights=True, # If the training is stopped, we take the best weights (the ones that lead to the lowest validation loss) of all the analysis we did
                mode = "max"
                )


    def add_layer(self, layers):
        """ Create a layer to the Neural Network

        layers - a list of Keras.layer
        """
        if self.model is None:
            self.model = Sequential()
        for layer in layers:
            self.model.add(layer)
        return self.model

    def create_default_model(self, *args, **kwargs):
        """
        Create the default model
        """
        layer = default_layer(self.X_train, *args, **kwargs)
        self.model = self.add_layer(layer)
        return self.model
    
        
    def fit(self, epochs = 2, batch_size = 512, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07):
        """
        Fits the model with the training dataset.
        """
        if self.model is None:
            raise ValueError("Model is None")
        opt = tf.keras.optimizers.Adam(learning_rate= learning_rate, beta_1=beta_1, beta_2 =  beta_2, epsilon =  epsilon) 
        self.model.compile(optimizer = opt, loss='categorical_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), batch_size=batch_size,
                       epochs=epochs, callbacks=self.callbacks, verbose=2)

    def predict(self, X_test=None): 
        """
        Predicts the target values for the input data.

        Inputs:
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
        return np.argmax(self.model.predict(X_test), axis=1)

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
        return accuracy_score(np.argmax(y_true, axis=1), y_pred)

    def default_model_best_fit(self, units_list = [20, 30, 40],
                 regularizers_list=[None, keras.regularizers.l1_l2(l1=0.01, l2=0.01), keras.regularizers.l1(0.01), keras.regularizers.l2(0.01)], dropout_rate=0.3,  batch_size = 512):
        """
        Finds the best model based on accuracy score.

        Inputs:
        units_list: list of list of int
            List of hidden layer unit configurations.
        regularizers_list: list of keras.regularizers.Regularizer or None, optional (default=[None])
            List of regularizers to try.
        dropout_rate: float, optional (default=0.3)
            Dropout rate.

        Returns:
        None
        """
        max_accuracy = 0.0
        best_params = None
        best_model = None
        
        for units in units_list:
            for regularizer in regularizers_list:
                model = self.create_default_model(units, regularizer, dropout_rate)
                self.model = model
                self.fit()
                predictions = self.predict()
                accuracy = self.score(self.y_test, predictions)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    best_params = {'units': units, 'regularizer': regularizer, 'dropout_rate': dropout_rate}
                    best_model = self.model

        self.accuracy = max_accuracy
        self.params = best_params
        self.model = best_model


    def print_best_params(self):
        """
        Prints the best hyperparameters found.
        """
        print(f"Best parameters - Units: {self.params['units']}, Regularizer: {self.params['regularizer']}, Dropout Rate: {self.params['dropout_rate']}")
        print(f"Validation accuracy for Neural Network: {self.accuracy:.3f}")

    def summary(self):
        """
        Print the model summary
        """
        print(self.model.summary())

    def get_training_metrics(self, plot = False):
        """
        Returns the training and validation loss, and accuracy from the training history.
        """
        if self.history is None:
            raise ValueError("History is not initialized. Fit the model first.")
        qm_loss = self.history.history['loss']
        qm_val_loss = self.history.history['val_loss']
        qm_binary_accuracy = self.history.history['accuracy']
        qm_val_binary_accuracy = self.history.history['val_accuracy']

        if plot == True:
            qm_history_df = pd.DataFrame(qm_fitting)
            qm_history_df[['loss', 'val_loss']].plot()
            qm_history_df[['accuracy', 'val_accuracy']].plot()
            qm_loss, qm_accuracy = qm_model.evaluate(test_images, test_labels)
            print()
            print(f"Validation Loss: {qm_loss}, Validation Accuracy: {qm_accuracy}")


        return qm_loss, qm_val_loss, qm_binary_accuracy, qm_val_binary_accuracy

    def save_model(self, filename = "nn_classifier_model", directory = "../../save/"):
        self.model.save(directory + filename + ".keras")

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

    

    def load_model(self, model_file="../../save/nn_classifier_model.keras"):
        self.model = tf.keras.models.load_model(model_file)

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
            if self.accuracy is None:
                val_predictions = self.predict(self.X_test[:max_cut])
                validation_accuracy = self.score(self.y_test[:max_cut], val_predictions)
            else:
                 validation_accuracy = self.accuracy

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

'''
## Need to pip install scikeras /// from scikeras.wrappers import KerasClassifier
    def default_model_best_fit(self, epochs = 25, units_list=[128, 256, 512],
                                regularizers_list=[None, keras.regularizers.l1_l2(l1=0.01, l2=0.01), keras.regularizers.l1(0.01), keras.regularizers.l2(0.01)], 
                                dropout_rate=0.3, batch_size=512):
        """
        Finds the best model based on accuracy score using RandomizedSearchCV.

        Inputs:
        units_list: list of list of int
            List of hidden layer unit configurations.
        regularizers_list: list of keras.regularizers.Regularizer or None, optional (default=[None])
            List of regularizers to try.
        dropout_rate: float, optional (default=0.3)
            Dropout rate.

        Returns:
        None
        """
        param_dist = {
            'units': units_list,
            'regularizer': regularizers_list,
            'dropout_rate': [dropout_rate],
        }

        model = KerasClassifier(build_fn=self.create_default_model, epochs=epochs, batch_size=batch_size, verbose=2)
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=3, n_jobs=-1)
        random_search.fit(self.X_train, self.y_train)

        self.accuracy = random_search.best_score_
        self.params = random_search.best_params_
        self.model = random_search.best_estimator_.model
'''





if __name__ == '__main__':
    from sys import path
    path.append("../../utils/")
    import pre_processing
    path.append("../")
    
    from pre_processing import Dataset
    data = Dataset()
    data.default()
    #
    n_train = 5000; n_test = 1000
    s_t = time()
    nn_classifier = NeuralNetworkClassifier(data.train_images[:n_train], data.test_images[:n_test], data.train_labels[:n_train], data.test_labels[:n_test])
    print(np.argmax(nn_classifier.y_test[:2], axis=1))
    
    print(time()-s_t)
    s_t = time()
    nn_classifier.default_model_best_fit(units_list = [30], regularizers_list=[None])
    print(time()-s_t)
    nn_classifier.print_best_params()


    # Example of saving and loading the model
    nn_classifier.save_model()
    nn_classifier.save_time(s_t)
    nn_classifier.save_accuracy()

