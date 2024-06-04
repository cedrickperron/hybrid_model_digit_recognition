import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



class Dataset:
    def __init__(self, seed = 2, train_images  = None, train_labels  = None , test_images  = None, test_labels  = None):
        self.seed = seed
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

    def load_dataset(self):
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

        

    def shuffle(self, seed= None):
        if seed is None:
            seed = self.seed

        np.random.seed(seed)
        perm = np.random.permutation(len(self.train_images))
        
        self.train_images = self.train_images[perm] / 255.00
        self.train_labels = self.train_labels[perm]

        perm = np.random.permutation(len(self.test_images))

        self.test_images = self.test_images[perm] / 255.00
        self.test_labels = self.test_labels[perm]

    def one_hot_transform_labels(self):
        self.train_labels = to_categorical(self.train_labels, num_classes = 0)
        self.test_labels = to_categorical(self.test_labels, num_classes = 0)

    def display_images(self, images, labels, N = 10):
            
            cols = 5
            rows = math.ceil(N / cols)

            fig, axes = plt.subplots(rows, cols, figsize= (15, 3*rows))
            axes = axes.ravel()

            for i in range(N):
                axes[i].imshow(images[i], cmap="gray")
                axes[i].set_title(f"Pred: {labels[i]}")
                axes[i].axis("off") # Hides axes ticks


            for j in range(N, rows * cols):
                axes[j].axis("off")

            plt.tight_layout()
            plt.show()

    def to_numpy(self):
        self.train_images = np.array(self.train_images)
        self.train_labels = np.array(self.train_labels)
        self.test_images = np.array(self.test_images)
        self.test_labels = np.array(self.test_labels)
    
    def flatten(self, new_shape):
        self.train_images = self.train_images.reshape(-1, *new_shape)
        self.test_images = self.test_images.reshape(-1, *new_shape)
    
    def default(self):
        self.load_dataset()
        self.shuffle()
        self.one_hot_transform_labels()
        

if __name__ == '__main__':
    dataset = Dataset()
    dataset.load_dataset()
    dataset.shuffle()
    dataset.one_hot_transform_labels()
    dataset.to_numpy()
    dataset.display_images(dataset.train_images, dataset.train_labels.argmax(axis=1), 10)
    
    

                                  
                
                                    

        
