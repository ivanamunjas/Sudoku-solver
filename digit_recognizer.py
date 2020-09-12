# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:13:42 2020

@author: ASUS
"""

import numpy as np
from tensorflow.keras.models import load_model


def prepare_images(test_images, normalise=True, flatten=True):
    if normalise:
        test_images = [img / 255 for img in test_images]

    if flatten:
        test_images = [img.flatten() for img in test_images]

    # Convert to a Numpy array if necessary
    if type(test_images) == list:
        test_images = np.array(test_images)

    return test_images


class DigitRecogniser:
    def __init__(self, model):
        """
        Loads the saved model and uses that to predict digits based on input images.
        Args:
            model (str): Path to checkpoint file for the neural network for digit recognition.
        """
        self.model = model

    def predict_digit(self, test_images, normalise=True, flatten=True):
        """
        Predicts digits from an `np.array` of test_images.
        Args:
            test_images (np.array): Array of test images to predict on. Expects 28x28 size.
            normalise (bool): Normalises the pixel values between 0 and 1.
            flatten (bool): Flattens each image so they are of shape (784) instead of (28, 28)
        Returns:
            np.array: One-dimensional array of predictions for each image provided.
        """
        test_images = prepare_images(test_images, normalise, flatten)
        new_model = load_model(self.model)
        
        y_pred = new_model.predict(np.reshape(test_images, (test_images.shape[0], 28, 28, 1)))
        
        out = []
        for i in range(y_pred.shape[0]):
            out.append(np.argmax(y_pred[i]))

        return out