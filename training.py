# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:41:10 2020

@author: ASUS
"""
import helper

# Run before model training. Prepares the data for the next steps. 
helper.random_train_test(90)  # Randomly splits the total archive into a training and test.
helper.prepare_data()  # Classifies digit images using .dat descriptors without manipulation.
helper.create_from_train_test()  # Creates a pickled data file from classified digit images for training.

# Training the model
helper.train(epochs=20)