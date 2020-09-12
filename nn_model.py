# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:10:35 2020

@author: ASUS
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train(data, model_path, epochs=20, batch_size=32, check_acc=True):
        
    x_train, y_train = data.train.images, data.train.labels
    x_test, y_test = data.test.images, data.test.labels
    
    
    X_train = np.zeros((len(x_train), 28, 28, 1))
    
    for i, im in enumerate(x_train):
        X_train[i] = np.reshape(im, (28,28,1))
        
    X_test = np.zeros((len(x_test), 28, 28,1))
    
    for i, im in enumerate(x_test):
        X_test[i] = np.reshape(im, (28,28,1))
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


    # checkpoint_path = "training_1/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # cp_callback = ModelCheckpoint(filepath=checkpoint_path,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)
    
    history = model.fit(X_train, y_train, 
                        epochs=epochs, batch_size=batch_size, verbose=True, shuffle=True, 
                        validation_data=(X_val, y_val))
                        # validation_data=(X_val, y_val), callbacks=[cp_callback])
    
    model.save(model_path)

    def accuracies(model, x, y):
        y_pred = model.predict(x, batch_size=32, verbose=0)
        print('Accuracy: ', accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1)))
        
    if check_acc:
        print('Training')
        accuracies(model, X_train, y_train)
        print()
        print('Validation')
        accuracies(model, X_val, y_val)
        print()
        print('Test')
        accuracies(model, X_test, y_test)