# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:08:08 2020

@author: ASUS
"""

import os
import cv2
import pickle
import numpy as np
from shutil import copyfile

import solver
import cv_helper as cv
from Dataset import Dataset
# from Sudoku import classification_mode
from Sudoku import Sudoku
import nn_model as nn 

# Static variables, mostly file locations
DIGITS = '0123456789'
IMAGE_DIR = os.path.join('data', 'images')
STAGE_DIR = os.path.abspath(os.path.join(IMAGE_DIR, 'stage'))
CLASSIFIED_DIR = os.path.abspath(os.path.join(IMAGE_DIR, 'classified', 'raw'))
CL_TRAIN_DIR = os.path.join(CLASSIFIED_DIR, 'train')
CL_TEST_DIR = os.path.join(CLASSIFIED_DIR, 'test')
DIGITS_DIR = os.path.abspath(os.path.join(IMAGE_DIR, 'raw', 'digits'))
GRID_DIR = os.path.abspath(os.path.join(IMAGE_DIR, 'grid', 'all'))
TRAIN_DIR = os.path.abspath(os.path.join(IMAGE_DIR, 'grid', 'train'))
TEST_DIR = os.path.abspath(os.path.join(IMAGE_DIR, 'grid', 'test'))
DATA_DIR = os.path.abspath(os.path.join('data', 'datasets'))
DATA_FILE = os.path.join(DATA_DIR, 'raw')
MODEL_DIR = os.path.join('data/saved_model')


def mkdir(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)
    return dir_


def get_next_file(name, digit):
    """Gets the maximum file number in the directory, assuming all filenames are numeric (except for the extension."""
    fs = os.listdir(os.path.join(CLASSIFIED_DIR, name, str(digit)))

    # Ignore any hidden files in the directory
    fs = list(filter(lambda x: not x.startswith('.'), fs))

    if len(fs) > 0:
        return max([int(os.path.basename(f).split('.')[0]) for f in fs]) + 1
    else:
        return 0


def prepare_data():

    mkdir(os.path.join(IMAGE_DIR, 'classified'))
    mkdir(CLASSIFIED_DIR)

    blank = cv.create_blank_image(28, 28, grayscale=True, include_gray_channel=True)

    def classify_digits(name, src):
        print('%s Classification' % name)
        # Some housekeeping
        mkdir(os.path.join(CLASSIFIED_DIR, name))

        for i in range(10):
            digit_dir = os.path.join(CLASSIFIED_DIR, name, str(i))
            mkdir(digit_dir)

        # Sort files by their number otherwise we'll run into problems when classifying the digits
        files = [f for f in os.listdir(src) if f.split('.')[1] == 'jpg']
        files = sorted(files, key=lambda x: int(x.split('.')[0]))
        for i, f in enumerate(files):
            print('Classifying %s...' % i)
            original = [v.replace('.', '0') for k, v in read_original_board(i, src).items()]
            grid = Sudoku(os.path.join(src, f), include_gray_channel=True, skip_recog=True)

            # Ignore completely blank images, not required in the training set
            digits_idx = [(j, digit) for j, digit in enumerate(grid.digits) if not np.array_equal(digit, blank)]


            for j, digit in digits_idx:
                cv2.imwrite(os.path.join(CLASSIFIED_DIR, name, original[j], '%s.jpg' % get_next_file(name, original[j])),
                            digit)

    classify_digits('train', TRAIN_DIR)
    classify_digits('test', TEST_DIR)



def read_original_board(sudoku_id, dir_path=None, as_string=False, as_list=False):
    """Reads the .dat file with the original board layout recorded."""
    print('Read original board function')
    folder = GRID_DIR
    if dir_path is not None:
        folder = dir_path

    with open(os.path.join(folder, '%s.dat' % sudoku_id), 'r') as f:
        original = f.read()

    if as_string:
        return original
    elif as_list:
        return [v for k, v in solver.parse_sudoku_puzzle(original).items()]
    else:
        return solver.parse_sudoku_puzzle(original)


def random_train_test(num_train=90):
    mkdir(TRAIN_DIR)
    mkdir(TEST_DIR)

    grids = [f.split('.')[0] for f in os.listdir(GRID_DIR) if f.split('.')[1] == 'dat' and not f.startswith('.')]
    rand = np.random.permutation(len(grids))
    train_idx, test_idx = rand[:num_train], rand[num_train:]

    def copy_subset(indices, dir_path):
        for f in os.listdir(dir_path):  # Clear out current directory
            os.unlink(os.path.join(dir_path, f))

        for i, idx in enumerate(np.array(grids)[indices]):
            copyfile(os.path.join(GRID_DIR, '%s.jpg' % idx), os.path.join(dir_path, '%s.jpg' % i))
            copyfile(os.path.join(GRID_DIR, '%s.dat' % idx), os.path.join(dir_path, '%s.dat' % i))

    copy_subset(train_idx, TRAIN_DIR)
    copy_subset(test_idx, TEST_DIR)


def save_data(data, file_name):
    """Saves Python object to disk."""
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def load_data(file_name):
    """Loads Python object from disk."""
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def img_labs_from_dir(dir_path):
    """Gets image paths and labels from the classified image files in a directory."""
    images, labels = [], []
    for digit in DIGITS:
        digit_dir = os.path.join(dir_path, digit)
        files = os.listdir(digit_dir)
        files = list(filter(lambda x: not x.startswith('.'), files))
        files = sorted(files, key=lambda x: int(x.split('.')[0]))

        for file in files:
            images.append(os.path.join(digit_dir, file))
            labels.append(int(digit))
    return images, labels



def create_from_train_test(save=True):
    """Creates a dataset from already divided test and training images."""
    train_images, train_labels = img_labs_from_dir(CL_TRAIN_DIR)
    test_images, test_labels = img_labs_from_dir(CL_TEST_DIR)

    if save:
        mkdir(DATA_DIR)
        print('Compiling training and test images to %s...' % DATA_FILE)
        ds = Dataset((train_images, test_images), (train_labels, test_labels), from_path=True, split=False)
        save_data(ds, DATA_FILE)
    else:
        return train_images, train_labels, test_images, test_labels



def train(epochs=20, batch_size=32):
    """Begins training using the training and test data from `DATA_FILE`."""
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    nn.train(data, MODEL_DIR, epochs, batch_size)
