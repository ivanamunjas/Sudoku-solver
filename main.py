# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:03:40 2020

@author: ASUS
"""

from Sudoku import Sudoku
import os

MODEL_DIR = os.path.join('data','saved_model')
TEST_IMG_DIR = os.path.join('data//images//grid','test')

example = Sudoku(img_path=TEST_IMG_DIR+'\\9.jpg', model_path=MODEL_DIR)
example.show_completed()