# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 20:41:19 2022

@author: hlinl
"""


import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F


ACOUSTIC_DATA_DIR = "acoustic"
PHOTODIODE_DATA_DIR = "photodiode"
PROCESS_PARAM_SHEET_PATH = "process_param.csv" # [ V | P ]. 

ACOUSTIC_DATA_EXTENSION = "npy"
PHOTODIODE_DATA_EXTENSION = "npy"

PROCESSED_DATA_DIR = "data_processed"
PROCESSED_DATA_CLIPS_SUBDIR = "clips"
PROCESSED_DATA_SPECTROGRAMS_SUBDIR = "spectrum"

AUDIO_SENSOR_NO = 0 # The chosen acoustic sensor of interest. Options: [0, 1, 2]. 
PHOTODIODE_SENSOR_NO = 0 # The chosen photodiode sensor of interest. Options: [0, 1]. 
INITIAL_LAYER_NUM = 17 # The number of beginning layers of the print. 
TRANSITIONAL_LAYER_NUM = 3 # The number of transitional layers in between two consecutive layers of interest. 
REPITITION_TIME = 3 # Number of times repeated. 

# MOVING_INTERVAL_WIDTH = 100 # 50 for a break. 
PHOTO_THRSLD = 0.05

SAMPLING_RATE = 100e3 # Sampling frequency of acoustic signals.
CLIP_DURATION = 2.0 # Duration of a single clip. Unit: s. Default: 2.0
CLIP_STRIDE_INPERC = 0.25 # Stride of clip sampling represented in percentage of the clip length. 

# Spectrogram. 
SPECTRUM_MODE = 'wavelet'

# Wavelet transform
WAVELET = 'morl' # Default: Morlet ('morl'). 'mexh'. 
SCALES = np.arange(1, 31) # 1D Array of Int. The scale of the wavelet. 
IS_LOG_YSCALE = False
SPECTRUM_DPI = 1200 # Default: 256. Intend to generate an image with size 256x256. 
SPECTRUM_EXTENSION = "png" # File format of the spectrogram image. 

# Model training. 
TRAIN_RATIO = 0.8
VALID_RATIO = 0.05
TEST_RATIO = 0.15

# ML Param. 
IMG_SIZE = 256 # Assume square image. 
IN_CHANNEL_NUM = 1
FIRST_CONV_CHANNEL_NUM = 16
OUTPUT_DIM = 4

CONV_KERNEL_SIZE = (3,3)
CONV_STRIDE_SIZE = (1,1)
CONV_PADDING_SIZE = (1,1)

POOLING_KERNEL_SIZE = (2,2)
POOLING_STRIDE_SIZE = (2,2)
POOLING_PADDING_SIZE = 0

CONV_HIDDEN_LAYER_NUM = 5
MLP_HIDDEN_LAYER_NUM = 2

ACTIVATION_LAYER = nn.ReLU()
POOLING_LAYER = nn.MaxPool2d(POOLING_KERNEL_SIZE)

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCH_NUM = 50
