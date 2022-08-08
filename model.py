# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 04:27:17 2022

@author: hlinl
"""


import os
import glob
import sys
import time

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt

from collections import defaultdict
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

from PARAM import *


class CNN_2D(nn.Module):
    """
    """

    def __init__(self, in_channel_num, output_dim, conv_hidden_layer_num, mlp_hidden_layer_num):
        """
        Conv block first, then MLP block. 
        """
        
        super(CNN_2D, self).__init__()
        self.in_channel_num = in_channel_num
        self.output_dim = output_dim
        self.conv_hidden_layer_num = conv_hidden_layer_num
        self.mlp_hidden_layer_num = mlp_hidden_layer_num
        
        self._hidden_architecture = None

    

    def forward(self, x):
        """
        """

        pass


    def _img_dim_conv(self):
        """
        """

    
    def architecture(self):
        """
        """

        architecture_pre_module = []

        # Conv. block. 
        for i in range(self.conv_hidden_layer_num):
            if i == 0: 
                in_channel_num_temp = self.in_channel_num
                out_channel_num_temp = FIRST_CONV_CHANNEL_NUM
            else: 
                in_channel_num_temp = copy.deepcopy(out_channel_num_temp)
                out_channel_num_temp = copy.deepcopy(int(in_channel_num_temp*2))
            
            architecture_pre_module.append(nn.Conv2d(in_channel_num_temp, out_channel_num_temp, kernel_size=CONV_KERNEL_SIZE, 
                                                     stride=CONV_STRIDE_SIZE, padding=CONV_PADDING_SIZE))
            # architecture_pre_module.append(nn.Dropout(0.5))
            architecture_pre_module.append(nn.BatchNorm2d(out_channel_num_temp))
            architecture_pre_module.append(ACTIVATION_LAYER)
            architecture_pre_module.append(POOLING_LAYER(POOLING_KERNEL_SIZE, stride=POOLING_STRIDE_SIZE, 
                                                         padding=POOLING_PADDING_SIZE))
            
        out_channel_num_temp * IMG_SIZE


            
            hidden_block_temp = nn.Sequential(nn.Conv2d(in_channel_num_temp, out_channel_num_temp, kernel_size=CONV_KERNEL_SIZE, 
                                                        stride=CONV_STRIDE_SIZE, padding=CONV_PADDING_SIZE), 
                                            #   nn.Dropout(0.5), 
                                              nn.BatchNorm2d(out_channel_num_temp), 
                                              ACTIVATION_LAYER,
                                              )
            pooling_layer = POOLING_LAYER(POOLING_KERNEL_SIZE, stride=POOLING_STRIDE_SIZE)

            architecture_pre_module.append(hidden_block_temp)
            architecture_pre_module.append(pooling_layer)
        
        

        
        
