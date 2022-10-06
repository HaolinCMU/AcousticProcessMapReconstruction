# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:40:13 2022

@author: hlinl
"""


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics
import torch
import torch.nn as nn
import torchvision
import torch.utils

from torchvision.models.resnet import model_urls as resnet_urls  # workaround for SSL error
from torchvision.models.densenet import model_urls as densenet_urls  # workaround for SSL error


class ResNet(object):
    """
    """

    def __init__(self, model_name, pretrained=True, out_channel=1):
        """
        """

        self.model_name = model_name
        self.pretrained = pretrained
        self.out_channel = out_channel

        self.resnet = None
        self.cuda_avail = torch.cuda.is_available()

        self.models = {'resnet18': torchvision.models.resnet18,
                       'resnet34': torchvision.models.resnet34
                       }

        self.init_resnet_model()
    

    def init_resnet_model(self):
        """
        """

        if self.model_name not in self.models.keys(): 
            raise ValueError("ResNet model name invalid. ")
        
        self.resnet = self.models[self.model_name](pretrained=self.pretrained)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, self.out_channel)  # changing output dimensions

