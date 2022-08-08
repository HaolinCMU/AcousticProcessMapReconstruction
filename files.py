# -*- coding: utf-8 -*-
"""
Created on Sat May 21 03:26:25 2022

@author: hlinl
"""


import os
import glob
import copy
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mig

from PARAM import *


def clr_type(directory, file_extension):
    """
    """
    
    file_path_list = glob.glob(os.path.join(directory, "*.{}".format(file_extension)))
    for file_path in file_path_list: os.remove(file_path)


def clr_dir(directory):
    """
    """
    
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if os.path.isdir(path): 
            shutil.rmtree(path)
        else: os.remove(path)

    
def rename(src, dst):
    """
    """

    os.rename(src, dst)