# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 03:39:29 2022

@author: hlinl
"""


import copy
import glob
import os
import math

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mig
import PIL

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from PARAM import *


class AcousticSpectrumDefectDataset(Dataset):
    """
    """

    def __init__(self, spectrum_dir, spectrum_extension=SPECTRUM_EXTENSION, dtype=torch.float32, 
                 img_transform=Compose([Resize(IMG_SIZE), ToTensor()])):
        """
        """

        self.spectrum_dir = spectrum_dir
        self.spectrum_extension = spectrum_extension
        self.dtype = dtype
        self.img_transform = img_transform

        self._dataset = defaultdict()
        self._dataset_size = 0
    

    def __len__(self):
        """
        """

        return self._dataset_size

    
    def __getitem__(self, id):
        """
        """

        
    def _set_data_repo(self):
        """
        """

        spectrum_folder_list = os.listdir(self.spectrum_dir)

        for spectrum_folder in spectrum_folder_list:
            spectrum_folder_path = os.path.join(self.spectrum_dir, spectrum_folder)
            spectrum_path_list_temp = glob.glob(os.path.join(spectrum_folder_path, "*.{}".format(self.spectrum_extension)))

            # Establish image label dict. 
            for i, path in enumerate(spectrum_path_list_temp):
                self._dataset[str(int(input_image_accum_num+i))] = [input_image_subfolder_name_temp, i, path]

            input_image_accum_num += len(input_image_path_list_temp)


class FrameAutoencoderDataset(Dataset):
    """
    """

    def __init__(self, input_data_dir, output_data_dir, img_pattern=IMG.IMAGE_EXTENSION, dtype=torch.float32, 
                 input_image_transform=Compose([Resize(ML_VAE.INPUT_IMAGE_SIZE), ToTensor()]), 
                 output_image_transform=Compose([Resize(ML_VAE.OUTPUT_IMAGE_SIZE), ToTensor()])):
        """
        Expected data directory structure: folder (data_dir) -> subfolders (layers) -> images (frames). 
        """

        self.input_data_dir = input_data_dir # The total directory of all input images for the autoencoder. 
        self.output_data_dir = output_data_dir # The total directory of all output images for the autoencoder. 
        self.img_pattern = img_pattern
        self.image_dtype = dtype
        self.input_image_transform = input_image_transform
        self.output_image_transform = output_image_transform
        
        # Data repos (path lists) & Data label dictionaries. 
        self._dataset_size = 0 
        self._input_image_label_dict = {} # {`ind`->int: [`subfolder`->str, `image_No`->int, `image_path`-> str]}. 
        self._output_image_label_dict = {} # {`ind`->int: [`subfolder`->str, `image_No`->int, `image_path`-> str]}. 

        self._set_data_repo()
    

    def __len__(self):
        """
        """

        return self._dataset_size


    def __getitem__(self, index):
        """
        """

        index = copy.deepcopy(str(int(index)))

        # input_img = PIL.Image.open(self._input_image_label_dict[index][2])
        # output_img = PIL.Image.open(self._output_image_label_dict[index][2])

        input_img = PIL.Image.fromarray(np.uint8(mig.imread(self._input_image_label_dict[index][2])*255))
        output_img = PIL.Image.fromarray(np.uint8(mig.imread(self._output_image_label_dict[index][2])*255))

        if self.input_image_transform:
            input_img = copy.deepcopy(self.input_image_transform(input_img).to(self.image_dtype)) # Transformed tensor of prescribed data type. [c, h, w]. 
        if self.output_image_transform:
            output_img = copy.deepcopy(self.output_image_transform(output_img).to(self.image_dtype)) # Transformed tensor of prescribed data type. [c, h, w]. 

        # # ---------- Reshape tensors to make them compatible with NN ---------- 
        # input_img_c, input_img_h, input_img_w = input_img.size()
        # output_img_c, output_img_h, output_img_w = output_img.size()

        # input_img = input_img.view(-1, input_img_c, input_img_h, input_img_w)
        # output_img = output_img.view(-1, output_img_c, output_img_h, output_img_w)

        sample = {'input': input_img, 'output': output_img}

        return sample
    

    @property
    def input_data_repo_dict(self):
        """
        """

        return self._input_image_label_dict
    

    @property
    def output_data_repo_dict(self):
        """
        """

        return self._output_image_label_dict
    

    def _set_data_repo(self):
        """
        """

        # ---------- Set input data repo ---------- 
        input_image_subfolder_list = os.listdir(self.input_data_dir)
        input_image_subdir_list = glob.glob(os.path.join(self.input_data_dir, "*"))
        input_image_accum_num = 0 # Used for establishing the image label dict. 

        for ind, input_image_subdir in enumerate(input_image_subdir_list):
            input_image_subfolder_name_temp = input_image_subfolder_list[ind]
            input_image_path_list_temp = copy.deepcopy(glob.glob(os.path.join(input_image_subdir, 
                                                                              "*.{}".format(self.img_pattern))))

            # Establish image label dict. 
            for i, path in enumerate(input_image_path_list_temp):
                self._input_image_label_dict[str(int(input_image_accum_num+i))] = [input_image_subfolder_name_temp, i, path]

            input_image_accum_num += len(input_image_path_list_temp)

        # ---------- Set output data repo ---------- 
        output_image_subfolder_list = os.listdir(self.output_data_dir)
        output_image_subdir_list = glob.glob(os.path.join(self.output_data_dir, "*"))
        output_image_accum_num = 0 # Used for establishing the image label dict. 

        for ind, output_image_subdir in enumerate(output_image_subdir_list):
            output_image_subfolder_name_temp = output_image_subfolder_list[ind]
            output_image_path_list_temp = copy.deepcopy(glob.glob(os.path.join(output_image_subdir, 
                                                                               "*.{}".format(self.img_pattern))))

            # Establish image label dict. 
            for i, path in enumerate(output_image_path_list_temp):
                self._output_image_label_dict[str(int(output_image_accum_num+i))] = [output_image_subfolder_name_temp, i, path]

            output_image_accum_num += len(output_image_path_list_temp)
        
        self._dataset_size = min(input_image_accum_num, output_image_accum_num)


    def extract(self, ind_array):
        """
        Return info of input and output data that corresponds to the given `ind_array` with the exact order. 
        """

        ind_array = copy.deepcopy(ind_array.astype(str).reshape(-1))
        
        # Input. 
        input_image_subfolder_list = [self._input_image_label_dict[ind][0] for ind in ind_array]
        input_image_ind_list = [self._input_image_label_dict[ind][1] for ind in ind_array]
        input_image_path_list = [self._input_image_label_dict[ind][2] for ind in ind_array]

        # Output. 
        output_image_subfolder_list = [self._output_image_label_dict[ind][0] for ind in ind_array]
        output_image_ind_list = [self._output_image_label_dict[ind][1] for ind in ind_array]
        output_image_path_list = [self._output_image_label_dict[ind][2] for ind in ind_array]

        return (input_image_subfolder_list, input_image_ind_list, input_image_path_list,
                output_image_subfolder_list, output_image_ind_list, output_image_path_list)



if __name__ == "__main__":
    """
    """

    pass