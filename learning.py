# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 20:36:09 2022

@author: hlinl
"""


import os
import gc
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

from numpy.lib.stride_tricks import sliding_window_view
from collections import defaultdict
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

from files import *
from PARAM import *
# from model import *
from audioBasics import *


class DP_APMR_Single(object):
    """
    """

    def __init__(self, acoustic_path, photodiode_path, process_param_vector, 
                 is_spectral_transform=False, spectrum_mode=SPECTRUM_MODE):
        """
        """

        self.acoustic_path = acoustic_path
        self.photodiode_path = photodiode_path
        self.process_param_vector = process_param_vector
        self.is_spectral_transform = is_spectral_transform # Default: False, to save memory. 
        self.spectrum_mode = spectrum_mode

        self._is_LOI, self._P, self._V, self._defect = self.process_param_vector

        self._audio_sensor_No = AUDIO_SENSOR_NO # Int. Indicate which aocustic sensor is chosen. 
        self._photodiode_sensor_No = PHOTODIODE_SENSOR_NO # Int. Indicate which photodiode sensor is chosen. 
        self._audio_sr = SAMPLING_RATE # Sampling frequency of acoustic signals.
        self._clip_length = int(CLIP_DURATION*SAMPLING_RATE) # Length (in pts) of a single clip. 
        self._clip_stride = int(CLIP_STRIDE_INPERC*CLIP_DURATION*SAMPLING_RATE) # Stride length (in pts) of clip sampling represented in percentage of the clip length.
        self._photodiode_sync_thrsld = PHOTO_THRSLD

        self._clips_num = 0
        self._clips_mat = None
        self._spectrum_list = []

        self._is_data_processed = False
        self._is_data_spectral_transformed = False

        self.processing(self.is_spectral_transform)
    

    @property
    def clips_num(self):
        """
        """

        if self._is_data_processed: return self._clips_num
        else: raise ValueError("Dataset not processed. ")


    @property
    def clips(self):
        """
        Return a matrix of clips. 
        """

        if self._is_data_processed and self._clips_mat is not None: return self._clips_mat
        else: raise ValueError("Dataset not processed. ")
    

    @property
    def spectrograms(self):
        """
        Return a list of spectrograms. 
        """

        if not self._is_data_processed: 
            raise ValueError("Dataset not processed. ")
        elif self._is_data_processed and not self._is_data_spectral_transformed: 
            raise ValueError("Spectrograms not generated. ")
        else: return self._spectrum_list


    @staticmethod
    def _synchronize(audio_sample, photodiode_sample, sync_threshold):
        """
        Use photodiode data to find the laser impinging point via a threshold in acoustic data. 
        Two 1D array input. Must have the same dimension. 
        """

        impinging_pt = np.where(photodiode_sample>=sync_threshold)[0][0]

        return audio_sample[impinging_pt:], photodiode_sample[impinging_pt:]
    

    @staticmethod
    def _audio_partition(data, clip_length_inDP, clip_stride_inDP, offset_inDP=0):
        """
        data: 1D array of float. 
        The rest of arguments must be non-negative integers. 
        """

        if offset_inDP != 0: data = copy.deepcopy(data[offset_inDP:])
        clips_mat = sliding_window_view(data, clip_length_inDP)[::clip_stride_inDP,:]

        return clips_mat
    

    @staticmethod
    def _one_hot_label(x):
        """
        x :<-- [0, 1, 2, 3]. Int. 
        """

        if x == 0: return np.array([1, 0, 0, 0]).astype(int) # Nominal. 
        elif x == 1: return np.array([0, 1, 0, 0]).astype(int) # Keyhole. 
        elif x == 2: return np.array([0, 0, 1, 0]).astype(int) # Lack of fusion. 
        elif x == 3: return np.array([0, 0, 0, 1]).astype(int) # Bead up. 
        else: pass

    
    def processing(self, is_spectral_transform=False):
        """
        """

        self._is_data_processed, self._is_data_spectral_transformed = False, False
        self._spectrum_list = copy.deepcopy([])

        audio_3 = np.load(self.acoustic_path)
        audio_sample = audio_3[self._audio_sensor_No,:] # 1D array of float. 

        photodiode_2 = np.load(self.photodiode_path)
        photodiode_sample = photodiode_2[self._photodiode_sensor_No,:] # 1D array of float. 

        audio_sample, _ = self._synchronize(audio_sample, photodiode_sample, self._photodiode_sync_thrsld)

        self._clips_mat = self._audio_partition(audio_sample, self._clip_length, self._clip_stride)
        self._clips_num = self._clips_mat.shape[0]

        if is_spectral_transform and self._clips_mat is not None:
            for ind in range(self._clips_num):
                clip, spectrum_obj = self._clips_mat[ind,:].reshape(-1), None
                if self.spectrum_mode == "wavelet":
                    spectrum_obj = WaveletSpectrum(data=clip, wavelet=WAVELET, scales=SCALES)
                elif self.spectrum_mode == "stft": pass # STFT.
                else: pass

                self._spectrum_list.append(spectrum_obj.spectrum()) # Default color_map, axis_off, and is_log_yscale.

                # spectrum_obj.free_all()
                del spectrum_obj
                gc.collect()

            self._is_data_spectral_transformed = True
            
        else: pass

        self._is_data_processed = True
    

    def save_offline(self, is_clips=False, clips_subdir=None, clips_extension=ACOUSTIC_DATA_EXTENSION, is_spectrums=False, 
                     spectrum_mode='wavelet', spectrums_subdir=None, spectrums_extension=SPECTRUM_EXTENSION):
        """
        """

        if self._is_data_processed and self._clips_mat is not None:
            for ind in range(self._clips_num):
                file_id_temp = str(ind).zfill(5)
                # Generate spectrogram. 
                clip, spectrum_obj = self._clips_mat[ind,:].reshape(-1), None
                if spectrum_mode == "wavelet":
                    spectrum_obj = WaveletSpectrum(data=clip, wavelet=WAVELET, scales=SCALES)
                elif spectrum_mode == "stft": pass # STFT.
                else: pass
                
                # Save clips. 
                if is_clips and clips_subdir is not None: 
                    clip_path_temp = os.path.join(clips_subdir, "{:1d}_{}_{:4d}_{:4d}_{}.{}".format(\
                                                  self._is_LOI, file_id_temp, self._P, self._V, 
                                                  '_'.join(self._one_hot_label(self._defect).astype(str)), clips_extension))
                    np.save(clip_path_temp, clip)
                else: pass
                
                # Save spectrograms.
                if is_spectrums and spectrums_subdir is not None:
                    spectrum_path_temp = os.path.join(spectrums_subdir, "{:1d}_{}_{:4d}_{:4d}_{}.{}".format(\
                                                      self._is_LOI, file_id_temp, self._P, self._V, 
                                                      '_'.join(self._one_hot_label(self._defect).astype(str)), spectrums_extension))
                    spectrum_obj.save_spectrum(spectrum_path=spectrum_path_temp)
                else: pass

                # spectrum_obj.free_all()
                del clip, spectrum_obj
                gc.collect()
        
        else: raise ValueError("Dataset is not processed. ")


class DataParser_AcousticProcessMapReconstruction(object):
    """
    General data parsing pipeline for acoustic to defect mapping. 
    """

    def __init__(self, acoustic_data_raw_dir=ACOUSTIC_DATA_DIR, photodiode_data_raw_dir=PHOTODIODE_DATA_DIR,
                 process_param_data_dir=PROCESS_PARAM_SHEET_PATH, data_processed_dir=PROCESSED_DATA_DIR, 
                 spectrum_mode=SPECTRUM_MODE, is_data_offline=True):
        """
        """

        self.acoustic_data_raw_dir = acoustic_data_raw_dir
        self.photodiode_data_raw_dir = photodiode_data_raw_dir
        self.process_param_data_dir = process_param_data_dir
        self.data_processed_dir = data_processed_dir
        self.spectrum_mode = spectrum_mode
        self.is_data_offline = is_data_offline

        self._is_data_parsed = False

        self._data_processed_clips_subdir = os.path.join(self.data_processed_dir, PROCESSED_DATA_CLIPS_SUBDIR)
        self._data_processed_spectrograms_subdir = os.path.join(self.data_processed_dir, PROCESSED_DATA_SPECTROGRAMS_SUBDIR)

        self._initial_layer_num = INITIAL_LAYER_NUM
        self._transitional_layer_num = TRANSITIONAL_LAYER_NUM
        self._repitition_time = REPITITION_TIME

        self._acoustic_file_path_list = glob.glob(os.path.join(self.acoustic_data_raw_dir, 
                                                  "*.{}".format(ACOUSTIC_DATA_EXTENSION)))
        self._photodiode_file_path_list = glob.glob(os.path.join(self.photodiode_data_raw_dir, 
                                                    "*.{}".format(PHOTODIODE_DATA_EXTENSION)))
        self._process_param_table = np.tile(np.genfromtxt(self.process_param_data_dir, delimiter=','), 
                                            reps=(REPITITION_TIME,1)) # For each row: [V, P, defect]. 

        # Not recommended - too much memory usage. 
        self._clip_data_dict = defaultdict(lambda: defaultdict(defaultdict))
        self._spectrogram_data_dict = defaultdict(lambda: defaultdict(defaultdict))
        self._label_data_dict = defaultdict(lambda: defaultdict(defaultdict))

        self._layer_num = len(self._acoustic_file_path_list)
        self._dataset_size = 0
        
        
    @property
    def layer_num(self):
        """
        """

        return self._layer_num
    

    @property
    def dataset_size(self):
        """
        """

        if not self._is_data_parsed: raise ValueError("Data is not yet parsed.")
        else: return self._dataset_size

    
    @property
    def clips(self):
        """
        """

        if not self.is_data_offline: return self._clip_data_dict
        else: raise ValueError("Data is not cached in device's memory.")


    @property
    def spectrums(self):
        """
        """

        if not self.is_data_offline: return self._spectrogram_data_dict
        else: raise ValueError("Data is not cached in device's memory.")

    
    @property
    def labels(self):
        """
        """

        if not self.is_data_offline: return self._label_data_dict
        else: raise ValueError("Data is not cached in device's memory.")


    def _clean_up(self):
        """
        """

        clr_dir(self.data_processed_dir)
        
        if not os.path.isdir(self._data_processed_clips_subdir): os.mkdir(self._data_processed_clips_subdir)
        if not os.path.isdir(self._data_processed_spectrograms_subdir): os.mkdir(self._data_processed_spectrograms_subdir)


    def batch_processing(self, clean_up=False):
        """
        """

        if clean_up: self._clean_up()
        self._is_data_parsed, is_LOI, self._dataset_size = False, False, 0

        for ind in range(self.layer_num):
            ready_layer_gap = self._transitional_layer_num + 1
            ready_layer_No = ind - self._initial_layer_num - self._transitional_layer_num

            if ind + 1 <= self._initial_layer_num or ready_layer_No % ready_layer_gap != 0: 
                V_temp, P_temp, defect_temp, is_LOI = 1200, 280, 0, False
            else: 
                V_temp, P_temp, defect_temp = self._process_param_table[int(ready_layer_No//ready_layer_gap),:]
                is_LOI = True
            
            if not is_LOI: continue
            process_param_vector_temp = np.array([is_LOI, P_temp, V_temp, defect_temp]).astype(int)
            
            audio_folder_name_temp = str(ind).zfill(5)
            clips_subdir_this_audio = os.path.join(self._data_processed_clips_subdir, audio_folder_name_temp)
            if not os.path.isdir(clips_subdir_this_audio): os.mkdir(clips_subdir_this_audio)
            specs_subdir_this_audio = os.path.join(self._data_processed_spectrograms_subdir, audio_folder_name_temp)
            if not os.path.isdir(specs_subdir_this_audio): os.mkdir(specs_subdir_this_audio)

            audio_parser_temp = DP_APMR_Single(acoustic_path=self._acoustic_file_path_list[ind], 
                                               photodiode_path=self._photodiode_file_path_list[ind], 
                                               process_param_vector=process_param_vector_temp)

            if self.is_data_offline: audio_parser_temp.save_offline(is_clips=True, clips_subdir=clips_subdir_this_audio,
                                                                    is_spectrums=True, spectrum_mode=self.spectrum_mode, 
                                                                    spectrums_subdir=specs_subdir_this_audio)
            else: 
                audio_parser_temp.processing(is_spectral_transform=True)
                for i, spectrogram in enumerate(audio_parser_temp.spectrograms):
                    clip_id_name = str(i).zfill(5)
                    self._clip_data_dict[audio_folder_name_temp][clip_id_name] = copy.deepcopy(audio_parser_temp.clips[i,:])
                    self._spectrogram_data_dict[audio_folder_name_temp][clip_id_name] = copy.deepcopy(spectrogram)
                    self._label_data_dict[audio_folder_name_temp][clip_id_name] = copy.deepcopy(process_param_vector_temp)
                
            self._dataset_size += audio_parser_temp.clips_num

            # audio_parser_temp.free_all()
            del audio_parser_temp
            gc.collect()
        
        self._is_data_parsed = True


################################################################################
DEBUG = False
IS_DATA_PROCESSING, IS_CLEAN_UP = True, True
IS_DATA_OFFLINE = True
IS_TRAIN = True

if __name__ == "__main__":
    """
    """

    if IS_DATA_PROCESSING:
        data_parser = DataParser_AcousticProcessMapReconstruction(is_data_offline=IS_DATA_OFFLINE)
        data_parser.batch_processing(clean_up=IS_CLEAN_UP)
        
    if IS_TRAIN:
        pass




