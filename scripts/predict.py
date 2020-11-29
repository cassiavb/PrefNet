# -*- coding: utf-8 -*-
#/usr/bin/python3

import os
import numpy as np
import tensorflow as tf
import random as rn
from keras.layers import Input, Dense, Average
from keras.models import Model, load_model, Sequential
from keras import losses
from keras import backend as K
from keras.utils.io_utils import HDF5Matrix
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import struct
import sys
import os
import glob
import h5py
from argparse import ArgumentParser
from data_utils import load_data_features, calculate_accuracy
import architectures
from files_io import load_config, read_file_to_list, array_to_binary_file, create_dir, write_dict_to_file, write_accuracy_results
from bucketed_sequence import BucketedSequenceTwoInputs
import time
import datetime
from shutil import copy2

from preprocessing import FeatureExtraction
import soundfile as sf

def predict_configuration(config, file1, file2):

    model_type = config['model_type']
    model_dir  = config['model_dir']
    extract_features = True

    print("Extracting features for test set")
    FeatExtractor = FeatureExtraction('', config)
    FeatExtractor = FeatExtractor.extractor

    ###### File1
    wav, fs = sf.read(file1)
    if fs != 16000: 
        raise ValueError('File1 is no sampled at 16kHz!')
    wav     = np.expand_dims(wav, axis=0)
    wav     = np.reshape(wav,(1, -1, 1))
    feats1  = FeatExtractor.predict(x=[wav])
    feats1  = np.squeeze(feats1, axis=3)

    ###### File2
    wav, fs = sf.read(file2)
    if fs != 16000: 
        raise ValueError('File2 is not sampled at 16kHz!')
    wav     = np.expand_dims(wav, axis=0)
    wav     = np.reshape(wav,(1, -1, 1))
    feats2  = FeatExtractor.predict(x=[wav])
    feats2  = np.squeeze(feats2, axis=3)
    
    ## Check if model exists
    model_file = '{0}/{1}'.format(config['model_dir'], config['model_name'])        
    if not os.path.isfile(model_file):
        raise ValueError('Model does not exist!')
    
    # Load model data
    print("Loading model: {0}".format(model_file))
    model = architectures.gru_and_attention_model( config , vocdim = config['n_mels'])
    model.load_weights(model_file)

    # Loop over pairs in test set
    print("Getting model prediction")

    # Add dim
    len1 = feats1.shape[1]
    len2 = feats2.shape[1]
    len1 = np.expand_dims(len1, axis=0)
    len2 = np.expand_dims(len2, axis=0)

    # Run prediction
    predicted = np.asscalar( model.predict(x=[feats1, feats2, len1, len2]) )
       
    print("--- Preference (input1 over input2): {:.2f}%".format(100.0*predicted))

if __name__=="__main__":

    print("Reading arguments")

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    a.add_argument('-i1', dest='file1', required=True)
    a.add_argument('-i2', dest='file2', required=True)
    opts       = a.parse_args()
    config     = {}

    #### Get config settings
    config = load_config(opts.config_fname)

    predict_configuration(config, opts.file1, opts.file2)



