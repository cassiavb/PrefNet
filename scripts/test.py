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

def test_configuration(config):
    
    data_dir   = config['data_dir']
    model_name = config['model_name']
    model_type = config['model_type']
    test_list_file = config['test_list_file']
    model_dir  = config['model_dir']
    extract_features = config.get('extract_features', False)

    # Creating output directories
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Getting list of experiments to be used for testing
    test_list  = read_file_to_list( test_list_file )

    if extract_features:
        print("Extracting DFT for test set")
        extractor = FeatureExtraction(test_list, config)
        extractor.get_features()

    ## Check if model exists
    model_file = '{0}/{1}'.format(config['model_dir'], config['model_name'])

    if not os.path.isfile(model_file):
        raise ValueError('Model does not exist!')
    
    # Load model data
    print("Loading model: {0}".format(model_file))
    model = architectures.gru_and_attention_model( config , vocdim = config['n_mels'])
    model.load_weights(model_file)

    # Loading test data --- this pads the data into max_length
    print("Loading test data")
    feats1_test, feats2_test, len1_test, len2_test, scores_test, max_length_test, files1, files2 = load_data_features( config, test_list )
    print("Num test wav pairs : " + str(len(scores_test)))

    out  = np.zeros( shape=(len1_test.shape[0] , 1 ))

    # Loop over pairs in test set
    print("Getting model prediction")
    scores_sys, scores_sys_count, actual_scores_sys = {}, {}, {}
    for i, (wav1, wav2, len1, len2, score, file1, file2) in enumerate(zip(feats1_test, feats2_test, len1_test, len2_test, scores_test, files1, files2)):

        # Remove padding
        wav1 = wav1[-int(len1):,:]
        wav2 = wav2[-int(len2):,:]

        # Add dim
        wav1 = np.expand_dims(wav1, axis=0)
        wav2 = np.expand_dims(wav2, axis=0)

        # Add dim
        len1 = np.expand_dims(len1, axis=0)
        len2 = np.expand_dims(len2, axis=0)

        # Run prediction
        out[i] = model.predict(x=[wav1, wav2, len1, len2])
        predicted = np.asscalar(out[i])
        actual = np.asscalar(score)
       
        #print("pair {} {} {:.2f} {:.2f}".format(file1,file2,predicted,actual))

        # Getting system score
        sys1 = file1.split('___')[-1]
        sys2 = file2.split('___')[-1]
        if sys1 > sys2:
            pair = sys1 + '___' + sys2
        else:
            pair = sys2 + '___' + sys1
            predicted = 1. - predicted
            actual = 1. - actual
        if pair in scores_sys.keys():
            scores_sys[ pair ] += predicted
            actual_scores_sys[ pair ] += actual
            scores_sys_count[ pair ] += 1
        else:
            scores_sys[ pair ] = predicted
            actual_scores_sys[ pair ] = actual
            scores_sys_count[ pair ] = 1

    print("Calculating sentence and system accuracy")

    num_pairs = len(scores_sys.keys())
    corr_pairs = 0
    for key in scores_sys.keys():
        scores_sys[key]        = scores_sys[key]/scores_sys_count[key]
        actual_scores_sys[key] = actual_scores_sys[key]/scores_sys_count[key]
        
        if ( ( scores_sys[key] <= 0.5 ) and ( actual_scores_sys[key] <= 0.5 ) ) or ( ( scores_sys[key] > 0.5 ) and ( actual_scores_sys[key] > 0.5 ) ):
            corr_pairs += 1
            #print("++ {} {:.2f} {:.2f}".format(key, scores_sys[key], actual_scores_sys[key],))
        #else:
            #print("-- {} {:.2f} {:.2f}".format(key, scores_sys[key], actual_scores_sys[key],))

    print("------ System accuracy: {:.1f}".format(100*corr_pairs/num_pairs))
    print("Num system pairs: " + str(num_pairs))

    # Calculate accuracy over the entire test set
    accuracy = calculate_accuracy(scores_test.copy(), out.copy())

    print("----- Sentence accuracy: {:.1f}".format(accuracy))
    print("Num sentence pairs: " + str(len(out)))
    
if __name__=="__main__":

    print("Reading arguments")

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    opts       = a.parse_args()
    config     = {}

    #### Get config settings
    config = load_config(opts.config_fname)

    test_configuration(config)


