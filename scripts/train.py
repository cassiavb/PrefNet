# -*- coding: utf-8 -*-
#/usr/bin/python3

import os
import numpy as np
import tensorflow as tf
import random as rn

np.random.seed(42)
rn.seed(12345)

from keras.layers import Input, Dense, Average
from keras.models import Model, load_model, Sequential
from keras import losses
from keras import backend as K
from keras.utils.io_utils import HDF5Matrix
from keras import optimizers
from keras.callbacks import ModelCheckpoint

tf.set_random_seed(12345)
sess = tf.Session(graph=tf.get_default_graph()) #, config=session_conf)
K.set_session(sess)

import struct
import sys
import os
import glob
import h5py
from argparse import ArgumentParser
import architectures
from files_io import load_config, read_file_to_list, array_to_binary_file, create_dir, write_dict_to_file
from bucketed_sequence import BucketedSequenceTwoInputs
import time
import datetime
from shutil import copy2
from data_utils import load_data_features, divide_data
from preprocessing import FeatureExtraction

def train_configuration(config):
    
    data_dir   = config['data_dir']
    model_name = config['model_name']
    model_type = config['model_type']
    train_list_file = config['train_list_file']
    test_list_file = config['test_list_file']
    model_dir  = config['model_dir']
    extract_features = config.get('extract_features', False)

    #### Creating output directories
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    ## Write config file in case model needs to be loaded
    write_dict_to_file( config , model_dir + '/this_config.cfg')

    ## Save config file in case model needs to be loaded
    copy2( opts.config_fname , config['model_dir'] + '/')

    ## Checking that dimensions are correct (conv_dim needs to be an even number and conv_dim/2 needs to be a multiple of n_heads)
    assert np.mod(config.get('conv_dim',40) , 2) == 0
    assert np.mod(config.get('conv_dim',40)/2 , config.get('n_heads',1)) == 0
    head_dim = (config.get('conv_dim',40)/2 ) / config.get('n_heads',1)

    ## Getting list of experiments to be used for training and testing
    exp_list_train = read_file_to_list( train_list_file )
    exp_list_test  = read_file_to_list( test_list_file ) # is used if there is no leave-one-fold-out cross validation

    if extract_features:
        extractor = FeatureExtraction(exp_list_train, config)
        extractor.get_features()

    # Partitioning train data
    if config.get('crossval',False) == False:
        num_folds    = 1
        fold_indexes = -1
    else: # leave-one-fold-out cross validation
        fold_indexes = config.get('fold_indexes',[1])
        fold_indexes = np.asarray(fold_indexes)
        num_folds    = np.minimum( config.get('num_folds', 1) , len(fold_indexes) )

    start_fold     = 0
    accuracy_folds = np.zeros( (2, num_folds ))

    for fold in range(num_folds):

        ## Check if model for this fold already exists and that it can be overwritten
        if num_folds == 1: # if only one fold dont add fold number to model name
            model_file = '{0}/{1}'.format(config['model_dir'], config['model_name'])
        else: # add fold number to model name
            model_file = '{0}/{1}_{2}'.format(config['model_dir'], fold, config['model_name'])

        if os.path.isfile(model_file) and ow_model is False and adapt_model is False:
            raise ValueError('Model already exists. To overwrite it add -M')

        if os.path.isfile(model_file) is False and adapt_model is True:
            raise ValueError('Model cant be adapted: model file ' + model_file + ' not found.')
        
        ## Getting test and train set for this fold
        if num_folds == 1 and config.get('crossval',False) == False:
            test_list  = exp_list_test
            train_list = exp_list_train
        else:
            test_list  = exp_list_train[ start_fold: fold_indexes[fold]+1 ]
            train_list = [item for item in exp_list_train if item not in test_list]
            start_fold = fold_indexes[fold] + 1

        print("Train list:")
        print(train_list)
        print("Test list:")
        print(test_list)

        print("Loading train data and initialise model")
        model = None
        
        feats1, feats2, len1, len2, scores, max_length, files1, files2  = load_data_features( config, train_list )
        model = architectures.gru_and_attention_model( config , vocdim = config['n_mels'], max_length = max_length)

        # Divide train/val randomly  
        [train_input, train_target, val_input, val_target] = divide_data(feats1, feats2, len1, len2, scores)

        if adapt_model: # Loading existing model -- for adaptation!!
            print("Load existing model for adaptation")
            model.load_weights(model_file)

        print("----- Training fold " + str(fold))

        # Training settings
        lr = config.get('learning_rate',0.001)
        optimizer = optimizers.Adam(lr=lr)
        model.compile(optimizer=optimizer, loss='mse')

        # Bucketing sequence with multiple inputs -- doesnt work well with inputs of two diff sizes
        train_generator = BucketedSequenceTwoInputs(config, train_input, train_target)
        val_generator   = BucketedSequenceTwoInputs(config, val_input, val_target)

        callback_list = []
        if config['early_stopping']:
            checkpoint = ModelCheckpoint(model_file, \
                monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
            callback_list.append(checkpoint)

        print("Start training")

        # Train all epochs
        # hist = model.fit_generator(train_generator, epochs=config.get('epochs',2), validation_data=val_generator, callbacks=callback_list, shuffle=True, verbose=1)

        # # Train epoch by epoch to stop if not converged
        # hist_train_loss, hist_val_loss = [], []
        # for epoch in range(config.get('epochs',2)):
        #     hist = model.fit_generator(train_generator, initial_epoch=epoch, epochs=epoch+1, validation_data=val_generator, callbacks=callback_list, shuffle=True, verbose=1)
        #     train_loss = float(hist.history['loss'][-1])
        #     val_loss = float(hist.history['val_loss'][-1])
        #     hist_train_loss.append(train_loss)
        #     hist_val_loss.append(val_loss)
        #     if epoch==0:
        #         if train_loss >= 0.1:
        #             print(f"Model didnt converge, stop trainning!")
        #             break
        # hist_train_loss = np.array(hist_train_loss)
        # hist_val_loss = np.array(hist_val_loss)
        # array_to_binary_file(hist_train_loss, config['model_dir'] + '/' + str(fold) + '_train_loss.float')
        # array_to_binary_file(hist_val_loss, config['model_dir'] + '/' + str(fold) + '_val_loss.float')


        # Keep training until it converges
        not_converged = True
        attempts = 1
        while not_converged:
            hist_train_loss, hist_val_loss = [], []
            
            if attempts>1:

                model = None
                model = architectures.gru_and_attention_model( config , vocdim = config['n_mels'], max_length = max_length)

                if adapt_model: # Loading existing model -- for adaptation!!
                    print("Load existing model for adaptation")
                    model.load_weights(model_file)
                    
                model.compile(optimizer=optimizer, loss='mse')

            for epoch in range(config.get('epochs',2)):
                hist = model.fit_generator(train_generator, initial_epoch=epoch, epochs=epoch+1, validation_data=val_generator, callbacks=callback_list, shuffle=True, verbose=1)
                train_loss = float(hist.history['loss'][-1])
                val_loss   = float(hist.history['val_loss'][-1])
                hist_train_loss.append(train_loss)
                hist_val_loss.append(val_loss)
                if epoch==0:
                    if train_loss >= 0.11:
                        print(f"Model didnt converge, stop trainning!")
                        break
                    else:
                        not_converged = False
            if attempts == 10:
                print(f"Model didnt converge -- TOO MANY ATTEMPTS!")
                not_converged = False
            attempts +=1

        hist_train_loss = np.array(hist_train_loss)
        hist_val_loss = np.array(hist_val_loss)
        array_to_binary_file(hist_train_loss, config['model_dir'] + '/' + str(fold) + '_train_loss.float')
        array_to_binary_file(hist_val_loss, config['model_dir'] + '/' + str(fold) + '_val_loss.float')

    
        if not config['early_stopping']:
            print("Saving model file: " + model_file)
            model.save_weights(model_file)

        print("----- Finished training fold " + str(fold))

if __name__=="__main__":

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    a.add_argument('-s', dest='hyperpar')
    a.add_argument('-M', dest='ow_model', action='store_true', help= "clear any previous MODEL file first")
    a.add_argument('-A', dest='adapt_model', action='store_true', help= "adapt model")
    opts       = a.parse_args()
    config     = {}
    ow_model   = opts.ow_model
    adapt_model   = opts.adapt_model

    #### Get config settings
    config = load_config(opts.config_fname)

    train_configuration(config)


