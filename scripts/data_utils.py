# -*- coding: utf-8 -*-
#/usr/bin/python3

import numpy as np
import struct
import sys
import os
import glob
import h5py
import soundfile as sf
from keras.preprocessing.sequence import pad_sequences

from files_io import read_file_to_list, read_file_collumn_to_list, check_file_exists, load_binary_file

import time
import datetime

scores_dir = '/processed/scores/'

def calculate_accuracy(scores_orig, out_orig):

    scores = scores_orig
    out    = out_orig

    scores[scores<=0.5] = 0.0
    scores[scores>0.5] = 1.0

    out[out<=0.5] = 0.0
    out[out>0.5] = 1.0

    num_correct = sum(scores==out)

    return float(100.0*num_correct / len(scores))

def divide_data(wavs1, wavs2, len1, len2, scores):

    num_all   = len(scores)
    num_train = int(num_all*0.9)
    num_val   = num_all - num_train

    # Divide train/val randomly
    tmp         = np.random.permutation(np.arange(num_all))
    # Get last ones as val
    #tmp = np.arange(num_all)
    
    val_index   = tmp[:num_val]
    train_index = tmp[num_val:]

    train_input1 = wavs1[train_index]
    train_input2 = wavs2[train_index]
    train_len1   = len1[train_index]
    train_len2   = len2[train_index]
    train_target = scores[train_index]

    val_input1 = wavs1[val_index]
    val_input2 = wavs2[val_index]
    val_len1   = len1[val_index]
    val_len2   = len2[val_index]
    val_target = scores[val_index]

    train_input = [train_input1, train_input2, train_len1, train_len2]
    val_input   = [val_input1, val_input2, val_len1, val_len2]

    return train_input, train_target, val_input, val_target

def get_exp_name(filename):

    return filename[:4]

def get_pair_wavs_score(config, pair):

    basedir  = config['data_dir']
    file1, file2 = pair.split(" : ")
    
    wav1, fs1  = sf.read( basedir + 'processed/wav16k/' + file1 + '.wav')
    wav2, fs2  = sf.read( basedir + 'processed/wav16k/' + file2 + '.wav')

    exp_name = get_exp_name(file1)

    filenames         = read_file_to_list( basedir + '/processed/filenames/' + exp_name + '_filenames.txt')
    scores, num_files = load_binary_file( basedir + scores_dir + exp_name + '_scores.float', len(filenames))
    index1            = filenames.index(file1)
    index2            = filenames.index(file2)

    # Note: this wont work for experiments where the same wavefile was used in different screens 
    # (such as E012, E013 and E044)
    # In this case, score will be -1 when the index is wrong (no pair exists), in that case go to 
    # next index by starting the search from index + 1:   filenames.index(file1 , index1 + 1)

    score = scores[index1,index2]

    # wav1 should always be the longer one so the bucketing works...
    if len(wav1) < len(wav2):
        tmp   = wav2
        wav2  = wav1
        wav1  = tmp
        score = 1.0 - score    
        file1, file2 = file2, file1

    assert score >= 0.0
    assert score <= 1.0

    if config.get('discrete_target',True):
        # Turn continuous score into 3 values: 0, 0.5 and 1.0
        if score < 0.5:
            score = 0.0
        elif score > 0.5:
            score = 1.0

    return wav1, wav2, score, file1, file2


def get_list_pairs(config, exp_list):

    pairs = []
    basedir  = config['data_dir']
    
    for n in range(len(exp_list)):
        new_pairs = read_file_to_list(basedir + 'processed/filenames/' + exp_list[n] + '_pairs.txt')
        for m in range(len(new_pairs)):
            if 'alba' not in new_pairs[m] and '__ctt' not in new_pairs[m]:
                pairs.append(new_pairs[m])

    return pairs

def get_pair_feats_score(config, pair):

    featdir  = config['feat_dir']
    datadir  = config['data_dir']


    n_mels = config['n_mels']
    file1, file2 = pair.split(" : ")

    #print(os.path.join(featdir, file1+'.feats'))
    feats1, len1 = load_binary_file(os.path.join(featdir, file1+'.feats'), n_mels)
    feats2, len2 = load_binary_file(os.path.join(featdir, file2+'.feats'), n_mels)

    exp_name = get_exp_name(file1)

    filenames         = read_file_to_list( datadir + '/processed/filenames/' + exp_name + '_filenames.txt')
    scores, num_files = load_binary_file( datadir + scores_dir + exp_name + '_scores.float', len(filenames))
    index1            = filenames.index(file1)
    index2            = filenames.index(file2)

    # Note: this wont work for experiments where the same wavefile was used in different screens 
    # (such as E012, E013 and E044)
    # In this case, score will be -1 when the index is wrong (no pair exists), in that case go to 
    # next index by starting the search from index + 1:   filenames.index(file1 , index1 + 1)
    score = scores[index1,index2]

    #if file1[-3:]=="REF" and score!=1.0:
        #print(str(score))
    #    print(file1 + " " + file2 + " " + str(score))

    # wav1 should always be the longer one so the bucketing works...
    if feats1.shape[0] < feats2.shape[0]:
        feats1, feats2 = feats2, feats1
        file1, file2   = file2, file1
        len2, len1 = len1, len2
        score = 1.0 - score
   
    assert score >= 0.0
    assert score <= 1.0

    if config.get('discrete_target',True):
        # Turn continuous score into 3 values: 0, 0.5 and 1.0
        if score < 0.5:
            score = 0.0
        elif score > 0.5:
            score = 1.0

    return feats1, feats2, len1, len2, score, file1, file2

def pad_features(seqs, lens, max_length):

    for i in range(len(seqs)):
        pad_len= max_length - lens[i]
        seqs[i] = np.pad(seqs[i], [(pad_len,0), (0,0)], 'constant')

    return np.array(seqs)


def get_data_features(config, pairs):

    feats1, feats2 = [], []
    files1, files2 = [] ,[]

    len1 = np.zeros(len(pairs), dtype=np.int)
    len2 = np.zeros(len(pairs), dtype=np.int)

    scores      = np.zeros(len(pairs), dtype=np.float)

    for idx, pair in enumerate(pairs):

        feat1, feat2, flen1, flen2, score, file1, file2 = get_pair_feats_score(config, pair)

        feats1.append(feat1)
        feats2.append(feat2)

        len1[idx] = flen1
        len2[idx] = flen2

        scores[idx] = score

        files1.append(file1)
        files2.append(file2)

        #print("{} {} {}".format(file1,file2,score))
        #assert len1[idx] >= len2[idx]

    max_length = np.max(np.concatenate([len1, len2]))
    print("Padding to max length: {0}".format(max_length))

    feats1 = pad_features(feats1, len1, max_length)
    feats2 = pad_features(feats2, len2, max_length)

    return feats1, feats2, len1, len2, scores, max_length, files1, files2



def load_data_features(config, exp_list):

    pairs = get_list_pairs(config, exp_list)
    feats1, feats2, len1, len2, scores, max_length, files1, files2  = get_data_features(config, pairs)

    len1   = np.expand_dims(len1, axis=1)
    len2   = np.expand_dims(len2, axis=1)
    scores = np.expand_dims(scores, axis=1)

    return feats1, feats2, len1, len2, scores, max_length, files1, files2


