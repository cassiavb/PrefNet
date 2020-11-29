#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
...

Date: 2018
Author: M. Sam Ribeiro
"""

import os
import numpy as np
import soundfile as sf

from data_utils import get_list_pairs
from files_io import read_file_to_list, array_to_binary_file
from spectrogram_extractor import get_spectrogram_extractor


class FeatureExtraction(object):

    def __init__(self, exp_list, config):

        self.exp_list = exp_list
        self.config = config

        self.out_dir = config['feat_dir']
        self.wav_dir = os.path.join(config['data_dir'], 'processed', 'wav16k')

        self.__initialize()


    def __initialize(self):
        ''' initializes feature extractor '''

        config = self.config

        # define DFT model -- layer initialised w a DFT: B x T X 1 -> B x N X F x 1
        DFT_layer = get_spectrogram_extractor(n_mels=config.get('n_mels',40), \
            normalise=config.get('norm_spectrogram', 'none'), \
            dft_window=config.get('dft_window', 512), n_hop=config.get('n_hop', 200), \
            decibel_melgram=config.get('decibel_melgram', False), spectrogram_type=config.get('spectrogram_type', 'MelSTFT'), \
            power=config.get('spectrogram_power', 2.0), trainable_kernel=config.get('trainable', False))

        self.extractor = DFT_layer

        # get filelist from experiments
        files = set([])
        for pair in get_list_pairs(config, self.exp_list):
            file1, file2 = pair.split(" : ")
            files.add(file1)
            files.add(file2)

        # print('WARNING: Truncating filelist!!!!')
        # self.filelist = list(files)[:2]
        self.filelist = list(files)

        # create output directory
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)



    def get_features(self):
        ''' extracts features from waveforms '''

        layer = self.extractor
        in_dir  = self.wav_dir
        out_dir = self.out_dir

        for file in self.filelist:

            in_filename  = os.path.join(in_dir,  file+'.wav')
            out_filename = os.path.join(out_dir, file+'.feats')

            wav, fs = sf.read(in_filename)

            # reshape input data for Kapre (1xNx1)
            wav = np.expand_dims(wav, axis=0)
            wav = np.reshape(wav,(1, -1, 1))

            # get features and remove unwanted dimensions (1xNxFx1 -> NxF)
            feats = layer.predict(x=[wav])
            feats = np.squeeze(feats)

            # save to file
            array_to_binary_file(feats, out_filename)

