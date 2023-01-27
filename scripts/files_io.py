# -*- coding: utf-8 -*-
#/usr/bin/python3

import numpy as np
import os
import glob
# import h5py
import sys
import json

def create_dir(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)
        
def load_config(config_fname):
    config = {}

    #execfile(config_fname, config)
    with open(config_fname, "rb") as source_file:
        code = compile(source_file.read(), config_fname, "exec")
    exec(code, config)

    del config['__builtins__']
    _, config_name = os.path.split(config_fname)
    config_name = config_name.replace('.cfg','').replace('.conf','')
    config['config_name'] = config_name
    return config

def read_file_to_list(filename):

    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    f.close()

    return content

def write_dict_to_file(dict, filename):

    f = open(filename, "w")

    for k, v in dict.items():
        f.write(str(k) + ' = '+ str(v) + '\n')

    f.close()

def read_file_collumn_to_list(filename, col=0, delimiter=' '):

    all_content = read_file_to_list(filename)
    content = []
    for line in all_content:
        content.append(line.split(delimiter)[col])

    return content

def check_file_exists(file):

    return os.path.isfile(file)

def write_files_from_data(data, data_dir, feat_ext, num_frames, list_file):

    list_file = open(list_file, "r") 
    files = list_file.readlines()
    num_files = len(files)
    f = 0
    start  = 0
    for base in files:
        m_data   =  data[start:start+num_frames[f],:]
        filename =  data_dir + base[:-1] + feat_ext
        start    += num_frames[f]
        write_binfile(m_data, filename) 
        f += 1
    list_file.close()

def load_binary_file(file_name, dimension):

    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0,'specified dimension not compatible with data'
    frame_number = int( features.size / dimension )
    features = features[:(dimension * frame_number)]
    features = features.reshape((-1, dimension))
    
    return  features, frame_number

def array_to_binary_file(data, output_file_name):

    data = np.array(data, 'float32')

    fid = open(output_file_name, 'wb')
    data.tofile(fid)
    fid.close()

def write_accuracy_results(data, output_file_name):

    np.savetxt(output_file_name, data,fmt='%0.2f')
