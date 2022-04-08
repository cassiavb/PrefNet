# -*- coding: utf-8 -*-
#/usr/bin/python3

import os
import numpy as np
import sys
import os
from argparse import ArgumentParser
from data_utils import load_scores
from files_io import read_file_to_list

def get_topline(test_list_file, data_dir, test_asone):
    
    # Getting list of experiments
    test_list  = read_file_to_list( test_list_file )

    topline_acc = []
    for test in test_list:

        test = [test]
        if test_asone:
            test = test_list

        max_scores = load_scores( data_dir, test, max_score=True )

        print(max_scores.shape)
        topline_acc.append(100.0*np.mean(max_scores))

        if test_asone:
            break

    topline_acc = np.array(topline_acc)

    np.set_printoptions(precision=1)

    print(topline_acc)
    
if __name__=="__main__":

    print("Reading arguments")

    a = ArgumentParser()
    a.add_argument('-l', dest='test_list_file', required=True)
    a.add_argument('-d', dest='data_dir', required=True)
    a.add_argument('-o', dest='test_asone', action='store_true')
    opts = a.parse_args()
    
    get_topline(opts.test_list_file, opts.data_dir, opts.test_asone)

