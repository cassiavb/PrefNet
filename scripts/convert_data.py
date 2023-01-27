# -*- coding: utf-8 -*-
# /usr/bin/python3

import numpy as np
import struct
import sys
import os
import glob
from files_io import read_file_to_list, read_file_collumn_to_list, check_file_exists, load_binary_file

discrete_target = False
DEBUG = False

def write_list_to_file(inlist, filename):

    with open(filename,'w') as f:
        for item in inlist:
            f.write("%s\n" % item)
    f.close()

def get_score(pair, score_files, filename_files):
    score = 0

    # Get experiment number and file names in pair
    experiment = pair.split('___')[0]
    file1, file2 = pair.split(" : ")

    # Some experiments the same wavefile was used in different screens (E012, E013 and E044)
    # and scores matrix is not reliable, so skip them
    if (experiment == "E012") or (experiment == "E013") or (experiment == "E044"):
        return -1

    # Find which score and filename file to read for this experiment
    score_file    = [s for s in score_files if experiment in s][0]
    filename_file = [s for s in filename_files if experiment in s][0]

    # Read score matrix, the row and col for file1 and file2 and read the score
    filenames = read_file_to_list(filename_file)
    scores, num_files = load_binary_file(score_file, len(filenames))
    index1 = filenames.index(file1)
    index2 = filenames.index(file2)
    score  = scores[index1, index2]

    if DEBUG:
        index1 = filenames.index(file2)
        index2 = filenames.index(file1)
        score2 = scores[index1, index2]
        assert score + score2 == 1

    # if file1[-3:]=="REF" and score!=1.0: ### isn't always true because of listeners that didn't do the task right.. in that case should we turn it to 1?
    #     print(file1 + " " + file2 + " " + str(score))
   
    assert score >= 0.0
    assert score <= 1.0

    if discrete_target:
        # Turn continuous score into 3 values: 0, 0.5 and 1.0
        if score < 0.5:
            score = 0.0
        elif score > 0.5:
            score = 1.0

    return score

def add_scores(pairs, score_files, filename_files):
    pairs_scores = []
    for pair in pairs:
        score = get_score(pair, score_files, filename_files)
        if score != -1: # only store valid ones
            pairs_scores_item = f"{pair} : {score}"
            pairs_scores.append(pairs_scores_item)
    return pairs_scores

def get_list_pairs(pairs_files):
    pairs = []
    for pairs_file in pairs_files:
        new_pairs = read_file_to_list(pairs_file)
        for m in range(len(new_pairs)):
            # remove pairs that are not Nick
            if 'alba' not in new_pairs[m] and '__ctt' not in new_pairs[m]:
                pairs.append(new_pairs[m])
    return pairs

if __name__ == "__main__":

    pairs_files = sys.argv[1]
    score_files = sys.argv[2]
    filename_files = sys.argv[3]

    pairs_files = read_file_to_list(pairs_files)
    score_files = read_file_to_list(score_files)
    filename_files = read_file_to_list(filename_files)

    pairs = get_list_pairs(pairs_files)
    pairs_scores = add_scores(pairs, score_files, filename_files)

    write_list_to_file(pairs_scores, 'pairs_scores_all.txt')
