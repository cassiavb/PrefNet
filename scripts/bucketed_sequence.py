# -*- coding: utf-8 -*-
#/usr/bin/python3

import math
import random
import numpy as np
from keras import utils

def _roundto(val, batch_size):
    return int(math.ceil( int( val / batch_size)) ) * batch_size

class BucketedSequenceTwoInputs(utils.Sequence):
    """
    A Keras Sequence (dataset reader) of input sequences read in bucketed bins.
    Assumes all inputs are already padded using `pad_sequences` (where padding 
    is prepended).
    This was adapted from: https://github.com/tbennun/keras-bucketed-sequence.git
    """

    def __init__(self, config, x, y):

        # Getting input
        x_seq1       = x[0]
        x_seq2       = x[1]
        seq_lengths1 = x[2]
        seq_lengths2 = x[3]

        # commented this out 28.08.20
        seq_lengths = seq_lengths1 # make sure that sequences in x_seq1 are longer than in x_seq2
        #seq_lengths = np.maximum(seq_lengths1, seq_lengths2)

        num_buckets = config.get('num_buckets',100)

        self.n_hop      = config.get('n_hop',200)
        self.batch_size = config.get('batch_size',8)

        # Count bucket sizes
        bucket_sizes, bucket_ranges = np.histogram(seq_lengths, bins=num_buckets)

        # Obtain the (non-sequence) shapes of the inputs and outputs
        input_shape1 = (1,) if len(x_seq1.shape) == 2 else x_seq1.shape[2:]
        input_shape2 = (1,) if len(x_seq2.shape) == 2 else x_seq2.shape[2:]
        output_shape = (1,) if len(y.shape) == 1 else y.shape[1:]

        # Looking for non-empty buckets
        actual_buckets = [bucket_ranges[i+1] for i,bs in enumerate(bucket_sizes) if bs > 0]
        # actual_buckets = the length of the seq inside each bucket

        actual_bucketsizes = [bs for bs in bucket_sizes if bs > 0]
        bucket_seqlen = [int(math.ceil(bs)) for bs in actual_buckets]
        num_actual = len(actual_buckets)

        lengths1 = [ np.ndarray([bs] + list(output_shape), dtype=y.dtype)
                     for bsl,bs in zip(bucket_seqlen, actual_bucketsizes)]


        lengths2 = [ np.ndarray([bs] + list(output_shape), dtype=y.dtype)
                     for bsl,bs in zip(bucket_seqlen, actual_bucketsizes)]

        # Check bucket len sequence of input1 and input2
        bctr = [0]*num_actual

        for i, (sl1, sl2) in enumerate(zip(seq_lengths1, seq_lengths2)):
            for j in range(num_actual): # loop thorugh buckets (starting from the ones that should hold smaller sequences)
                bsl = bucket_seqlen[j]
                if sl1 < bsl or j == num_actual - 1: # put sequence in bucket where it fits

                    lengths1[j][bctr[j]] = sl1
                    lengths2[j][bctr[j]] = sl2

                    bctr[j] += 1
                    break

        # Get max length in each bucket
        bucket_seqlen1 = np.ndarray(len(bucket_seqlen),dtype=np.int)
        bucket_seqlen2 = np.ndarray(len(bucket_seqlen),dtype=np.int)
        for j in range(num_actual):
             bucket_seqlen1[j] = 1+np.max(np.squeeze(np.max(lengths1[j])))
             bucket_seqlen2[j] = 1+np.max(np.squeeze(np.max(lengths2[j])))
        bucket_seqlen1[-1]-=1 
        bucket_seqlen2[-1]-=1 

        self.bins = [( np.ndarray([bs, bsl1] + list(input_shape1), dtype=x_seq1.dtype),
                       np.ndarray([bs, bsl2] + list(input_shape2), dtype=x_seq2.dtype),
                       np.ndarray([bs] + list(output_shape), dtype=y.dtype)) 
                     for bsl1,bsl2,bs in zip(bucket_seqlen1, bucket_seqlen2, actual_bucketsizes)]
        assert len(self.bins) == num_actual

        # Insert the sequences into the bins
        bctr = [0]*num_actual
        for i, (sl1, sl2) in enumerate(zip(seq_lengths1, seq_lengths2)):
            for j in range(num_actual): # loop thorugh buckets (starting from the ones that should hold smaller sequences)
                bsl1 = bucket_seqlen1[j]
                bsl2 = bucket_seqlen2[j]
                if sl1 < bsl1 or j == num_actual - 1: # put sequence in bucket where it fits

                    self.bins[j][0][bctr[j],:bsl1] = x_seq1[i,-bsl1:]
                    self.bins[j][1][bctr[j],:bsl2] = x_seq2[i,-bsl2:]
                    self.bins[j][2][bctr[j],:] = y[i]

                    bctr[j] += 1
                    break

        self.num_samples = x_seq1.shape[0]
        self.dataset_len = int(sum([math.ceil( int( bs / self.batch_size ) ) 
                                    for bs in actual_bucketsizes]))
        self._permute()
        print('Finished bucketing.')


    def _permute(self):
        # Shuffle bins
        random.shuffle(self.bins)

        # Shuffle bin contents
        for i, (xbin1, xbin2, ybin) in enumerate(self.bins):
            index_array = np.random.permutation(xbin1.shape[0])
            self.bins[i] = (xbin1[index_array], xbin2[index_array], ybin[index_array])

    def on_epoch_end(self):
        self._permute()

    def __len__(self):
        """ Returns the number of minibatches in this sequence. """
        return self.dataset_len

    def __getitem__(self, idx):
        idx_begin, idx_end = self.batch_size*idx, self.batch_size*(idx+1)

        # Obtain bin index
        for i,(xbin1, xbin2, ybin) in enumerate(self.bins):

            rounded_bin = _roundto(xbin1.shape[0], self.batch_size)
             
            if idx_begin >= rounded_bin:
                idx_begin -= rounded_bin
                idx_end -= rounded_bin
                continue
                
            # Found bin
            idx_end = min(xbin1.shape[0], idx_end) # Clamp to end of bin

            N1 = xbin1[idx_begin:idx_end].shape[1]
            N2 = xbin2[idx_begin:idx_end].shape[1]

            N1 = N1 * np.ones( shape=(self.batch_size , 1 ))
            N2 = N2 * np.ones( shape=(self.batch_size , 1 ))

            return [xbin1[idx_begin:idx_end], xbin2[idx_begin:idx_end], N1, N2], ybin[idx_begin:idx_end]

        raise ValueError('out of bounds')

