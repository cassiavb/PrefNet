# -*- coding: utf-8 -*-
#/usr/bin/python3

'''
Adjusted Kapre Spectrogram and Melspectrogram classes which don't 
apply range compression in the case that melgram=True
'''

from kapre.time_frequency import Spectrogram, Melspectrogram
import numpy as np
from keras import backend as K


def amplitude_to_decibel(x, amin=1e-10, dynamic_range=80.0):
    """[K] Convert (linear) amplitude to decibel (log10(x)).
    x: Keras tensor or variable. 
    amin: minimum amplitude. amplitude smaller than `amin` is set to this.
    dynamic_range: dynamic_range in decibel
    """
    log_spec = 10 * K.log(K.maximum(x, amin)) / np.log(10).astype(K.floatx())

    #### These lines which to batch-wise range compression removed:-
    ## log_spec = log_spec - K.max(log_spec)  # [-?, 0]
    ## log_spec = K.maximum(log_spec, -1 * dynamic_range)  # [-80, 0]
    return log_spec



class SpectrogramModified(Spectrogram):

    def call(self, x):
        output = self._spectrogram_mono(x[:, 0:1, :])
        if self.is_mono is False:
            for ch_idx in range(1, self.n_ch):
                output = K.concatenate((output,
                                        self._spectrogram_mono(x[:, ch_idx:ch_idx + 1, :])),
                                       axis=self.ch_axis_idx)
        if self.power_spectrogram != 2.0:
            output = K.pow(K.sqrt(output), self.power_spectrogram)
        if self.return_decibel_spectrogram:
            output = amplitude_to_decibel(output) ## only difference from non-modified class
        return output


class MelspectrogramModified(Melspectrogram):

    def call(self, x):
        power_spectrogram = super(Melspectrogram, self).call(x)
        # now,  channels_first: (batch_sample, n_ch, n_freq, n_time)
        #       channels_last: (batch_sample, n_freq, n_time, n_ch)
        if self.image_data_format == 'channels_first':
            power_spectrogram = K.permute_dimensions(power_spectrogram, [0, 1, 3, 2])
        else:
            power_spectrogram = K.permute_dimensions(power_spectrogram, [0, 3, 2, 1])
        # now, whatever image_data_format, (batch_sample, n_ch, n_time, n_freq)
        output = K.dot(power_spectrogram, self.freq2mel)
        if self.image_data_format == 'channels_first':
            output = K.permute_dimensions(output, [0, 1, 3, 2])
        else:
            output = K.permute_dimensions(output, [0, 3, 2, 1])
        if self.power_melgram != 2.0:
            output = K.pow(K.sqrt(output), self.power_melgram)
        if self.return_decibel_melgram:
            output = amplitude_to_decibel(output) ## only difference from non-modified class
        return output

