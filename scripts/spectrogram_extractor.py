
from kapre.utils import Normalization2D

from kapre_modified import SpectrogramModified 
from kapre_modified import MelspectrogramModified
from kapre_variants import FixedNormalization2D, Scale2D, Shift2D

import keras.backend as K

from keras.models import Model
from keras.layers import Input, TimeDistributed, Flatten, Permute, Reshape

def get_spectrogram_extractor(dft_window=512, n_hop=200, n_mels=82, sr=16000, normalise='freq', \
            spectrogram_mean=None, spectrogram_std=None, decibel_melgram=False, spectrogram_type='MelSTFT', power=2.0, trainable_kernel=False):
    
    '''
    n_mels = 0  ->  plain uncompressed spectrogram

    Returns a Keras model which takes batches of waveform fragments (B x T x 1) and returns 
    batches of (mel)spectrogram fragments (B x T x F x 1). 
    return data in form (batch, time, freq, dummychannel)
    '''

    # inputs = Input(shape=(1, None))
    inputs = Input(shape=(None, 1))      ### (B x T x 1)

    data = Permute((2,1))(inputs)        ### (B x T x 1) -> (B x 1 x T)     (form required by kapre's Melspectrogram)
                                         ##  ("Permutation pattern, does not include the samples dimension. Indexing starts at 1.")
    
    if spectrogram_type == 'STFT':

        data = SpectrogramModified(n_dft=dft_window, n_hop=n_hop, input_shape=(1, None),
                             padding='same', trainable_kernel=trainable_kernel,
                             power_spectrogram=power, return_decibel_spectrogram=decibel_melgram,
                             name='stft_layer')(data)

    elif spectrogram_type == 'MelSTFT':

        data = MelspectrogramModified(n_dft=dft_window, n_hop=n_hop, input_shape=(1, None),
                             padding='same', sr=sr, n_mels=n_mels,
                             fmin=0.0, fmax=sr/2, power_melgram=power,
                             return_decibel_melgram=decibel_melgram, trainable_fb=False,
                             trainable_kernel=trainable_kernel,
                             name='mel_stft_layer')(data)

    assert normalise in ['batch', 'data_sample', 'channel', 'freq', 'time'] + \
                        ['freq_global_norm', 'freq_scale_only', 'freq_shift_only'] + \
                        ['none']

    if normalise=='freq_global_norm':
        data = FixedNormalization2D(str_axis='freq', mean=spectrogram_mean, std=spectrogram_std)(data)
        #print('*FIXED* NORMALISE MEL SPECTROGRAMS (FREQ)')
    elif normalise=='freq_scale_only': ## doesn't work
        data = Scale2D(str_axis='freq')(data)
        #print('ONLY SCALE NORMALISE MEL SPECTROGRAMS (FREQ)')
    elif normalise=='freq_shift_only':
        data = Shift2D(str_axis='freq')(data)
        #print('ONLY SHIFT NORMALISE MEL SPECTROGRAMS (FREQ)')
    elif normalise=='none':
        print("")
        #print('DO NOT NORMALISE MEL SPECTORGRAMS')
        ## do nothing further here
    else:
        assert normalise in ['batch', 'data_sample', 'channel', 'freq', 'time'] 
        #print('NORMALISE MEL SPECTROGRAMS by kapre builtin (%s)'%(normalise))
        data = Normalization2D(str_axis=normalise)(data)

    assert K.image_data_format() == 'channels_last'  ## this means data now:  (B x F x T x 1)
    
    if len(data.shape)==4:
        data = Permute((2,1,3))(data)    ## (B x F x T x 1)   ->    (B x T x F x 1)
    if len(data.shape)==5:
        data = Permute((3,1,2,4))(data)    ## (B x F x N x T x 1 )   ->    (B x T x F x N x 1)

    model = Model(inputs=inputs, outputs=data)
    #model.summary()

    return model
