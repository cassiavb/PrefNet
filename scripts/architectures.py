# -*- coding: utf-8 -*-
#/usr/bin/python3

from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, \
                UpSampling1D, Conv1D, ZeroPadding1D, Add, \
                BatchNormalization, Activation, LeakyReLU, \
                Dropout, Reshape, Lambda, Multiply, GRU, Permute, CuDNNGRU
import keras.layers as layers 
import keras.backend as K
from tensorflow.keras import initializers
import numpy as np
import tensorflow as tf
from keras.regularizers import l2

import sys
from spectrogram_extractor import get_spectrogram_extractor

DEBUG=False

def norm():
    return BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, \
                 beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', \
                 moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, \
                 beta_constraint=None, gamma_constraint=None)    


def linear_transform(dim, activation=None):
    return Conv1D(dim, 1, strides=1, padding='same', dilation_rate=1, activation=activation, \
                  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', \
                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, \
                  kernel_constraint=None, bias_constraint=None)

def conv_relu(dim, conv_width, padding='same', activation='relu', dilation_rate=1, conv_type='1D',L2regularizer=False, initialiser = 'glorot_uniform'):

    kernel_regularizer=None
    if L2regularizer:
        kernel_regularizer=l2(0.001)
    
    return Conv1D(dim, conv_width, strides=1, padding=padding, dilation_rate=dilation_rate, activation=activation, \
                     use_bias=True, kernel_initializer=initialiser, bias_initializer='zeros', \
                     kernel_regularizer=kernel_regularizer, bias_regularizer=None, activity_regularizer=None, \
                     kernel_constraint=None, bias_constraint=None)

def get_linear_model(dim, activation=None):

    model = linear_transform(dim, activation)

    return model

def get_encoder_model(input_dim=1, norm_data=False, include_gru=False, conv_dim = 40, enc_dim = 128, convlayers=6, convwidth=3, padding='same', dilation_rate=1, activation='relu', conv_type='1D', dropout_rate=0.0,L2regularizer=False):

    data = Input(shape=(None, input_dim))

    transformed_data = data

    initialiser = 'he_normal' # 'he_normal' # 'glorot_uniform'

    # Convolutional layers
    transformed_data = conv_relu(conv_dim, convwidth, padding=padding, dilation_rate=dilation_rate, activation=activation, conv_type=conv_type, L2regularizer=L2regularizer, initialiser=initialiser)(transformed_data)
    transformed_data = Dropout(rate=dropout_rate)(transformed_data)
    for subsequent_layer in range(convlayers-1):
        transformed_data = conv_relu(conv_dim, convwidth, padding=padding, dilation_rate=1, activation=activation, conv_type=conv_type,L2regularizer=L2regularizer, initialiser=initialiser)(transformed_data)
        transformed_data = Dropout(rate=dropout_rate)(transformed_data)

    if norm_data:
        transformed_data = norm()(transformed_data)

    if include_gru:

        transformed_data = CuDNNGRU(conv_dim, kernel_initializer=initialiser, recurrent_initializer='orthogonal', bias_initializer='zeros', \
            kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, \
            kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, \
            return_sequences=False, return_state=False, go_backwards=False, stateful=False)(transformed_data)

    model = Model(inputs=data, outputs=transformed_data)

    if DEBUG:
        print("Encoder model:")
        print(model.summary())

    return model

def get_predictor_model(input_dim=40, layers_size=[1], activation='linear', dropout_rate=0.0):
    '''
    Fully connected layer w sigmoid activation
    '''

    layers_size = np.asarray(layers_size, dtype=np.int)
    num_layers  = len(layers_size)

    ## check that last layer has one unit
    assert layers_size[-1] == 1
    
    data = Input(shape=(input_dim, ))

    transformed_data = Dense( units = layers_size[0], activation=activation)(data)

    for ( subsequent_layer, layer_size ) in enumerate( layers_size[1:] ):
        transformed_data = Dense( units = layer_size, activation=activation)(transformed_data)

    model = Model(inputs=data, outputs=transformed_data)

    if DEBUG:
        print("Predictor model:")
        print(model.summary())

    return model

def softmax_2dim( x ):
    max1 = K.max(x, axis=1, keepdims=True)
    max_value = K.max(max1, axis=2, keepdims=True)
    e = K.exp(x - max_value)
    s = K.sum(e, axis=(1,2), keepdims=True)
    return e / ( s )

def expand_repeat(x, tilepattern='', axis=-1):
    x = K.expand_dims(x, axis=axis)
    x = tf.tile(x, tilepattern)
    return x


def gru_and_attention_model(config, vocdim = 1, max_length=61280):

    ## Define two inputs
    input_A = Input(shape=(None, vocdim ), dtype="float32") # B x T1 X F 
    input_B = Input(shape=(None, vocdim ), dtype="float32") # B x T2 X F 
    Ta      = layers.Input(shape=( 1, ), dtype="float32")   # B x 1 
    Tb      = layers.Input(shape=( 1, ), dtype="float32")   # B x 1

    if config.get('model_type','GRU') == 'GRU': # GRU based model
        include_gru = True
    else:
        include_gru = False

    ## Define encoder model 
    encoder_model = get_encoder_model(input_dim=config.get('n_mels',40), norm_data=config.get('norm_enc_data',False), \
            enc_dim=config.get('dft_window', 512), convlayers=config.get('convlayers',2), \
            convwidth=config.get('convwidth',3), dilation_rate=config.get('dilation_rate',1), \
            conv_type=config.get('conv_type','1D'), dropout_rate=config.get('dropout_rate',0.0), \
            L2regularizer=config.get('L2regularizer',False), include_gru=include_gru, \
            conv_dim=config.get('conv_dim', 64))

    ## Multihead attention settings
    kv_dim   = int( config.get('conv_dim',40) / 2 )
    head_dim = int( kv_dim / config.get('n_heads',1) )
    D_dim    = head_dim
    if config.get('multihead_type', 'Add') == 'Concatenate':
        D_dim = head_dim * config.get('n_heads',1)

    ## Define predictor model
    fc_layers_size = np.asarray(config.get('fc_layers_size',[10,1]))
    if include_gru: # GRU based model
        predictor_model = get_predictor_model(input_dim= 2*config.get('conv_dim',40), layers_size=fc_layers_size, activation=config.get('fc_activation', 'linear'))
    else: # Attention based model
        predictor_model = get_predictor_model(input_dim= D_dim, layers_size=fc_layers_size, activation=config.get('fc_activation', 'linear'))

    ## Define linear transform models
    linear_model_K = get_linear_model(kv_dim)
    linear_model_V = get_linear_model(kv_dim)
    linear_model_K_head = get_linear_model(head_dim)
    linear_model_V_head = get_linear_model(head_dim)

    ## Input is made of pre computed DFT
    dft_A = input_A
    dft_B = input_B

    ## Encode the inputs using a CNN
    encoded_A = encoder_model(dft_A) # (B x N1 x F) -> (B x N1 x D)
    encoded_B = encoder_model(dft_B) # (B x N2 x F) -> (B x N2 x D)

    if include_gru: # GRU based model

        D = layers.concatenate([encoded_A , encoded_B], axis=1) # encoded is: None, D

        # Apply function to it
        fAB = predictor_model( D ) # B x 1
        DBA = layers.concatenate([encoded_B, encoded_A], axis=1)
        fBA = predictor_model( DBA ) # B x 1

        # Sum and apply sigmoid
        fABBA = layers.subtract([ fAB, fBA ])
        prob  = layers.Activation('sigmoid')(fABBA)

    else: # Attention based model

        ## Separating Keys and Values
        K_A = linear_model_K(encoded_A)
        K_B = linear_model_K(encoded_B)
        V_A = linear_model_V(encoded_A)
        V_B = linear_model_V(encoded_B)

        N1 = K.cast([1,1,1,Ta[0,0]], 'int32') 
        N2 = K.cast([1,1,1,Tb[0,0]], 'int32')

        # Multihead attention
        for head in range( config.get('n_heads',1) ):

            # Apply linear projections to KA, KB, VA and VB
            if kv_dim != head_dim:
                # Apply linear projections to KA, KB, VA and VB
                K_A_h = linear_model_K_head(K_A)
                K_B_h = linear_model_K_head(K_B)
                V_A_h = linear_model_V_head(V_A)
                V_B_h = linear_model_V_head(V_B)
            else:
                K_A_h = K_A
                K_B_h = K_B
                V_A_h = V_A
                V_B_h = V_B

            # Calculate attention matrix
            A = layers.dot( [ K_A_h , K_B_h ] , axes = 2) # B x N1 x D  .  B x N2 x D  ==> B x N1 x N2
            A = layers.Lambda( softmax_2dim ) ( A ) # Softmax along both 'time' axes => B x None x None
            A = layers.Lambda(lambda x: K.expand_dims(x, axis=1))(A) # B x 1 x N1 x N2

            # Expand dimension and repeat
            V_A_h  = layers.Lambda( expand_repeat, arguments={'tilepattern': N2, 'axis': 3} ) ( V_A_h ) # B x N1 X D x N2
            V_B_h  = layers.Lambda( expand_repeat, arguments={'tilepattern': N1, 'axis': 3} ) ( V_B_h ) # B x N2 X D x N1

            # Permute so that B x D x N1 X N2
            V_A_h = layers.Lambda(lambda x: K.permute_dimensions(x, (0,2,1,3)))(V_A_h)
            V_B_h = layers.Lambda(lambda x: K.permute_dimensions(x, (0,2,3,1)))(V_B_h)

            # Calculate difference
            difference = layers.subtract( [ V_A_h , V_B_h ]) # B x D x N1 X N2

            # Apply weight (attention matrix)
            D_h = layers.multiply([difference , A]) # B x D x N1 X N2  .  B x 1 x N1 X N2  ==> B x D x N1 X N2
            # Sum across both time axis
            D_h = layers.Lambda(lambda x: K.sum( x , axis=(2,3)))(D_h) # B x D
            D_h = layers.Lambda(lambda x: K.reshape( x , ( -1, head_dim ) )) (D_h) # None x None -> None x D # otherwise Dense wont work

            # 'Accumulate' across heads
            if head == 0:
                D = D_h
            else:
                if config.get('multihead_type', 'Add') == 'Add':
                    D = layers.add([D , D_h])
                elif config.get('multihead_type', 'Add') == 'Concatenate':
                    D = layers.concatenate([D , D_h], axis=1)

        # Apply function to it
        fAB = predictor_model( D ) # B x 1
        DBA = layers.Lambda(lambda x: tf.negative(x))(D)
        fBA = predictor_model( DBA ) # B x 1

        # Sum and apply sigmoid
        fABBA = layers.subtract([ fAB, fBA ])
        prob  = layers.Activation('sigmoid')(fABBA)


    model = Model( inputs = [input_A , input_B, Ta, Tb], outputs = prob)

    if DEBUG:
        print("Final model:")
        print(model.summary())

    return model





