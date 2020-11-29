
from kapre.utils import Normalization2D
import keras.backend as K
from tensorflow import atan

class FixedNormalization2D(Normalization2D):
    """
    As kapre's Normalization2D, but initialise with precomputed ('fixed') mean and std,
    rather than computing these on the batch to be normalised. 
    """

    # def __init__(self, str_axis=None, int_axis=None, image_data_format='default',
    #              eps=1e-10, mean=None, std=None, **kwargs):

    def __init__(self, **kwargs):
        self.mean = kwargs['mean']
        self.std = kwargs['std']
        del kwargs['mean']
        del kwargs['std']
        super(FixedNormalization2D, self).__init__(**kwargs)

    def call(self, x, mask=None):
        # if self.axis == -1:
        #     mean = K.mean(x, axis=[3, 2, 1, 0], keepdims=True)
        #     std = K.std(x, axis=[3, 2, 1, 0], keepdims=True)
        # elif self.axis in (0, 1, 2, 3):
        #     all_dims = [0, 1, 2, 3]
        #     del all_dims[self.axis]
        return (x - self.mean) / (self.std + self.eps)

class Scale2D(Normalization2D):
    """
    As kapre's Normalization2D, but adds the mean back in.
    """
    def call(self, x, mask=None):
        if self.axis == -1:
            mean = K.mean(x, axis=[3, 2, 1, 0], keepdims=True)
            std = K.std(x, axis=[3, 2, 1, 0], keepdims=True)
        elif self.axis in (0, 1, 2, 3):
            all_dims = [0, 1, 2, 3]
            del all_dims[self.axis]
            mean = K.mean(x, axis=all_dims, keepdims=True)
            std = K.std(x, axis=all_dims, keepdims=True)
        return ((x - mean) / (std + self.eps)) + mean

class Shift2D(Normalization2D):
    """
    As kapre's Normalization2D, but without scaling.
    """
    def call(self, x, mask=None):
        if self.axis == -1:
            mean = K.mean(x, axis=[3, 2, 1, 0], keepdims=True)
            # std = K.std(x, axis=[3, 2, 1, 0], keepdims=True)
        elif self.axis in (0, 1, 2, 3):
            all_dims = [0, 1, 2, 3]
            del all_dims[self.axis]
            mean = K.mean(x, axis=all_dims, keepdims=True)
            # std = K.std(x, axis=all_dims, keepdims=True)
        return (x - mean) 
