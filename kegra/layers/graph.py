from __future__ import print_function

from keras import activations, initializers
from keras import regularizers
from keras.engine import Layer
import keras.backend as K


class GraphConvolution(Layer):
    """TODO: Docstring"""
    def __init__(self, output_dim, support=1, init='glorot_uniform',
                 activation='linear', weights=None, W_regularizer=None,
                 b_regularizer=None, bias=False, **kwargs):
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim  # number of features per node
        self.support = support  # filter support / number of weights

        assert support >= 1

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.bias = bias
        self.initial_weights = weights

        # these will be defined during build()
        self.input_dim = None
        self.W = None
        self.b = None

        super(GraphConvolution, self).__init__(**kwargs)

    # def get_output_shape_for(self, input_shapes):
    #     features_shape = input_shapes[0]
    #     output_shape = (features_shape[0], self.output_dim)
    #     return output_shape  # (batch_size, output_dim)

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.output_dim)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 2
        self.input_dim = features_shape[1]

        self.W = self.add_weight((self.input_dim * self.support, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer)
        if self.bias:
            self.b = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, inputs, mask=None):
        features = inputs[0]
        A = inputs[1:]  # list of basis functions

        # convolve
        supports = list()
        for i in range(self.support):
            supports.append(K.dot(A[i], features))
        supports = K.concatenate(supports, axis=1)
        output = K.dot(supports, self.W)

        if self.bias:
            output += self.b
        return self.activation(output)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))