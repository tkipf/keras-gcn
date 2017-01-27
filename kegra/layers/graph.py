from __future__ import print_function

from keras import activations, initializations
from keras import regularizers
from keras.engine import Layer
from keras.engine.topology import Node
import keras.backend as K


def GraphInput(name=None, dtype=K.floatx(), sparse=False,
          tensor=None):
    """ TODO: Docstring """
    shape = (None, None)
    input_layer = GraphInputLayer(batch_input_shape=shape,
                                name=name, sparse=sparse, input_dtype=dtype)
    outputs = input_layer.inbound_nodes[0].output_tensors
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


class GraphInputLayer(Layer):
    """ TODO: Docstring """
    def __init__(self, input_shape=None, batch_input_shape=None,
                 input_dtype=None, sparse=False, name=None):
        self.input_spec = None
        self.supports_masking = False
        self.uses_learning_phase = False
        self.trainable = False
        self.built = True

        self.inbound_nodes = []
        self.outbound_nodes = []

        self.trainable_weights = []
        self.non_trainable_weights = []
        self.regularizers = []
        self.constraints = {}

        self.sparse = sparse

        if not name:
            prefix = 'input'
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        if not batch_input_shape:
            assert input_shape, 'An Input layer should be passed either a `batch_input_shape` or an `input_shape`.'
            batch_input_shape = (None,) + tuple(input_shape)
        else:
            batch_input_shape = tuple(batch_input_shape)
        if not input_dtype:
            input_dtype = K.floatx()

        self.batch_input_shape = batch_input_shape
        self.input_dtype = input_dtype

        input_tensor = K.placeholder(shape=batch_input_shape,
                                     dtype=input_dtype,
                                     sparse=self.sparse,
                                     name=self.name)

        input_tensor._uses_learning_phase = False
        input_tensor._keras_history = (self, 0, 0)
        shape = input_tensor._keras_shape
        Node(self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=[input_tensor],
             output_tensors=[input_tensor],
             input_masks=[None],
             output_masks=[None],
             input_shapes=[shape],
             output_shapes=[shape])

    def get_config(self):
        config = {'batch_input_shape': self.batch_input_shape,
                  'input_dtype': self.input_dtype,
                  'sparse': self.sparse,
                  'name': self.name}
        return config



class GraphConvolution(Layer):
    """TODO: Docstring"""
    def __init__(self, output_dim, support=1, init='glorot_uniform',
                 activation='linear', weights=None, W_regularizer=None,
                 b_regularizer=None, bias=False, **kwargs):
        self.init = initializations.get(init)
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

    def get_output_shape_for(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.output_dim)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 2
        self.input_dim = features_shape[1]

        self.W = self.init((self.input_dim * self.support, self.output_dim),
                           name='{}_W'.format(self.name))
        if self.bias:
            self.b = K.zeros((self.output_dim,),
                             name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

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