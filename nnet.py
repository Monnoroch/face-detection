import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.utils import as_tuple


class NeuralNet(object):
    def __init__(self, means=None, stds=None):
        self._means = means
        self._stds = stds
        self._num_classes = None
        sizes = self.sizes()
        self.inputs = list(map(T.tensor4, map(lambda v: "input" + str(v[0]) + "_" + str(v[1]), zip(range(len(sizes)), sizes))))

    def build_norm(self, inputs, gpu):
        if self._means is None:
            means = len(inputs) * [128.0]
        else:
            means = self._means
        if self._stds is None:
            stds = len(inputs) * [128.0]
        else:
            stds = self._stds

        self.build(list(map(lambda x: (x[0] - x[1]) / x[2], zip(inputs, means, stds))), gpu)
        return self.net

    def load(self, fname):
        epoch = 0
        with np.load(fname) as f:
            cnt = len(f.files)
            if "epoch" in f.keys():
                cnt -= 1
                epoch = f["epoch"]
            if "means" in f.keys():
                cnt -= 1
                if self._means is None:
                    self._means = f["means"]
            if "stds" in f.keys():
                cnt -= 1
                if self._stds is None:
                    self._stds = f["stds"]

            lasagne.layers.set_all_param_values(self.net, [f['arr_%d' % i] for i in range(cnt)])
        return epoch

    def load_build_norm(self, fname, inputs, gpu):
        with np.load(fname) as f:
            cnt = len(f.files)
            if "epoch" in f.keys():
                cnt -= 1
            if "means" in f.keys():
                cnt -= 1
                if self._means is None:
                    self._means = f["means"]
            if "stds" in f.keys():
                cnt -= 1
                if self._stds is None:
                    self._stds = f["stds"]

            if self._means is None:
                means = np.array(len(inputs) * [128.0], dtype=theano.config.floatX)
            else:
                means = np.array(self._means, dtype=theano.config.floatX)
            if self._stds is None:
                stds = np.array(len(inputs) * [128.0], dtype=theano.config.floatX)
            else:
                stds = np.array(self._stds, dtype=theano.config.floatX)

            self.build(list(map(lambda x: (x[0] - x[1]) / x[2], zip(inputs, means, stds))), gpu)

            lasagne.layers.set_all_param_values(self.net, [np.array(f['arr_%d' % i], dtype=theano.config.floatX) for i in range(cnt)])
        return self.net

    def load(self, fname):
        epoch = 0
        with np.load(fname) as f:
            cnt = len(f.files)
            if "epoch" in f.keys():
                cnt -= 1
                epoch = f["epoch"]
            if "means" in f.keys():
                cnt -= 1
                if self._means is None:
                    self._means = f["means"]
            if "stds" in f.keys():
                cnt -= 1
                if self._stds is None:
                    self._stds = f["stds"]

            lasagne.layers.set_all_param_values(self.net, [np.array(f['arr_%d' % i], dtype=theano.config.floatX) for i in range(cnt)])
        return epoch

    def save(self, epoch, fname):
        np.savez(fname, epoch=epoch, means=self._means, stds=self._stds, *lasagne.layers.get_all_param_values(self.net))

    def _get_layers(self, gpu):
        import lasagne
        conv2d, max_pool = GoodConv2DLayer, GoodMaxPool2DLayer
        if gpu:
            import lasagne.layers.dnn

            class GoodGPUConv2DLayer(lasagne.layers.dnn.Conv2DDNNLayer):
                def __init__(self, incoming, num_filters, filter_size, stride=(1, 1), pad="same", untie_biases=False,
                    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.1),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    dimshuffle=True, flip_filters=False, **kwargs):
                    super().__init__(
                        incoming=incoming, num_filters=num_filters, filter_size=filter_size, stride=stride, pad=pad, untie_biases=untie_biases,
                        W=W, b=b, nonlinearity=nonlinearity,
                        flip_filters=flip_filters, **kwargs
                    )

            class GoodGPUMaxPool2DLayer(lasagne.layers.dnn.MaxPool2DDNNLayer):
                def __init__(self, incoming, pool_size=(3, 3), stride=2, ignore_border=True, dimshuffle=True, **kwargs):
                    super().__init__(incoming=incoming, pool_size=pool_size, stride=stride, ignore_border=ignore_border, **kwargs)

            conv2d, max_pool = GoodGPUConv2DLayer, GoodGPUMaxPool2DLayer
        return conv2d, max_pool

    def num_classes(self):
        return self._num_classes

    def names(self):
        sizes = self.sizes()
        return list(map(lambda x: "input" + str(x[0]) + "_" + str(x[1]), zip(range(len(sizes)), sizes)))

    # implement this for children classes
    def sizes(self):
        return []

    # implement this for children classes
    def dataset_class(self, all=False):
        return None

    # implement this for children classes
    def build(self, inputs, gpu):
        self.net = None
        return self.net

class GoodDenseLayer(lasagne.layers.DenseLayer):
    def __init__(self, incoming, num_units,
        W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.1),
        nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super().__init__(incoming=incoming, num_units=num_units, W=W, b=b, nonlinearity=nonlinearity, **kwargs)

class GoodClsDenseLayer(lasagne.layers.DenseLayer):
    def __init__(self, incoming, num_units,
        W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.1),
        nonlinearity=lasagne.nonlinearities.softmax, **kwargs):
        super().__init__(incoming=incoming, num_units=num_units, W=W, b=b, nonlinearity=nonlinearity, **kwargs)

class GoodConv2DLayer(lasagne.layers.Conv2DLayer):
    # temporarily fixed theano bug (https://groups.google.com/forum/#!topic/lasagne-users/Amj6YE8gX7w)
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1), pad="same", untie_biases=False,
        W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.1),
        nonlinearity=lasagne.nonlinearities.rectify,
        convolution=theano.tensor.nnet.conv2d, **kwargs):
        if pad == "same":
            pad = filter_size[0] // 2

        super().__init__(
            incoming=incoming, num_filters=num_filters, filter_size=filter_size, stride=stride, pad=pad, untie_biases=untie_biases,
            W=W, b=b, nonlinearity=nonlinearity, convolution=convolution, **kwargs
        )

class GoodMaxPool2DLayer(lasagne.layers.MaxPool2DLayer):
    def __init__(self, incoming, pool_size=(3, 3), stride=2, pad=(0, 0), ignore_border=True, **kwargs):
        super().__init__(incoming=incoming, pool_size=pool_size, stride=stride, pad=pad, ignore_border=ignore_border, **kwargs)

def get_layers(gpu):
    return NeuralNet()._get_layers(gpu)

def conv_output_length(input_length, filter_size, stride, pad=0):
    if input_length is None:
        return None
    if pad == 'valid':
        output_length = input_length - filter_size + 1
    elif pad == 'full':
        output_length = input_length + filter_size - 1
    elif pad == 'same':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length

class ConvMMax2DLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad="same", untie_biases=False,
                 W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.1),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 convolution=T.nnet.conv2d, **kwargs):
        super(ConvMMax2DLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = lasagne.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, 2)
        self.stride = as_tuple(stride, 2)
        self.untie_biases = untie_biases
        self.convolution = convolution

        if pad == 'valid':
            self.pad = (0, 0)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, 2, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2], self.
                                output_shape[3])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0],
                self.filter_size[1])

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * 2

        output_rows = conv_output_length(input_shape[2],
                                         self.filter_size[0],
                                         self.stride[0],
                                         pad[0])

        output_columns = conv_output_length(input_shape[3],
                                            self.filter_size[1],
                                            self.stride[1],
                                            pad[1])

        return (input_shape[0], self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, input_shape=None, **kwargs):
        # The optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = self.input_shape

        if self.stride == (1, 1) and self.pad == 'same':
            # simulate same convolution by cropping a full convolution
            conved1 = self.convolution(input, self.W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode='full')
            shift_x = (self.filter_size[0] - 1) // 2
            shift_y = (self.filter_size[1] - 1) // 2
            conved1 = conved1[:, :, shift_x:input.shape[2] + shift_x,
                            shift_y:input.shape[3] + shift_y]

            mirrored = input[:,:,:,::-1]
            conved2 = self.convolution(mirrored, self.W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode='full')
            shift_x = (self.filter_size[0] - 1) // 2
            shift_y = (self.filter_size[1] - 1) // 2
            conved2 = conved2[:, :, shift_x:mirrored.shape[2] + shift_x,
                            shift_y:mirrored.shape[3] + shift_y]
        else:
            # no padding needed, or explicit padding of input needed
            if self.pad == 'full':
                border_mode = 'full'
                pad = [(0, 0), (0, 0)]
            elif self.pad == 'same':
                border_mode = 'valid'
                pad = [(self.filter_size[0] // 2,
                        (self.filter_size[0] - 1) // 2),
                       (self.filter_size[1] // 2,
                        (self.filter_size[1] - 1) // 2)]
            else:
                border_mode = 'valid'
                pad = [(self.pad[0], self.pad[0]), (self.pad[1], self.pad[1])]
            if pad != [(0, 0), (0, 0)]:
                input = padding.pad(input, pad, batch_ndim=2)
                input_shape = (input_shape[0], input_shape[1],
                               None if input_shape[2] is None else
                               input_shape[2] + pad[0][0] + pad[0][1],
                               None if input_shape[3] is None else
                               input_shape[3] + pad[1][0] + pad[1][1])
            conved1 = self.convolution(input, self.W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode=border_mode)
            mirrored = input[:,:,:,::-1]
            conved2 = self.convolution(mirrored, self.W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode=border_mode)


        conved = T.max([conved1, conved2], axis=0)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)
