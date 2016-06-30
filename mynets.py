import math
import lasagne
import batchnorm
from nnet import *
from datasets import *


class NetCls(NeuralNet):
    def __init__(self, *argv, **args):
        super().__init__(*argv, **args)
        self._num_classes = 2

    def dataset_class(self, all=False):
        if all:
            return AllBestClsDataSet
        return BestClsDataSet

class NetCal(NeuralNet):
    def __init__(self, *argv, **args):
        super().__init__(*argv, **args)
        self._num_classes = CalClasses().count()

    def dataset_class(self, all=False):
        if all:
            return AllBetterCalDataSet
        return BetterCalDataSet

class Net12(NetCls):
    def build(self, inputs, gpu):
        conv, max_pool = self._get_layers(gpu)
        self.net = build_cnn12(inputs[0], self._num_classes, conv, max_pool)
        return self.net

    def sizes(self):
        return [12]

class Net12MMax(NetCls):
    def build(self, inputs, gpu):
        conv, max_pool = self._get_layers(gpu)
        self.net = build_cnn12_mmax(inputs[0], self._num_classes, conv, max_pool)
        return self.net

    def sizes(self):
        return [12]

class Net12Cal(NetCal):
    def build(self, inputs, gpu):
        conv, max_pool = self._get_layers(gpu)
        self.net = build_cnn12_cal(inputs[0], self._num_classes, conv, max_pool)
        return self.net

    def sizes(self):
        return [12]

class Net24(NetCls):
    def build(self, inputs, gpu):
        conv, max_pool = self._get_layers(gpu)
        self.net = build_cnn24(inputs, self._num_classes, conv, max_pool)
        return self.net

    def sizes(self):
        return [24, 12]

class Net24Sp(NetCls):
    def build(self, inputs, gpu):
        conv, max_pool = self._get_layers(gpu)
        self.net = build_cnn24_sp(inputs[0], self._num_classes, conv, max_pool)
        return self.net

    def sizes(self):
        return [24]

class Net24Cal(NetCal):
    def build(self, inputs, gpu):
        conv, max_pool = self._get_layers(gpu)
        self.net = build_cnn24_cal(inputs[0], self._num_classes, conv, max_pool)
        return self.net

    def sizes(self):
        return [24]

class Net48(NetCls):
    def build(self, inputs, gpu):
        conv, max_pool = self._get_layers(gpu)
        self.net = build_cnn48(inputs, self._num_classes, conv, max_pool)
        return self.net

    def sizes(self):
        return [48, 24, 12]

class Net482X(NetCls):
    def build(self, inputs, gpu):
        conv, max_pool = self._get_layers(gpu)
        self.net = build_cnn48_2x(inputs, self._num_classes, conv, max_pool)
        return self.net

    def sizes(self):
        return [48, 24, 12]

class Net48MMax(NetCls):
    def build(self, inputs, gpu):
        conv, max_pool = self._get_layers(gpu)
        self.net = build_cnn48_mmax(inputs, self._num_classes, conv, max_pool)
        return self.net

    def sizes(self):
        return [48, 24, 12]

class Net48Cal(NetCal):
    def build(self, inputs, gpu):
        conv, max_pool = self._get_layers(gpu)
        self.net = build_cnn48_cal(inputs[0], self._num_classes, conv, max_pool)
        return self.net

    def sizes(self):
        return [48]

class Net64(NetCls):
    def build(self, inputs, gpu):
        conv, max_pool = self._get_layers(gpu)
        self.net = build_cnn64(inputs[0], self._num_classes, conv, max_pool)
        return self.net

    def sizes(self):
        return [64]

class Net64Inc(NetCls):
    def build(self, inputs, gpu):
        conv, max_pool = self._get_layers(gpu)
        self.net = build_cnn64_inc(inputs[0], self._num_classes, conv, max_pool)
        return self.net

    def sizes(self):
        return [64]

class Net64Mp(NetCls):
    def build(self, inputs, gpu):
        conv, max_pool = self._get_layers(gpu)
        self.net = build_cnn64_mp(inputs, self._num_classes, conv, max_pool)
        return self.net

    def sizes(self):
        return [64, 48, 24]

class NetSmiles(NetCls):
    def __init__(self, *argv, **args):
        super().__init__(*argv, **args)
        self._num_classes = 17

    def build(self, inputs, gpu):
        conv, max_pool = self._get_layers(gpu)
        self.net = build_cnn48(inputs, self._num_classes, conv, max_pool)
        return self.net

    def sizes(self):
        return [48, 24, 12]

    def dataset_class(self, all=False):
        if all:
            return AllBestSmilesDataSet
        return BestSmilesDataSet


def build_cnn12(input_var, num_classes, conv, max_pool):
    network = lasagne.layers.InputLayer(shape=(None, 3, 12, 12), input_var=input_var)
    network = lasagne.layers.dropout(network, p=.1)
    network = conv(network, num_filters=16, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = GoodDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=16)
    network = batchnorm.batch_norm(network)
    network = GoodClsDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=num_classes)
    return network

def build_cnn12_mmax(input_var, num_classes, conv, max_pool):
    network = lasagne.layers.InputLayer(shape=(None, 3, 12, 12), input_var=input_var)
    network = lasagne.layers.dropout(network, p=.1)
    network = ConvMMax2DLayer(network, num_filters=16, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = GoodDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=16)
    network = batchnorm.batch_norm(network)
    network = GoodClsDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=num_classes)
    return network

def build_cnn12_cal(input_var, num_classes, conv, max_pool):
    network = lasagne.layers.InputLayer(shape=(None, 3, 12, 12), input_var=input_var)
    network = lasagne.layers.dropout(network, p=.1)
    network = conv(network, num_filters=16, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = GoodDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=128)
    network = batchnorm.batch_norm(network)
    network = GoodClsDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=num_classes)
    return network

def build_cnn24(input_var, num_classes, conv, max_pool):
    network12 = lasagne.layers.InputLayer(shape=(None, 3, 12, 12), input_var=input_var[1])
    network12 = lasagne.layers.dropout(network12, p=.1)
    network12 = conv(network12, num_filters=16, filter_size=(3, 3))
    network12 = batchnorm.batch_norm(network12)
    network12 = max_pool(network12)
    network12 = GoodDenseLayer(lasagne.layers.dropout(network12, p=.3), num_units=16)
    network12 = batchnorm.batch_norm(network12)

    network = lasagne.layers.InputLayer(shape=(None, 3, 24, 24), input_var=input_var[0])
    network = lasagne.layers.dropout(network, p=.1)
    network = conv(network, num_filters=64, filter_size=(5, 5))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = GoodDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=128)
    network = batchnorm.batch_norm(network)

    network = lasagne.layers.ConcatLayer([network, network12])
    network = GoodClsDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=num_classes)
    return network

def build_cnn24_sp(input_var, num_classes, conv, max_pool):
    network = lasagne.layers.InputLayer(shape=(None, 3, 24, 24), input_var=input_var)
    network = lasagne.layers.dropout(network, p=.1)
    network = conv(network, num_filters=64, filter_size=(5, 5))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = GoodDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=128)
    network = batchnorm.batch_norm(network)
    network = GoodClsDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=num_classes)
    return network

def build_cnn24_cal(input_var, num_classes, conv, max_pool):
    network = lasagne.layers.InputLayer(shape=(None, 3, 24, 24), input_var=input_var)
    network = lasagne.layers.dropout(network, p=.1)
    network = conv(network, num_filters=32, filter_size=(5, 5))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = GoodDenseLayer(lasagne.layers.dropout(network, p=.5), num_units=64)
    network = batchnorm.batch_norm(network)
    network = GoodClsDenseLayer(lasagne.layers.dropout(network, p=.5), num_units=num_classes)
    return network

def build_cnn48(input_var, num_classes, conv, max_pool):
    network12 = lasagne.layers.InputLayer(shape=(None, 3, 12, 12), input_var=input_var[2])
    network12 = lasagne.layers.dropout(network12, p=.1)
    network12 = conv(network12, num_filters=16, filter_size=(3, 3))
    network12 = batchnorm.batch_norm(network12)
    network12 = max_pool(network12)
    network12 = GoodDenseLayer(lasagne.layers.dropout(network12, p=.3), num_units=16)
    network12 = batchnorm.batch_norm(network12)

    network24 = lasagne.layers.InputLayer(shape=(None, 3, 24, 24), input_var=input_var[1])
    network24 = lasagne.layers.dropout(network24, p=.1)
    network24 = conv(network24, num_filters=32, filter_size=(5, 5))
    network24 = batchnorm.batch_norm(network24)
    network24 = max_pool(network24)
    network24 = GoodDenseLayer(lasagne.layers.dropout(network24, p=.3), num_units=64)
    network24 = batchnorm.batch_norm(network24)

    network = lasagne.layers.InputLayer(shape=(None, 3, 48, 48), input_var=input_var[0])
    network = lasagne.layers.dropout(network, p=.1)
    network = conv(network, num_filters=64, filter_size=(5, 5))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = conv(network, num_filters=64, filter_size=(5, 5))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = conv(network, num_filters=64, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = conv(network, num_filters=64, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = GoodDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=256)
    network = batchnorm.batch_norm(network)

    network = lasagne.layers.ConcatLayer([network, network24, network12])
    network = GoodClsDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=num_classes)
    return network

def build_cnn48_2x(input_var, num_classes, conv, max_pool):
    network12 = lasagne.layers.InputLayer(shape=(None, 3, 12, 12), input_var=input_var[2])
    network12 = lasagne.layers.dropout(network12, p=.1)
    network12 = conv(network12, num_filters=32, filter_size=(3, 3))
    network12 = batchnorm.batch_norm(network12)
    network12 = max_pool(network12)
    network12 = GoodDenseLayer(lasagne.layers.dropout(network12, p=.3), num_units=32)
    network12 = batchnorm.batch_norm(network12)

    network24 = lasagne.layers.InputLayer(shape=(None, 3, 24, 24), input_var=input_var[1])
    network24 = lasagne.layers.dropout(network24, p=.1)
    network24 = conv(network24, num_filters=64, filter_size=(5, 5))
    network24 = batchnorm.batch_norm(network24)
    network24 = max_pool(network24)
    network24 = GoodDenseLayer(lasagne.layers.dropout(network24, p=.3), num_units=64)
    network24 = batchnorm.batch_norm(network24)

    network = lasagne.layers.InputLayer(shape=(None, 3, 48, 48), input_var=input_var[0])
    network = lasagne.layers.dropout(network, p=.1)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = GoodDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=256)
    network = batchnorm.batch_norm(network)

    network = lasagne.layers.ConcatLayer([network, network24, network12])
    network = GoodClsDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=num_classes)
    return network

def build_cnn64(input_var, num_classes, conv, max_pool):
    network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
    network = lasagne.layers.dropout(network, p=.1)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = GoodDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=256)
    network = batchnorm.batch_norm(network)
    network = GoodClsDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=num_classes)
    return network

def build_cnn64_inc(input_var, num_classes, conv, max_pool):
    network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
    network = lasagne.layers.dropout(network, p=.1)

    b1 = conv(network, num_filters=32, filter_size=(1, 1))
    b2 = conv(network, num_filters=48, filter_size=(1, 1))
    b2 = conv(b2, num_filters=64, filter_size=(3, 3))
    b3 = conv(network, num_filters=8, filter_size=(1, 1))
    b3 = conv(b3, num_filters=16, filter_size=(5, 5))
    network = lasagne.layers.ConcatLayer([b1, b2, b3], axis=1)
    
    network = max_pool(network, pad=(1, 1))
    # network = batchnorm.batch_norm(network)

    b1 = conv(network, num_filters=64, filter_size=(1, 1))
    b2 = conv(network, num_filters=64, filter_size=(1, 1))
    b2 = conv(b2, num_filters=96, filter_size=(3, 3))
    b3 = conv(network, num_filters=16, filter_size=(1, 1))
    b3 = conv(b3, num_filters=48, filter_size=(5, 5))
    network = lasagne.layers.ConcatLayer([b1, b2, b3], axis=1)

    network = max_pool(network, pad=(1, 1))
    # network = batchnorm.batch_norm(network)
    
    b1 = conv(network, num_filters=96, filter_size=(1, 1))
    b2 = conv(network, num_filters=48, filter_size=(1, 1))
    b2 = conv(b2, num_filters=104, filter_size=(3, 3))
    b3 = conv(network, num_filters=8, filter_size=(1, 1))
    b3 = conv(b3, num_filters=24, filter_size=(5, 5))
    network = lasagne.layers.ConcatLayer([b1, b2, b3], axis=1)

    network = max_pool(network, pad=(1, 1))
    # network = batchnorm.batch_norm(network)
    
    # b1 = conv(network, num_filters=96, filter_size=(1, 1))
    # b2 = conv(network, num_filters=48, filter_size=(1, 1))
    # b2 = conv(b2, num_filters=104, filter_size=(3, 3))
    # b3 = conv(network, num_filters=8, filter_size=(1, 1))
    # b3 = conv(b3, num_filters=24, filter_size=(5, 5))
    # network = lasagne.layers.ConcatLayer([b1, b2, b3], axis=1)

    # network = batchnorm.batch_norm(network)
    network = GoodDenseLayer(lasagne.layers.dropout(network, p=.5), num_units=256)
    # network = batchnorm.batch_norm(network)
    network = GoodClsDenseLayer(lasagne.layers.dropout(network, p=.5), num_units=num_classes)
    return network

def build_cnn64_mp(input_var, num_classes, conv, max_pool):
    network24 = lasagne.layers.InputLayer(shape=(None, 3, 24, 24), input_var=input_var[2])
    network24 = lasagne.layers.dropout(network24, p=.1)
    network24 = conv(network24, num_filters=64, filter_size=(3, 3))
    network24 = batchnorm.batch_norm(network24)
    network24 = max_pool(network24)
    network24 = conv(network24, num_filters=64, filter_size=(3, 3))
    network24 = batchnorm.batch_norm(network24)
    network24 = conv(network24, num_filters=64, filter_size=(3, 3))
    network24 = batchnorm.batch_norm(network24)
    network24 = max_pool(network24)
    network24 = GoodDenseLayer(lasagne.layers.dropout(network24, p=.3), num_units=128)
    network24 = batchnorm.batch_norm(network24)

    network48 = lasagne.layers.InputLayer(shape=(None, 3, 48, 48), input_var=input_var[1])
    network48 = lasagne.layers.dropout(network48, p=.1)
    network48 = conv(network48, num_filters=96, filter_size=(3, 3))
    network48 = batchnorm.batch_norm(network48)
    network48 = max_pool(network48)
    network48 = conv(network48, num_filters=96, filter_size=(3, 3))
    network48 = batchnorm.batch_norm(network48)
    network48 = conv(network48, num_filters=96, filter_size=(3, 3))
    network48 = batchnorm.batch_norm(network48)
    network48 = max_pool(network48)
    network48 = conv(network48, num_filters=96, filter_size=(3, 3))
    network48 = batchnorm.batch_norm(network48)
    network48 = conv(network48, num_filters=96, filter_size=(3, 3))
    network48 = batchnorm.batch_norm(network48)
    network48 = max_pool(network48)
    network48 = GoodDenseLayer(lasagne.layers.dropout(network48, p=.3), num_units=192)
    network48 = batchnorm.batch_norm(network48)

    network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64), input_var=input_var[0])
    network = lasagne.layers.dropout(network, p=.1)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = conv(network, num_filters=128, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = GoodDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=256)
    network = batchnorm.batch_norm(network)

    network = lasagne.layers.ConcatLayer([network, network48, network24])
    network = GoodClsDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=num_classes)
    return network

def build_cnn48_mmax(input_var, num_classes, conv, max_pool):
    network12 = lasagne.layers.InputLayer(shape=(None, 3, 12, 12), input_var=input_var[2])
    network12 = lasagne.layers.dropout(network12, p=.1)
    network12 = ConvMMax2DLayer(network12, num_filters=16, filter_size=(3, 3))
    network12 = batchnorm.batch_norm(network12)
    network12 = max_pool(network12)
    network12 = GoodDenseLayer(lasagne.layers.dropout(network12, p=.3), num_units=16)
    network12 = batchnorm.batch_norm(network12)

    network24 = lasagne.layers.InputLayer(shape=(None, 3, 24, 24), input_var=input_var[1])
    network24 = lasagne.layers.dropout(network24, p=.1)
    network24 = ConvMMax2DLayer(network24, num_filters=32, filter_size=(5, 5))
    network24 = batchnorm.batch_norm(network24)
    network24 = max_pool(network24)
    network24 = GoodDenseLayer(lasagne.layers.dropout(network24, p=.3), num_units=64)
    network24 = batchnorm.batch_norm(network24)

    network = lasagne.layers.InputLayer(shape=(None, 3, 48, 48), input_var=input_var[0])
    network = lasagne.layers.dropout(network, p=.1)
    network = ConvMMax2DLayer(network, num_filters=64, filter_size=(5, 5))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = ConvMMax2DLayer(network, num_filters=64, filter_size=(5, 5))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = ConvMMax2DLayer(network, num_filters=64, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = ConvMMax2DLayer(network, num_filters=64, filter_size=(3, 3))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = GoodDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=256)
    network = batchnorm.batch_norm(network)

    network = lasagne.layers.ConcatLayer([network, network24, network12])
    network = GoodClsDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=num_classes)
    return network

def build_cnn48_cal(input_var, num_classes, conv, max_pool):
    network = lasagne.layers.InputLayer(shape=(None, 3, 48, 48), input_var=input_var)
    network = lasagne.layers.dropout(network, p=.1)
    network = conv(network, num_filters=64, filter_size=(5, 5))
    network = batchnorm.batch_norm(network)
    network = max_pool(network)
    network = conv(network, num_filters=64, filter_size=(5, 5))
    network = batchnorm.batch_norm(network)
    network = GoodDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=256)
    network = batchnorm.batch_norm(network)
    network = GoodClsDenseLayer(lasagne.layers.dropout(network, p=.3), num_units=num_classes)
    return network

class NetCreator(object):
    def __init__(self, nets=None):
        if nets is None:
            nets = {}
        self._nets = nets

    def create(self, name, means=None, stds=None):
        return self._nets[name](means, stds)

availdable_nets = {
    "net12": Net12,
    "net12-cal": Net12Cal,
    "net24": Net24,
    "net24-sp": Net24Sp,
    "net24-cal": Net24Cal,
    "net48": Net48,
    "net48-cal": Net48Cal,
    "net-smiles": NetSmiles,
    "net12-mmax": Net12MMax,
    "net48-mmax": Net48MMax,
    "net48-2x": Net482X,
    "net64": Net64,
    "net64-mp": Net64Mp,
    "net64-inc": Net64Inc,
}
