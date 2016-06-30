#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import time
import math
import random
import itertools
import argparse
import requests

import numpy as np
import theano
import theano.tensor as T

import lasagne
from PIL import Image, ImageDraw, ImageFont

from deepevent import *
from osutils import *
from modchooser import *
from nnet import *
from mynets import *
from datasets import *
from boxutils import *
from imgpyramid import *
from imgutils import *


def binary_crossentropy_unsimmetric(p, t, a=2):
    '''
    This is a fun binary crossentropy which gives a times higher loss for false negative than false positive
    '''
    if a == 1:
        return lasagne.objectives.binary_crossentropy(p, t)

    val = float((t == 1 and p > 0.5) or (t == 0 and p < 0.5))
    p1 = val * lasagne.objectives.binary_crossentropy(p, t)
    p2 = (1.0 - val) * (- t * T.log(p) - float(a) * (1.0 - t) * T.log(1.0 - p))
    return p1 + p2

def get_loss(name, num_classes):
    convert_target = lambda x: x
    objective = lasagne.objectives.categorical_crossentropy
    if name == "binary_crossentropy":
        assert num_classes == 2
        objective = lasagne.objectives.binary_crossentropy
        convert_target = lambda x: lasagne.utils.one_hot(x, m=num_classes)
    elif name == "binary_crossentropy_unsimmetric":
        assert num_classes == 2
        objective = binary_crossentropy_unsimmetric
        convert_target = lambda x: lasagne.utils.one_hot(x, m=num_classes)
    return convert_target, objective

def main_train(args):
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--name", required=True, dest="net_name", help="network type")
    parser.add_argument("--data", required=True, dest="dataset", help="dataset to train on")
    parser.add_argument("--epochs", dest="num_epochs", type=int, default=500, help="number of epochs to train")
    parser.add_argument("--batch", dest="batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=0.01, help="learning rate for SGD")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--file", help="network file")
    parser.add_argument("--out", required=True, help="output network file prefix")
    parser.add_argument("--loss", default="categorical_crossentropy", help="loss to optimize")
    parser.add_argument("--deepevent", nargs=2, help="url to deepevent service")
    parser.add_argument("--gpu", action="store_true", default=False, help="enable GPU layers")
    parser.add_argument("--image", nargs="*", help="test image")
    parser.add_argument("--image-size", type=int, dest="image_size", help="maximum of test image width or height")
    parser.add_argument("--image-min-wnd", type=int, dest="image_min_wnd", help="test image minimum window size")
    parser.add_argument("--image-stride", type=int, dest="image_stride", help="test image stride")
    parser.add_argument("--l2-reg", type=float, dest="l2_reg", help="L2 regularizer coefficient")
    parser.add_argument("--l1-reg", type=float, dest="l1_reg", help="L1 regularizer coefficient")
    parser.add_argument("--queue", type=str, default=None, help="enable queue dataset")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false", default=True, help="disable image mirroring")
    parser.add_argument("--only-train-last", dest="only_train_last", action="store_true", default=False, help="only train the last layer")
    args = parser.parse_args(args)

    if args.queue != "all" and args.queue is not None:
        args.queue = int(args.queue)

    mkdir_p(os.path.dirname(args.out))

    network_obj = NetCreator(availdable_nets).create(args.net_name)

    print("Loading data...")
    data = network_obj.dataset_class()(args.dataset, network_obj.sizes())
    if args.queue is not None:
        data = QueueDataSet(data, args.batch_size, ntrain=args.queue, ntest=1, config=Config(mirror=args.mirror))
    assert data.num_classes() == network_obj.num_classes()

    print("Building model and compiling functions...")
    network_obj._means = data._means
    network_obj._stds = data._stds

    labels, images = T.ivector('targets'), network_obj.inputs
    network = network_obj.build(images, args.gpu)
    convert_target, objective = get_loss(args.loss, data.num_classes())

    loss = objective(lasagne.layers.get_output(network), convert_target(labels))
    opt_loss = loss.mean()
    if args.l1_reg is not None:
        opt_loss = opt_loss + lasagne.regularization.regularize_layer_params(network, lasagne.regularization.l1) * args.l1_reg
    if args.l2_reg is not None:
        opt_loss = opt_loss + lasagne.regularization.regularize_layer_params(network, lasagne.regularization.l2) * args.l2_reg

    params = lasagne.layers.get_all_params(network, trainable=True)
    if args.only_train_last:
        params = network.get_params(trainable=True)
    updates = lasagne.updates.nesterov_momentum(opt_loss, params, learning_rate=args.learning_rate, momentum=args.momentum)
    train_fn = theano.function([labels] + images, loss.sum(), updates=updates)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = objective(test_prediction, convert_target(labels)).sum()
    test_acc = T.eq(T.argmax(test_prediction, axis=1), labels).sum()
    val_fn = theano.function([labels] + images, [test_loss, test_acc])

    if args.image:
        classify_fn = theano.function(images, [T.argmax(test_prediction, axis=1), T.max(test_prediction, axis=1)])

    # initialize deepevent events
    if args.deepevent:
        deepevent = DeepEvent(args.deepevent[0], args.deepevent[1])
    else:
        deepevent = DeepEventDummy()

    learning_rate_evt = StepScalarEventCreator("Learning rate")
    epoch_time_evt = StepScalarEventCreator("Epoch time")

    train_loss_evt = StepScalarEventCreator("Losses", "Training loss")
    valid_loss_evt = StepScalarEventCreator("Losses", "Validation loss")
    valid_acc_evt = StepScalarEventCreator("Accuracy", "Validation accuracy")

    train_loss_ema_evt = StepScalarEMAEventCreator("Losses EMA", "Training loss")
    valid_loss_ema_evt = StepScalarEMAEventCreator("Losses EMA", "Validation loss")
    valid_acc_ema_evt = StepScalarEMAEventCreator("Accuracy EMA", "Validation accuracy")

    sizes = network_obj.sizes()
    test_images = None
    if args.image:
        test_images_evt = list(map(lambda name: StepImageEventCreator("Recognition example", name), args.image))
        test_images_time_evt = list(map(lambda name: StepScalarEventCreator("Recognition time, seconds", name), args.image))
        test_images_count_evt = list(map(lambda name: StepScalarEventCreator("Recognition example best windows count", name), args.image))
        test_images = list(map(Image.open, args.image))
        if args.image_size:
            for i in range(len(test_images)):
                img = test_images[i]
                w = img.size[0]
                h = img.size[1]
                coef = float(args.image_size) / float(max(w, h))
                test_images[i] = img.resize((int(math.floor(float(w) * coef)), int(math.floor(float(h) * coef))), Image.ANTIALIAS)

        if args.image_min_wnd:
            min_wnds = [args.image_min_wnd] * len(test_images)
        else:
            min_wnds = list(map(lambda img: round(max(48, min(img.size[0], img.size[1]) / 10)), test_images))

        if args.image_stride:
            strides = [args.image_stride] * len(test_images)
        else:
            strides = list(map(lambda x: int(max(4, x // 5)), min_wnds))


    # Create neural network model (depending on first command line parameter)

    print("Starting training...")
    print("Learning rate:", args.learning_rate)

    epoch = 0
    if args.file:
        epoch = network_obj.load(args.file)

    start_time = time.time()
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in data.iter(args.batch_size, Config(train=False)):
        err, acc = val_fn(*batch)
        val_err += err
        val_acc += acc
        val_batches += len(batch[0])

    best_val_err = val_err
    epoch_time = time.time() - start_time
    print("Epoch {} of {} took {:.3f}s".format(epoch, args.num_epochs, epoch_time))
    print("  validation loss:\t\t{:.6f}".format(float(val_err) / val_batches))
    print("  validation accuracy:\t\t{:.3f} %".format(float(val_acc) / val_batches * 100.0))

    deepevent.send(learning_rate_evt.new(epoch, args.learning_rate))
    deepevent.send(valid_loss_evt.new(epoch, float(val_err) / val_batches))
    deepevent.send(valid_acc_evt.new(epoch, float(val_acc) / val_batches * 100.0))
    if epoch == 0 and test_images:
        for img, evt in zip(test_images, test_images_evt):
            deepevent.send(evt.new(epoch, img))

    # Finally, launch the training loop.
    # We iterate over epochs:
    for epoch in range(epoch + 1, epoch + 1 + args.num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in data.iter(args.batch_size, Config(train=True, mirror=args.mirror)):
            train_err += train_fn(*batch)
            train_batches += len(batch[0])

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in data.iter(args.batch_size, Config(train=False)):
            err, acc = val_fn(*batch)
            val_err += err
            val_acc += acc
            val_batches += len(batch[0])

        train_err /= train_batches

        epoch_time = time.time() - start_time
        print("Epoch {} of {} took {:.3f}s".format(epoch, args.num_epochs, epoch_time))
        print("  training loss:\t\t{:.6f}".format(train_err))
        network_obj.save(epoch, args.out + "-%d" % epoch)
        if val_err < best_val_err:
            best_val_err = val_err
            network_obj.save(epoch, args.out + "-best")

        val_err /= val_batches
        val_acc = val_acc / val_batches * 100.0

        print("  validation loss:\t\t{:.6f}".format(val_err))
        print("  validation accuracy:\t\t{:.3f} %".format(val_acc))

        deepevent.send(epoch_time_evt.new(epoch, epoch_time))

        deepevent.send(train_loss_evt.new(epoch, train_err))
        deepevent.send(valid_loss_evt.new(epoch, val_err))
        deepevent.send(valid_acc_evt.new(epoch, val_acc))

        deepevent.send(train_loss_ema_evt.new(epoch, train_err))
        deepevent.send(valid_loss_ema_evt.new(epoch, val_err))
        deepevent.send(valid_acc_ema_evt.new(epoch, val_acc))

        if test_images:
            for img, min_wnd, stride, evt, evt_time, evt_count in zip(test_images, min_wnds, strides, test_images_evt, test_images_time_evt, test_images_count_evt):
                start_rec_time = time.time()
                img, counts = draw_faces_rects(img, network_obj, classify_fn, sizes, args.batch_size, min_wnd, stride)
                count_all, count_true, count_best, count_best_all = counts
                rec_time = time.time() - start_rec_time
                print("  windows ({}): {} -> {} -> {} -> {};  {:.3f}s".format(evt.name, count_all, count_true, count_best, count_best_all, rec_time))
                deepevent.send(evt_time.new(epoch, rec_time))
                deepevent.send(evt_count.new(epoch, count_best_all))
                deepevent.send(evt.new(epoch, img))

    network_obj.save(args.num_epochs, args.out + "-final")
    data.close()

def get_good_layers(net, gpu):
    good_layers = []
    l1, l2 = get_layers(False)
    match = [
        lasagne.layers.dense.DenseLayer,
        l1, l2,
    ]
    if gpu:
        gl1, gl2 = get_layers(True)
        match.append(gl1)
        match.append(gl2)

    for l in lasagne.layers.get_all_layers(net):
        for m in match:
            if isinstance(l, m):
                good_layers.append(l)
    return good_layers

def load_params_boost(boost, boost_file):
    iteration = 0
    with np.load(boost_file) as f:
        if "iteration" in f.keys():
            iteration = f["iteration"]
        boost.load_model(bytearray(f["model"].tobytes()))
    return iteration

def load_dataset_boost(ds, train_perc=0.75):
    data = np.load(ds)
    labels = data["labels"]
    features = data["features"]
    split = int(len(labels) * train_perc)
    y_val, y_train = labels[:split], labels[split:]
    X_val, X_train = features[:split], features[split:]
    num_classes = len(np.unique(labels))
    return xgb.DMatrix(X_train, label=y_train), xgb.DMatrix(X_val, label=y_val), num_classes

def predict_boost_model(model, data):
    preds = model.predict(data, output_margin=True)
    maxs = np.max(preds, axis=1)
    maxs = np.repeat(maxs[:, np.newaxis], len(preds[0]), axis=1)
    logits = np.exp(preds - maxs)
    lsum = np.sum(logits, axis=1)
    lsums = np.repeat(lsum[:, np.newaxis], len(logits[0]), axis=1) # for broadcasting
    return logits / lsums

def cross_entropy(preds, targets):
    res = 0.0
    for i in range(len(targets)):
        v = max(1e-6, preds[i][targets[i]])
        res -= np.log(v)
    return res

def main_train_boost(args):
    parser = argparse.ArgumentParser(description="Train a tree boost model.")
    parser.add_argument("--net", required=True, nargs=2, help="network name and file")
    parser.add_argument("--batch", dest="batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--gpu", action="store_true", default=False, help="enable GPU layers")
    parser.add_argument("--data", required=True, dest="dataset", help="dataset to train on")
    parser.add_argument("--iters", type=int, default=500, help="number of iterations to train")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=0.01, help="learning rate for SGD")
    parser.add_argument("--file", help="three model file")
    parser.add_argument("--out", required=True, help="output network file prefix")
    parser.add_argument("--deepevent", nargs=2, help="url to deepevent service")
    parser.add_argument("--threads", type=int, default=available_cpu_count(), help="number of threads")
    parser.add_argument("--sample", type=float, nargs=2, default=[1.0, 0.5], help="subsampling settings (samples, features)")
    parser.add_argument("--lambda", dest="plambda", type=float, default=1, help="lambda parameter for booster")
    parser.add_argument("--depth", type=int, default=6, help="tree depth parameter for booster")
    parser.add_argument("--max-bytes", dest="max_bytes", type=int, default=4, help="maximum booster minibatch size in Gb")
    parser.add_argument("--val-perc", dest="val_perc", type=float, help="percentage of the validation set to leave for validation")
    args = parser.parse_args(args)

    args.max_bytes *= 1024 * 1024 * 1024

    import xgboost as xgb

    mkdir_p(os.path.dirname(args.out))

    network_obj = NetCreator(availdable_nets).create(args.net[0])

    print("Loading data...")
    net_data = network_obj.dataset_class()(args.dataset, network_obj.sizes())
    assert net_data.num_classes() == network_obj.num_classes()

    print("Building model and compiling functions...")
    network_obj._means = net_data._means
    network_obj._stds = net_data._stds

    images = network_obj.inputs
    network = network_obj.build(images, args.gpu)
    network_obj.load(args.net[1])
    output = lasagne.layers.get_output(lasagne.layers.concat(
        list(map(
            lambda x: lasagne.layers.flatten(x),
            get_good_layers(network, args.gpu)
        ))
    ), deterministic=True)

    outputs_fn = theano.function(images, output)

    metric = "mlogloss"
    params = {
        "silent": 1,
        "nthread": args.threads,
        "bst:max_depth": args.depth,
        "bst:subsample": args.sample[0],
        "bst:colsample_bytree": args.sample[1],
        "bst:lambda": args.plambda,
        "objective": "multi:softmax",
        "eval_metric": metric,
        "num_class": num_classes,
    }

    if args.deepevent:
        deepevent = DeepEvent(args.deepevent[0], args.deepevent[1])
    else:
        deepevent = DeepEventDummy()

    learning_rate_evt = StepScalarEventCreator("Learning rate")
    epoch_time_evt = StepScalarEventCreator("Epoch time")
    train_loss_evt = StepScalarEventCreator("Losses", "Training loss")
    valid_loss_evt = StepScalarEventCreator("Losses", "Validation loss")
    valid_acc_evt = StepScalarEventCreator("Accuracy", "Validation accuracy")

    print("Starting training...")
    print("Learning rate:", args.learning_rate)

    model = xgb.Booster(params)
    iteration = 0
    if args.file and os.path.exists(args.file):
        iteration = load_params_boost(model, args.file)

    start_time = time.time()
    val_err = 0.0
    val_acc = 0
    features_buf_size = None
    for batch in net_data.iter(args.batch_size, Config(train=False)):
        vals = outputs_fn(*(batch[1:])).astype(theano.config.floatX)
        if features_buf_size is None:
            features_buf_size = len(vals[0]) * 4
        data = xgb.DMatrix(vals, label=batch[0])
        val_predictions = predict_boost_model(model, data)
        val_err += cross_entropy(val_predictions, batch[0])
        val_acc += (np.argmax(val_predictions, axis=1) == batch[0]).sum()

    best_val_err = val_err
    train_size = net_data.size(train=True)
    test_size = net_data.size(train=False)
    iter_time = time.time() - start_time
    print("Iteration {} of {} took {:.3f}s".format(iteration, args.iters, iter_time))
    print("  validation loss:\t\t{:.6f}".format(val_err / test_size))
    print("  validation accuracy:\t\t{:.3f} %".format(val_acc / test_size * 100.0))

    deepevent.send(learning_rate_evt.new(iteration, args.learning_rate))
    deepevent.send(valid_loss_evt.new(iteration, val_err / test_size))
    deepevent.send(valid_acc_evt.new(iteration, val_acc / test_size * 100.0))

    # There is a bug in xgboost: xgb.train segfaults when xgb_model is not a trained model, but just a xgb.Booster(params).
    if iteration == 0:
        model = None

    for iteration in range(iteration + 1, iteration + 1 + args.iters):
        start_time = time.time()
        train_err = 0.0
        big_batch = min(train_size, int(math.floor(args.max_bytes / features_buf_size)))
        for batch in net_data.iter(big_batch, Config(train=True)):
            bsize = len(batch[0])
            vals_buffer = np.zeros((bsize, features_buf_size / 4), dtype=theano.config.floatX)
            num = 0
            for b in iterate_minibatches(batch, args.batch_size, shuffle=False):
                vals_buffer[num:num+len(b[0])] = outputs_fn(*(b[1:])).astype(theano.config.floatX)
                num += len(b[0])

            assert vals.nbytes < args.max_bytes
            data = xgb.DMatrix(vals_buffer, label=batch[0])
            model = xgb.train(
                params=params,
                dtrain=data,
                num_boost_round=1,
                learning_rates=[args.learning_rate],
                verbose_eval=False,
                xgb_model=model,
            )
            train_predictions = predict_boost_model(model, data)
            train_err += cross_entropy(train_predictions, batch[0])

        val_err = 0.0
        val_acc = 0
        for batch in net_data.iter(args.batch_size, Config(train=False)):
            vals = outputs_fn(*(batch[1:])).astype(theano.config.floatX)
            data = xgb.DMatrix(vals, label=batch[0])
            val_predictions = predict_boost_model(model, data)
            val_err += cross_entropy(val_predictions, batch[0])
            val_acc += (np.argmax(val_predictions, axis=1) == batch[0]).sum()

        iter_time = time.time() - start_time
        print("Iteration {} of {} took {:.3f}s".format(iteration, args.iters, iter_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_size))
        print("  validation loss:\t\t{:.6f}".format(val_err / test_size))
        print("  validation accuracy:\t\t{:.3f} %".format(val_acc / test_size * 100.0))

        deepevent.send(train_loss_evt.new(iteration, train_err / train_size))
        deepevent.send(valid_loss_evt.new(iteration, val_err / test_size))
        deepevent.send(valid_acc_evt.new(iteration, val_acc / test_size * 100.0))

        np.savez(args.out + "-%d" % iteration, iteration=iteration, model=np.array(model.save_raw(), dtype=np.uint8))

        if val_err < best_val_err:
            best_val_err = val_err
            np.savez(args.out + "-best", iteration=iteration, model=np.array(model.save_raw(), dtype=np.uint8))

    np.savez(args.out + "-final", iteration=args.iters, model=np.array(model.save_raw(), dtype=np.uint8))
    data.close()

def main_eval(args):
    parser = argparse.ArgumentParser(description="Evaluate the model.")
    parser.add_argument("--data", required=True, dest="dataset", help="dataset to train on")
    parser.add_argument("--batch", dest="batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--net", action="append", nargs=2, required=True, help="network file")
    parser.add_argument("--loss", dest="loss", default="categorical_crossentropy", help="loss to optimize")
    parser.add_argument("--epochs", type=int, default=1, help="epochs to test on")
    parser.add_argument("--gpu", action="store_true", default=False, help="enable GPU layers")
    parser.add_argument("--queue", type=str, default=None, help="enable queue dataset")
    args = parser.parse_args(args)

    if args.queue != "all" and args.queue is not None:
        args.queue = int(args.queue)

    params = [T.ivector('targets')]
    sizes = []
    preds = []
    creator = NetCreator(availdable_nets)
    dcls = None
    for nn in args.net:
        network_obj = creator.create(nn[0])
        if dcls is None:
            dcls = network_obj.dataset_class(all=True)
        else:
            assert network_obj.dataset_class(all=True) == dcls

        for inp in network_obj.inputs:
            params.append(inp)
        for s in network_obj.sizes():
            sizes.append(s)
        network = network_obj.load_build_norm(nn[1], network_obj.inputs, args.gpu)
        preds.append(lasagne.layers.get_output(network, deterministic=True))

    print("Loading data...")
    data = dcls(args.dataset, sizes)
    if args.queue is not None:
        data = QueueDataSet(data, args.batch_size, ntrain=0, ntest=args.queue, config=Config(norm=False, mirror=True, move=True))

    print("Building model and compiling functions...")
    convert_target, objective = get_loss(args.loss, data.num_classes())
    labels = params[0]
    test_prediction = T.mean(preds, axis=0)
    test_loss = objective(test_prediction, convert_target(labels)).sum()
    prediction = T.argmax(test_prediction, axis=1)
    val_fn = theano.function(params, [test_loss, T.eq(prediction, labels).sum(), prediction, labels])

    print("Starting...")

    matrix = np.zeros((data.num_classes(), data.num_classes()))
    start_time = time.time()
    val_err = 0
    val_acc = 0
    val_batches = 0
    for i in range(args.epochs):
        lst_time = time.time()
        step_start_time = time.time()
        for batch in data.iter(args.batch_size, Config(train=False, norm=False, mirror=True, move=True)):
            cur_time = time.time()
            print("Batch generated in {:.3f}s" .format(cur_time - lst_time))
            lst_time = cur_time
            err, acc, preds, labels = val_fn(*batch)
            cur_time = time.time()
            print("Network result calculated in {:.3f}s" .format(cur_time - lst_time))
            lst_time = cur_time
            val_err += err
            val_acc += acc
            val_batches += len(batch[0])
            for p, r in zip(preds, labels):
                matrix[p][r] += 1

            print("Evaluation step finished in {:.3f}s".format(time.time() - step_start_time))
            print("Loss:\t\t{:.6f}".format(float(val_err) / val_batches))
            print("Accuracy:\t{:.4f} %".format(float(val_acc) / val_batches * 100.0))
            assert matrix.sum() == val_batches
            fullprint(matrix / matrix.sum())
            step_start_time = time.time()

    print("Evaluation finished in {:.3f}s".format(time.time() - start_time))
    print("Loss:\t\t{:.6f}".format(float(val_err) / val_batches))
    print("Accuracy:\t{:.4f} %".format(float(val_acc) / val_batches * 100.0))
    assert matrix.sum() == val_batches
    fullprint(matrix / matrix.sum())


def make_fun(net, args):
    output = lasagne.layers.get_output(net, deterministic=True)
    return theano.function(args, [T.argmax(output, axis=1), T.max(output, axis=1)])

def make_fun_cal(net, args):
    return theano.function(args, lasagne.layers.get_output(net, deterministic=True))

def join_all(data):
    res = []
    for k in data.keys():
        for v in data[k]:
            res.append(v)
    return res

def get_best_gen(data, perc):
    data.sort(key=lambda x: x[2][1], reverse=True)
    while len(data) != 0:
        maxi = 0
        maxv = data[maxi][2][1]
        maxbox = data[maxi][1]
        i = 0
        for (_, box, val) in data:
            if val[1] != maxv:
                break
            if box_area(box) < box_area(maxbox):
                maxbox = box
                maxi = i
            i += 1

        resd, box, val = data[maxi]
        data1 = []
        i = 0
        ba = box_area(box)
        for v in data:
            if i == maxi:
                i += 1
                continue
            bc = v[1]
            if box_area(box_intersection(bc, box)) / min(box_area(bc), ba) > perc:
                i += 1
                continue
            data1.append(v)
            i += 1
        data = data1
        yield resd

def get_best_sin_gen(data, perc):
    data.sort(key=lambda x: x[2][1], reverse=True)
    while len(data) != 0:
        resd, box, val = data[0]
        data1 = []
        ba = box_area(box)
        for v in data[1:]:
            bc = v[1]
            if box_area(box_intersection(bc, box)) / min(box_area(bc), ba) > perc:
                continue
            data1.append(v)
        data = data1
        yield resd


def get_bests_gen(data, perc, join=False):
    if join:
        data = join_all(data)

    if isinstance(data, list):
        for d in get_best_gen(data, perc):
            yield d
    else:
        for k in data.keys():
            for d in get_best_sin_gen(data[k], perc):
                yield d

def find_faces(img, net, classifier, sizes, batch_size, min_size=None, stride=None):
    if min_size is None:
        min_size = round(max(48, min(img.size[0], img.size[1]) / 10))
    if stride is None:
        stride = min_size // 5

    datas = {}
    res = {}
    num = 0
    count_all = 0
    count_true = 0

    pyramid = ImagePyramid(img, 48, min_size)
    for v in make_batches_tr(pyramid.arrays(sizes, stride), batch_size):
        coefs, boxs = v[:2]
        arrs = list(v[2:])
        arrs_np = []
        for i in range(len(arrs)):
            arr = np.array(arrs[i], dtype=theano.config.floatX)
            arrs[i] = None
            arr -= net._means[i]
            arr /= net._stds[i]
            arrs_np.append(arr)

        vals = classifier(*tuple(arrs_np))
        del(arrs_np)
        clss = vals[0]
        cnt = len(coefs)
        count_all += cnt
        for i in range(cnt):
            if clss[i] == 1:
                count_true += 1
                box = boxs[i]
                val = (clss[i], vals[1][i])
                res[num] = [box, val, False, False]
                datas.setdefault(coefs[i], []).append(((num, box, val), box, val))
                num += 1

    datas_all = []
    count_best = 0
    for num, box, val in get_bests_gen(datas, 0.6, False):
        res[num][2] = True
        count_best += 1
        datas_all.append((num, box, val))
    del(datas)

    count_best_all = 0
    for num in get_best_gen(datas_all, 0.6):
        count_best_all += 1
        res[num][3] = True
    del(datas_all)

    return list(map(lambda k: tuple(res[k]), sorted(res.keys()))), (count_all, count_true, count_best, count_best_all)

def draw_faces_rects(img, net, classifier, sizes, batch_size, min_size=None, stride=None):
    rects, counts = find_faces(img, net, classifier, sizes, batch_size, min_size, stride)
    copy = img.copy()
    draw = ImageDraw.Draw(copy)
    for (box, _, bests, bests_all) in rects:
        if not bests_all and not bests:
            draw.rectangle(list(box), outline=(0,0,255,255))

    for (box, _, bests, bests_all) in rects:
        if not bests_all and bests:
            draw.rectangle(list(box), outline=(255,0,0,255))

    for (box, _, bests, bests_all) in rects:
        if bests_all and bests:
            draw.rectangle(list(box), outline=(0,255,0,255))

    return copy, counts

def best_box_val(bv):
    box, val = bv
    return val[1]

class WndContext(object):
    def __init__(self, num, coef):
        self.num = num
        self.coef = coef
        self.box = None
        self.wnd = None
        self.val = None
        self.box_old = None
        self.wnd_old = None
        self.val_old = None
        self.arrays = {}

    def get_arr(self, size):
        res = self.arrays.get(size, None)
        if res is not None:
            return res
        res = img_to_arr_bytes(self.wnd.resize((size, size), Image.ANTIALIAS))
        self.arrays[size] = res
        return res

    def set_box(self, box, wnd):
        self.box_old = self.box
        self.wnd_old = self.wnd
        self.val_old = self.val
        self.box = box
        self.wnd = wnd
        self.val = None
        self.arrays = {}

def make_batches_ctx(iterator, batch_size):
    blen = 0
    ctxs = [None] * batch_size
    for ctx in iterator:
        if blen == batch_size:
            yield ctxs
            blen = 0

        ctxs[blen] = ctx
        blen += 1

    if blen != 0:
        yield ctxs[:blen]
        blen = 0

def test_nets_gen(img, batch_size, cls_nets, cal_nets, topn=None, log=True):
    ith = lambda x, i: tuple(map(lambda v: v[i], x))
    cal_classes_np = np.array(CalClasses().classes, dtype=theano.config.floatX)
    min_cal_prob = (1.0 / len(cal_classes_np))

    def ctxs_to_arrs(ctxs, sizes):
        arrs = list(map(lambda _: [], sizes))
        for ctx in ctxs:
            for s in range(len(sizes)):
                arrs[s].append(ctx.get_arr(sizes[s]))

        for i in range(len(arrs)):
            arrs[i] = np.array(arrs[i], dtype=theano.config.floatX)

        return arrs

    def step0(pyramid, stride):
        num = 0
        for coef, box, wnd in pyramid.arrays_gen(stride):
            ctx = WndContext(num, coef)
            num += 1
            ctx.set_box(box, wnd)
            yield ctx

    def step1(iterator, sizes, cls_fun, rects, ref):
        max_size = sizes[0]
        for ctxs in iterator:
            arrs = ctxs_to_arrs(ctxs, sizes)
            vals12 = cls_fun(*arrs)
            del(arrs)
            for i in range(len(ctxs)):
                if vals12[0][i] == 1:
                    ctx = ctxs[i]
                    ctx.val = ith(vals12, i)
                    if log:
                        print("Window {} ({}): {}, {}".format(max_size, ctx.num, ctx.box, ctx.val))
                    rects.append(ctx.box)
                    yield ctx

                if ref is not None:
                    ref[0] += 1

    def step2(iterator, sizes_cal, sizes_cls, cal_fun, cls_fun, rects):
        max_size = sizes_cls[0]
        datas = {}
        for ctxs in iterator:
            arrs = ctxs_to_arrs(ctxs, sizes_cal)
            inds = cal_fun(*arrs) > min_cal_prob
            arrs = list(map(lambda _: [], sizes_cls))
            for i in range(len(ctxs)):
                ctx = ctxs[i]
                ind = inds[i:i+1]
                box = move_box_back(img.size[0], img.size[1], ctx.box, (cal_classes_np * ind.T).sum(axis=0) / ind.sum(axis=1))
                ctx.set_box(box, img.crop(box))
                for s in range(len(sizes_cls)):
                    arrs[s].append(ctx.get_arr(sizes_cls[s]))

            for i in range(len(arrs)):
                arrs[i] = np.array(arrs[i], dtype=theano.config.floatX)

            vals_cal = cls_fun(*arrs)
            del(arrs)
            for i in range(len(ctxs)):
                ctx = ctxs[i]
                ctx.val = ith(vals_cal, i)
                datas.setdefault(ctx.coef, []).append((ctx, ctx.box, ctx.val))
                rects.append(ctx.box)
                if log:
                    print("Window {} cal ({}): {}, {} -> {}, {}".format(max_size, ctx.num, ctx.box_old, ctx.val_old, ctx.box, ctx.val))

        return datas

    def step3(iterator, sizes_cls, rects):
        max_size = sizes_cls[0]
        for ctx in iterator:
            if log:
                print("Window {} cal best ({}): {}, {}, {} ".format(max_size, ctx.num, ctx.box, ctx.val_old, ctx.val))
            rects.append(ctx.box)
            yield ctx

    rects = list(map(lambda _: [], range(len(cls_nets) * 3 - 1 + 2)))
    rects_num = 0

    datas = None
    cnt_wnds = None

    max_size = cal_nets[-1][0].sizes()[0]

    min_size = round(max(max_size, min(img.size[0], img.size[1]) / 3))
    stride = min_size // 5
    pyramid = ImagePyramid(img, max_size, min_size)
    iter0 = make_batches_ctx(step0(pyramid, stride), batch_size)
    for inum in range(len(cls_nets)):
        cls_fun = cls_nets[inum][1]
        cal_fun = cal_nets[inum][1]
        cls_sizes = cls_nets[inum][0].sizes()
        cal_sizes = cal_nets[inum][0].sizes()

        counts_ref = [0]
        iter1 = make_batches_ctx(step1(iter0, cls_sizes, cls_fun, rects[rects_num], counts_ref), batch_size)
        datas = step2(iter1, cal_sizes, cls_sizes, cal_fun, cls_fun, rects[rects_num + 1])
        if inum == 0:
            cnt_wnds = counts_ref[0]

        rects_num += 2

        if inum != len(cls_nets) - 1:
            iter0 = make_batches_ctx(step3(get_bests_gen(datas, 0.6, False), cls_sizes, rects[rects_num]), batch_size)
            rects_num += 1

    datas_all = {}
    for ctx in get_bests_gen(datas, 0.6, False):
        if log:
            print("Window {} cal best ({}): {}, {}, {} ".format(max_size, ctx.num, ctx.box, ctx.val_old, ctx.val))
        rects[rects_num].append(ctx.box)
        # NOTE: we used ctx.val_old before
        datas_all.setdefault(ctx.coef, []).append((ctx, ctx.box, ctx.val))

    del(datas)
    rects_num += 1

    rs = []
    for ctx in get_bests_gen(datas_all, 0.6, True):
        if log:
            print("Window {} cal best all ({}): {}, {}, {} ".format(max_size, ctx.num, ctx.box, ctx.val_old, ctx.val))
        rs.append((ctx.box, ctx.val))

    del(datas_all)

    rs.sort(key=best_box_val, reverse=True)
    if topn is None:
        for r, _ in rs:
            rects[rects_num].append(r)
    else:
        for r, _ in rs[:topn]:
            rects[rects_num].append(r)

    return cnt_wnds, rects


def main_test(args):
    parser = argparse.ArgumentParser(description="Evaluate the model.")
    parser.add_argument("--net", nargs=2, required=True, action="append", help="classification network")
    parser.add_argument("--net-cal", dest="net_cal", nargs=2, required=True, action="append", help="calibration network")
    parser.add_argument("--net-smiles", dest="net_smiles", nargs=2, help="smiles network")
    parser.add_argument("--image", required=True, help="image to test on")
    parser.add_argument("--gpu", action="store_true", default=False, help="enable GPU layers")
    parser.add_argument("--file", help="marks.txt file")
    parser.add_argument("--dir", help="images dir")
    parser.add_argument("--smiles-dir", dest="smiles_dir", help="smiles data dir")
    parser.add_argument("--batch", dest="batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--out", help="output image")
    parser.add_argument("--show", action="store_true", default=False, help="show result")
    parser.add_argument("--font", default=None, help="font file")
    args = parser.parse_args(args)

    all_smiles = {}
    if args.smiles_dir:
        for subdir, dirs, files in os.walk(args.smiles_dir):
            for f in files:
                if f != "0_smile.png":
                    continue
                path = os.path.join(subdir, f)
                label = int(os.path.split(subdir)[-1]) - 1
                assert label >= 0
                all_smiles[label] = Image.open(path)


    print("Building model and compiling functions...")

    creator = NetCreator(availdable_nets)

    net_smiles_obj = None
    classify_smiles = None
    if args.net_smiles is not None:
        net_smiles_obj = creator.create(args.net_smiles[0])
        net_smiles = net_smiles_obj.load_build_norm(args.net_smiles[1], net_smiles_obj.inputs, args.gpu)

        smile_probs_sum = T.sum(lasagne.layers.get_output(net_smiles, deterministic=True), axis=0)
        classify_smiles = theano.function(net_smiles_obj.inputs, [T.argmax(smile_probs_sum), T.max(smile_probs_sum)])

    nets = []
    for v in args.net:
        net_obj = creator.create(v[0])
        net = net_obj.load_build_norm(v[1], net_obj.inputs, args.gpu)
        classify = make_fun(net, net_obj.inputs)
        nets.append((net_obj, classify))

    nets_cal = []
    for v in args.net_cal:
        net_obj = creator.create(v[0])
        net = net_obj.load_build_norm(v[1], net_obj.inputs, args.gpu)
        classify = make_fun_cal(net, net_obj.inputs)
        nets_cal.append((net_obj, classify))

    print("Starting...")

    cal_classes = CalClasses().classes

    boxes = []
    img = None
    imgname = args.image.replace("/", "_")
    for name, im, bx in iterate_boxes(args.file, args.dir):
        name = name.replace("/", "_")
        if name.find(imgname) != -1 or imgname.find(name) != -1:
            img = im
            boxes = list(map(lambda x: bbox_bound(x, im.size[0], im.size[1]), bx))
            break

    if img is None:
        img = Image.open(args.image)

    print("Image size: (%d, %d)" % (img.size[0], img.size[1]))

    start_time = time.time()
    cnt_wnds, rects = test_nets_gen(img, args.batch_size, nets, nets_cal)
    if args.net_smiles is not None:
        rects.append([rects[-1][0]])

    print("Finished in {:.3f}s".format(time.time() - start_time))
    print("Windows:", cnt_wnds)
    print("Windows 12:", len(rects[1]))
    print("Windows 12 best:", len(rects[2]))
    print("Windows 24:", len(rects[4]))
    print("Windows 24 best:", len(rects[5]))
    print("Windows 48:", len(rects[7]))
    print("Windows 48 best:", len(rects[8]))
    print("Windows 48 best all:", len(rects[9]))

    copies = list(map(lambda _: img.copy(), rects))
    draws = list(map(ImageDraw.Draw, copies))

    for b in boxes:
        for d in draws:
            d.rectangle(list(b), outline=(255,255,255,255))
        # wnd = img.crop(b)
        # sizes = {}
        # for n in nets:
        #     for s in n[0].sizes():
        #         sizes[s] = True

        # arrs = {}
        # for s in sizes.keys():
        #     arrs[s] = img_to_arr(wnd.resize((s, s), Image.ANTIALIAS))
        # del(sizes)

        # prt = str(b)
        # for net in nets:
        #     prt += " " + str(net[1](*tuple(map(lambda s: arrs[s], net[0].sizes()))))
        # print(prt)

    colors = [(255,0,0,255), (0,255,0,255), (0,0,255,255), (255,255,0,255), (255,255,0,255), (255,0,255,255), (255,255,0,255), (255,255,0,255), (0,0,255,255), (255,255,0,255), (0,255,0,255)]

    for rs, col, draw in zip(rects, colors, draws):
        for r in rs:
            draw.rectangle(list(r), outline=col)

    if args.net_smiles is not None:
        best_wnd = rects[10][0]
        arrs = list(map(lambda _: [], net_smiles_obj.sizes()))
        for c in cal_classes:
            mr = move_box(img.size[0], img.size[1], best_wnd, c)
            if mr is None:
                continue
            minx, miny, maxx, maxy = mr
            face = img.crop(mr)

            for i, s in enumerate(net_smiles_obj.sizes()):
                arrs[i].append(img_to_arr_bytes(face.resize((s, s), Image.ANTIALIAS)))

        for i in range(len(arrs)):
            arrs[i] = np.array(arrs[i], dtype=theano.config.floatX)

        smile_cls, smile_val = classify_smiles(*arrs)
        smile_cls = int(smile_cls)
        smile_val = float(smile_val)

        minx, miny, maxx, maxy = best_wnd
        new_box = (maxx - round((maxx - minx) / 5), maxy - round((maxy - miny) / 5), maxx, maxy)
        im = all_smiles[smile_cls].resize((new_box[2] - new_box[0], new_box[3] - new_box[1]), Image.ANTIALIAS)
        copies[-1].paste(im, box=new_box)

        if args.font:
            draws[-1].text(
                (minx, maxy - round((maxy - miny) / 5 / 1.5)),
                "{:.3f}".format(smile_val / len(arrs[0])),
                font=ImageFont.truetype(args.font, round((maxy - miny) / 5 / 1.5)),
                fill=(255,255,255,128),
            )

    bighn = 3
    bigwn = 4
    out_img = Image.new("RGBA", (img.size[0] * bigwn, img.size[1] * bighn))
    for y in range(0, bighn):
        for x in range(0, bigwn):
            idx = bigwn * y + x
            if idx >= len(copies):
                break
            out_img.paste(copies[idx], box=(img.size[0] * x, img.size[1] * y))

    if args.show:
        out_img.show()

    if args.out is not None:
        out_img.save(args.out)

def get_empty_boxes(imw, imh, taken):
    taken_start_set = {}
    xbounds = [0, imw]
    ybounds = [0, imh]
    for b in taken:
        taken_start_set[(b[0], b[1])] = True
        xbounds.append(b[0])
        xbounds.append(b[2])
        ybounds.append(b[1])
        ybounds.append(b[3])
    xbounds = list(set(xbounds))
    ybounds = list(set(ybounds))
    xbounds.sort()
    ybounds.sort()

    boxes = []
    for ix in range(len(xbounds) - 1):
        x = xbounds[ix]
        nx = xbounds[ix + 1]
        for iy in range(len(ybounds) - 1):
            y = ybounds[iy]
            if taken_start_set.get((x, y), None) is None:
                ny = ybounds[iy + 1]
                box = (x, y, nx, ny)
                if not box_inside_any(box, taken):
                    boxes.append(box)

    res = []
    boxes.sort(key=box_area, reverse=True)
    while len(boxes) > 0:
        best = boxes[0]
        any_joined = False
        for i in range(len(boxes[1:])):
            b = boxes[1 + i]
            if is_neighbours(b, best):
                boxes = [box_join(best, b)] + boxes[1:i + 1] + boxes[i + 1 + 1:]
                any_joined = True
                break

        if not any_joined:
            res.append(best)
            boxes = boxes[1:]

    return res

def main_prep_data(args):
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument("--smiles-dir", dest="smiles_dir", help="images dir")
    parser.add_argument("--dir", nargs=2, help="images dir")
    parser.add_argument("--dir-lfw", dest="dir_lfw", help="images dir from lfw")
    parser.add_argument("--dir-celeba", dest="dir_celeba", help="images dir from CelebA")
    parser.add_argument("--out", required=True, help="output dataset")
    parser.add_argument("--min-box", type=int, default=12, dest="min_box", help="minimum boxsize for non faces")

    args = parser.parse_args(args)

    mkdir_p(os.path.dirname(args.out))

    cal_classes = CalClasses()

    boxes_true = []
    boxes_false = []
    images = []
    names = []

    num = 0
    if args.dir is not None:
        for name, im, box in iterate_boxes(args.dir[0], args.dir[1], nprint=True):
            imw, imh = im.size[0], im.size[1]
            for b in map(lambda x: bbox_bound(x, imw, imh), box):
                boxes_true.append([1, num] + list(b))

            imw, imh = im.size[0], im.size[1]
            big_bound_box = list(map(lambda x: bbox_bound(cal_classes.bbox(x), imw, imh), box))
            for b in get_empty_boxes(imw, imh, big_bound_box):
                if min(b[2] - b[0], b[3] - b[1]) >= args.min_box:
                    boxes_false.append([0, num] + list(b))

            names.append(name)
            images.append(img_to_arr_bytes(im))
            num += 1

    if args.smiles_dir is not None:
        for subdir, _, files in os.walk(args.smiles_dir):
            for f in files:
                if f == "0_smile.png":
                    continue
                path = os.path.join(subdir, f)
                print(path)
                names.append(path)
                im = Image.open(path)
                boxes_true.append([1, num] + list(cal_classes.small((0, 0, im.size[0], im.size[0]))))
                images.append(img_to_arr_bytes(im))
                num += 1

    if args.dir_lfw is not None:
        for subdir, _, files in os.walk(args.dir_lfw):
            for f in files:
                path = os.path.join(subdir, f)
                print(path)
                names.append(path)
                im = Image.open(path)
                boxes_true.append([1, num] + list(cal_classes.small((0, 0, im.size[0], im.size[0]))))
                images.append(img_to_arr_bytes(im))
                num += 1

    if args.dir_celeba is not None:
        for subdir, _, files in os.walk(args.dir_celeba):
            for f in files:
                path = os.path.join(subdir, f)
                print(path)
                names.append(path)
                im = Image.open(path)
                boxes_true.append([1, num] + list(cal_classes.small((0, 0, im.size[0], im.size[0]))))
                images.append(img_to_arr_bytes(im))
                num += 1

    names = np.array(names)
    boxes_true = np.array(boxes_true, dtype=np.int32)
    boxes_false = np.array(boxes_false, dtype=np.int32)
    shapes = np.array(list(map(lambda x: [product(x.shape)] + list(x.shape), images)), dtype=np.int64)
    acc = 0
    for i in range(len(shapes)):
        l = shapes[i][0]
        shapes[i][0] = acc
        acc += l

    for i in range(len(images)):
        images[i] = images[i].flatten()
    images = np.concatenate(images)

    res = {"boxes_true": boxes_true, "boxes_false": boxes_false, "shapes": shapes, "images": images, "names": names}
    with open(args.out, "wb") as fout:
        np.savez(fout, **res)

def main_prep_data_smiles(args):
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument("--dir", required=True, help="images dir")
    parser.add_argument("--out", required=True, help="output dataset")

    args = parser.parse_args(args)

    mkdir_p(os.path.dirname(args.out))

    boxes_true = []
    images = []
    names = []
    num = 0

    cal_classes = CalClasses()

    for subdir, dirs, files in os.walk(args.dir):
        for f in files:
            if f == "0_smile.png":
                continue
            path = os.path.join(subdir, f)
            label = int(os.path.split(subdir)[-1]) - 1
            assert label >= 0
            names.append(path)
            im = Image.open(path)
            boxes_true.append([label, num] + list(cal_classes.small((0, 0, im.size[0], im.size[0]))))
            images.append(img_to_arr_bytes(im))
            num += 1
            print(path)

    names = np.array(names)
    boxes_true = np.array(boxes_true, dtype=np.int32)
    shapes = np.array(list(map(lambda x: [product(x.shape)] + list(x.shape), images)), dtype=np.int64)
    acc = 0
    for i in range(len(shapes)):
        l = shapes[i][0]
        shapes[i][0] = acc
        acc += l

    for i in range(len(images)):
        images[i] = images[i].flatten()
    images = np.concatenate(images)

    res = {"boxes_true": boxes_true, "shapes": shapes, "images": images, "names": names}
    with open(args.out, "wb") as fout:
        np.savez(fout, **res)

def iterate_boxes(fname, dirname, nprint=False):
    with open(fname, "r") as fin:
        while True:
            l = fin.readline()
            if l == "":
                break
            name = l[:-1]
            fname = os.path.join(dirname, name + ".jpg")
            if nprint:
                print(name)
            im = Image.open(fname)
            if (im.getbands() != ("R", "G", "B")):
                im = im.convert("RGB")
            cnt = int(fin.readline()[:-1])
            bx = []
            for _ in range(cnt):
                ellipse = tuple(map(float, fin.readline()[:-1].split(" ")[:5]))
                a, b, angle, cx, cy = ellipse
                bx.append(rectify_bbox(ellipse_to_bbox(a, b, angle, cx, cy)))
            yield name, im, bx

def iterate_images(fname, dirname):
    cal_classes = CalClasses()
    num = 0
    for name, im, bx in iterate_boxes(fname, dirname):
        imw, imh = im.size[0], im.size[1]
        btrue = list(map(lambda x: bbox_bound(cal_classes.bbox(x, imw, imh)), bx))
        yield (num, im, btrue, get_empty_boxes(imw, imh, btrue))
        num += 1

def iterate_images_data(fname):
    data = np.load(fname)
    boxes_true = data["boxes_true"].astype(np.int32)
    boxes_false = data["boxes_false"].astype(np.int32)
    shapes = data["shapes"].astype(np.int64)
    images = data["images"]
    del(data)

    poss = shapes[:, 0]
    shapes = shapes[:, 1:]
    lens = shapes.prod(axis=1)

    for i in range(len(poss)):
        btrue = boxes_true[boxes_true[:, 1] == i][:, 2:].tolist()
        bfalse = boxes_false[boxes_false[:, 1] == i][:, 2:].tolist()
        img = images[poss[i] : poss[i] + lens[i]].reshape(shapes[i])
        img = arr_to_img(img)
        yield (i, img, btrue, bfalse)

def draw_rects(im, btrue, bfalse, blend=0.2):
    copy = Image.new(im.mode, im.size)
    draw = ImageDraw.Draw(copy)
    for b in map(list, btrue):
        b[2] -= 1
        b[3] -= 1
        draw.rectangle(b, fill=(0,255,0,128))
    for b in map(list, bfalse):
        b[2] -= 1
        b[3] -= 1
        draw.rectangle(b, fill=(255,0,0,128))

    copy = Image.blend(im, copy, blend)
    draw = ImageDraw.Draw(copy)
    for b in map(list, btrue):
        b[2] -= 1
        b[3] -= 1
        draw.rectangle(b, outline=(0,255,0,255))
    for b in map(list, bfalse):
        b[2] -= 1
        b[3] -= 1
        draw.rectangle(b, outline=(255,0,0,255))
    return copy

def main_mon_test(args):
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--dir", required=True, help="images dir")
    parser.add_argument("--file", required=True, help="marks file")
    parser.add_argument("--data", required=True, help="data file")
    args = parser.parse_args(args)

    import cv2

    images1 = iterate_images(args.file, args.dir)
    images2 = iterate_images_data(args.data)

    cv2.namedWindow("frame1")
    cv2.namedWindow("frame2")

    num, im, btrue, bfalse = next(images1)
    cv2.imshow("frame1", im_to_ocv(draw_rects(im.convert("RGBA"), btrue, bfalse)))
    num, im, btrue, bfalse = next(images2)
    cv2.imshow("frame2", im_to_ocv(draw_rects(im.convert("RGBA"), btrue, bfalse)))

    for _ in iterate_keys("c", "q"):
        num, im, btrue, bfalse = next(images1)
        cv2.imshow("frame1", im_to_ocv(draw_rects(im.convert("RGBA"), btrue, bfalse)))
        num, im, btrue, bfalse = next(images2)
        cv2.imshow("frame2", im_to_ocv(draw_rects(im.convert("RGBA"), btrue, bfalse)))

    cv2.destroyAllWindows()

def main_iter_dataset(args):
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--data", required=True, help="data file")
    args = parser.parse_args(args)

    import cv2

    data = AllBestClsDataSet(args.data, [256])
    images = data.iter(2, Config(train=False, shuffle=True, mirror=True, norm=False, move=True))

    cv2.namedWindow("frame1")
    cv2.namedWindow("frame2")

    labels, imgs = next(images)
    imgs = list(map(lambda x: arr_to_img(x).convert("RGBA"), imgs.astype(np.uint8)))
    if labels[0] == 0:
        imgs = [imgs[1], imgs[0]]
    cv2.imshow("frame1", im_to_ocv(imgs[0]))
    cv2.imshow("frame2", im_to_ocv(imgs[1]))
    for _ in iterate_keys("c", "q"):
        labels, imgs = next(images)
        imgs = list(map(lambda x: arr_to_img(x).convert("RGBA"), imgs.astype(np.uint8)))
        if labels[0] == 0:
            imgs = [imgs[1], imgs[0]]
        cv2.imshow("frame1", im_to_ocv(imgs[0]))
        cv2.imshow("frame2", im_to_ocv(imgs[1]))

    cv2.destroyAllWindows()


def iterate_frames(video):
    import cv2

    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        yield frame
    cap.release()

def best_rect(rects):
    print(rects)
    i = 0
    br = rects[0]
    for r in rects[1:]:
        if r[1][0] > br[1][0]:
            br = r
    return br

def main_mark_video(args):
    parser = argparse.ArgumentParser(description="Evaluate models on a video.")
    parser.add_argument("--net", nargs=2, required=True, action="append", help="classification network")
    parser.add_argument("--net-cal", dest="net_cal", nargs=2, required=True, action="append", help="calibration network")
    parser.add_argument("--net-smiles", required=True, nargs=2, dest="net48_smiles", help="network name and file")
    parser.add_argument("--video", required=True, help="video file")
    parser.add_argument("--video-dst", dest="video_dst", help="output video file")
    parser.add_argument("--smiles-dir", required=True, dest="smiles_dir", help="smiles data dir")
    parser.add_argument("--gpu", action="store_true", default=False, help="enable GPU layers")
    parser.add_argument("--batch", dest="batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--top1", dest="top_one", action="store_true", default=False, help="only leave top 1 square")
    parser.add_argument("--show", action="store_true", default=False, help="show result on screen")
    parser.add_argument("--font", default=None, help="font file")
    args = parser.parse_args(args)

    import cv2

    out_video = None
    if args.video_dst:
        v = cv2.VideoCapture(args.video)
        import struct
        buf = struct.pack('f', v.get(6))
        # fcc = cv2.VideoWriter_fourcc(chr(buf[0]), chr(buf[1]), chr(buf[2]), chr(buf[3]))
        fcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out_video = cv2.VideoWriter(args.video_dst, fcc, 5, (int(v.get(3)), int(v.get(4))))
        v.release()
        v = None

    print("Loading data...")

    all_smiles = {}
    for subdir, dirs, files in os.walk(args.smiles_dir):
        for f in files:
            if f != "0_smile.png":
                continue
            path = os.path.join(subdir, f)
            label = int(os.path.split(subdir)[-1]) - 1
            assert label >= 0
            all_smiles[label] = Image.open(path)

    print("Building model and compiling functions...")

    creator = NetCreator(availdable_nets)

    nets = []
    for v in args.net:
        net_obj = creator.create(v[0])
        net = net_obj.load_build_norm(v[1], net_obj.inputs, args.gpu)
        classify = make_fun(net, net_obj.inputs)
        nets.append((net_obj, classify))

    nets_cal = []
    for v in args.net_cal:
        net_obj = creator.create(v[0])
        net = net_obj.load_build_norm(v[1], net_obj.inputs, args.gpu)
        classify = make_fun_cal(net, net_obj.inputs)
        nets_cal.append((net_obj, classify))

    net48_smiles_obj = creator.create(args.net48_smiles[0])
    net48_smiles = net48_smiles_obj.load_build_norm(args.net48_smiles[1], net48_smiles_obj.inputs, args.gpu)

    smile_probs_sum = T.sum(lasagne.layers.get_output(net48_smiles, deterministic=True), axis=0)
    classify_smiles = theano.function(net48_smiles_obj.inputs, [T.argmax(smile_probs_sum), T.max(smile_probs_sum)])

    print("Starting...")

    cal_classes = CalClasses()

    topn = None
    if args.top_one:
        topn = 1

    if args.show:
        cv2.namedWindow("frame")


    num = 0
    for frame in iterate_frames(args.video):
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        _, rects = test_nets_gen(im, args.batch_size, nets, nets_cal, topn=topn, log=False)
        draw = ImageDraw.Draw(im)
        for r in rects[-1]:
            arrs = list(map(lambda _: [], net48_smiles_obj.sizes()))
            for c in cal_classes.classes:
                mr = move_box(im.size[0], im.size[1], r, c)
                if mr is None:
                    continue
                minx, miny, maxx, maxy = mr
                face = im.crop(mr)

                for i, s in enumerate(net48_smiles_obj.sizes()):
                    arrs[i].append(img_to_arr_bytes(face.resize((s, s), Image.ANTIALIAS)))

            for i in range(len(arrs)):
                arrs[i] = np.array(arrs[i], dtype=theano.config.floatX)

            smile_cls, smile_val = classify_smiles(*arrs)
            smile_cls = int(smile_cls)
            smile_val = float(smile_val)

            minx, miny, maxx, maxy = r
            new_box = (maxx - round((maxx - minx) / 5), maxy - round((maxy - miny) / 5), maxx, maxy)
            img = all_smiles[smile_cls].resize((new_box[2] - new_box[0], new_box[3] - new_box[1]), Image.ANTIALIAS)
            im.paste(img, box=new_box)

            if args.font:
                draw.text(
                    (minx, maxy - round((maxy - miny) / 5 / 1.5)),
                    "{:.3f}".format(smile_val / len(arrs[0])),
                    font=ImageFont.truetype(args.font, round((maxy - miny) / 5 / 1.5)),
                    fill=(255,255,255,128),
                )
            draw.rectangle(list(r), outline=(0, 255, 0))

        frame_out = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

        if args.show:
            cv2.imshow("frame", frame_out)
            for _ in range(10):
                cv2.waitKey(1)

        if out_video:
            out_video.write(frame_out)

        print("Frame", num)

        num += 1

    if out_video:
        out_video.release()

    cv2.destroyAllWindows()


def main_conv_net(args):
    parser = argparse.ArgumentParser(description="Evaluate models on a video.")
    parser.add_argument("--net", nargs=2, required=True, help="source net")
    parser.add_argument("--out-net", required=True, dest="out_net", help="resulting net file")
    parser.add_argument("--num", required=True, type=int, help="new number of classes")
    parser.add_argument("--gpu", action="store_true", default=False, help="enable GPU layers")
    args = parser.parse_args(args)

    network_obj = NetCreator(availdable_nets).create(args.net[0])
    network = network_obj.load_build_norm(args.net[1], network_obj.inputs, args.gpu)
    all_layers = lasagne.layers.get_all_layers(network)
    network_obj.net = GoodClsDenseLayer(all_layers[-2], num_units=args.num)
    network_obj.save(0, args.out_net)


if __name__ == '__main__':
    (ModChooser("Deep learning tools.")
        .add("prep_data", main_prep_data, "prepare dataset for a network")
        .add("prep_data_smiles", main_prep_data_smiles, "prepare dataset for a smiles network")
        .add("train", main_train, "train network")
        .add("eval", main_eval, "eval network")
        .add("test", main_test, "test network on image")
        .add("train_boost", main_train_boost, "train xgboost")
        .add("mon_test", main_mon_test, "iterate through images in directory and a prepared dataset")
        .add("iter_dataset", main_iter_dataset, "iterate through a prepared dataset")
        .add("mark_video", main_mark_video, "mark video")
        .add("conv_net", main_conv_net, "change last dence layer to have different number of classes")
        .main())
