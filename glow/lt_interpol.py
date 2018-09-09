#!/usr/bin/env python

# Modified Horovod MNIST example

import os
import sys
import time
from data_loaders.read_tf_records2 import build_tfrecord_single
from data_loaders.read_tf_records2 import CONF as CARTGRIPPER_CONF
import matplotlib.pyplot as plt
from glow.graphics import save_interpolations

import glow.model as model

from glow.train import tensorflow_session, get_data, get_its

import numpy as np
import tensorflow as tf

import pdb

from data_loaders.get_data import make_batch

learn = tf.contrib.learn

# Surpress verbose warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def _print(*args, **kwargs):
    print(*args, **kwargs)


def process_results(results):
    stats = ['loss', 'bits_x', 'bits_y', 'pred_loss']
    assert len(stats) == results.shape[0]
    res_dict = {}
    for i in range(len(stats)):
        res_dict[stats[i]] = "{:.4f}".format(results[i])
    return res_dict

def interpolate(test_iterator, model, steps = 10):

    if hps.direct_iterator:
        x, y_ = test_iterator.get_next()
        imshape = x.get_shape()
    else:
        x, _ = test_iterator()
        imshape = x.shape
    x1_pl = tf.placeholder(dtype=x.dtype, shape=imshape)
    x2_pl = tf.placeholder(dtype=x.dtype, shape=imshape)

    lt1 = model.build_exmp_encoder(x1_pl, reuse=True)
    lt2 = model.build_exmp_encoder(x2_pl, reuse=True)

    factor = tf.placeholder(tf.float32, shape=())

    int_lt = []
    for lt1_, lt2_ in zip(lt1, lt2):
        int_lt.append(lt1_*factor + lt2_*(1-factor))

    x_dec = model.build_exmp_decoder(int_lt)

    if hps.direct_iterator:
        im1, _y = model.sess.run([x, y_])
        im2, _ = model.sess.run([x, y_])
    else:
        im1, _y = test_iterator()
        im2, _ = test_iterator()


    rows = 10 if hps.image_size <= 64 else 4
    cols = rows
    _y = np.asarray([_y % hps.n_y for _y in (
            list(range(cols)) * rows)], dtype='int32')

    images = []
    int_factors = np.linspace(0, 1., steps)
    for s in range(steps):
        feed_dict = {
            x1_pl: im1,
            x2_pl: im2,
            factor:int_factors[s],
            model.Y:_y
        }
        images.append(model.sess.run(x_dec, feed_dict))
    return images

def lt_interpol(hps):


    # Create tensorflow session
    sess = tensorflow_session()

    # Download and load dataset.
    tf.set_random_seed(hps.seed)
    np.random.seed(hps.seed)

    # Get data and set train_its and valid_its
    train_iterator, test_iterator, data_init = get_data(hps, sess)
    hps.train_its, hps.test_its, hps.full_test_its = get_its(hps)

    ### debug:
    # sess.run(train_iterator.get_next())

    # Create log dir
    logdir = os.path.abspath(hps.logdir) + "/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # Create model
    model = model.model(sess, hps, train_iterator, test_iterator, data_init, train=False)

    images = interpolate(test_iterator, model, steps=20)
    sess.graph.finalize()

    save_interpolations(images, 'logs/interpolations')


if __name__ == "__main__":

    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--restore_path", type=str, default='',
                        help="Location of checkpoint to restore")
    parser.add_argument("--logdir", type=str,
                        default='./logs', help="Location to save logs")

    # Dataset hyperparams:
    parser.add_argument("--problem", type=str, default='cifar10',
                        help="Problem (mnist/cifar10/imagenet")
    parser.add_argument("--category", type=str,
                        default='', help="LSUN category")
    parser.add_argument("--data_dir", type=str, default='',
                        help="Location of data")
    parser.add_argument("--dal", type=int, default=1,
                        help="Data augmentation level: 0=None, 1=Standard, 2=Extra")

    # New dataloader params
    parser.add_argument("--fmap", type=int, default=1,
                        help="# Threads for parallel file reading")
    parser.add_argument("--pmap", type=int, default=16,
                        help="# Threads for parallel map")

    # Optimization hyperparams:
    parser.add_argument("--n_train", type=int,
                        default=50000, help="Train epoch size")
    parser.add_argument("--n_test", type=int, default=-
                        1, help="Valid epoch size")
    parser.add_argument("--n_batch_train", type=int,
                        default=64, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=50, help="Minibatch size")
    parser.add_argument("--n_batch_init", type=int, default=256,
                        help="Minibatch size for data-dependent init")
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--polyak_epochs", type=float, default=1,
                        help="Nr of averaging epochs for Polyak and beta2")
    parser.add_argument("--weight_decay", type=float, default=1.,
                        help="Weight decay. Switched off by default.")
    parser.add_argument("--epochs", type=int, default=1000000,
                        help="Total number of training epochs")
    parser.add_argument("--epochs_warmup", type=int,
                        default=10, help="Warmup epochs")
    parser.add_argument("--epochs_full_valid", type=int,
                        default=50, help="Epochs between valid")
    parser.add_argument("--gradient_checkpointing", type=int,
                        default=1, help="Use memory saving gradients")

    # Model hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=-1, help="Image size")
    parser.add_argument("--anchor_size", type=int, default=32,
                        help="Anchor size for deciding batch size")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=32,
                        help="Depth of network")
    parser.add_argument("--weight_y", type=float, default=0.00,
                        help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=8,
                        help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=3,
                        help="Number of levels")

    # Synthesis/Sampling hyperparameters:
    parser.add_argument("--n_sample", type=int, default=1,
                        help="minibatch size for sample")
    parser.add_argument("--epochs_full_sample", type=int,
                        default=50, help="Epochs between full scale sample")

    # Ablation
    parser.add_argument("--learntop", action="store_true",
                        help="Learn spatial prior")
    parser.add_argument("--ycond", action="store_true",
                        help="Use y conditioning")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--flow_permutation", type=int, default=2,
                        help="Type of flow. 0=reverse (realnvp), 1=shuffle, 2=invconv (ours)")
    parser.add_argument("--flow_coupling", type=int, default=0,
                        help="Coupling type: 0=additive, 1=affine")

    hps = parser.parse_args()  # So error if typo
    lt_interpol(hps)