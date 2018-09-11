#!/usr/bin/env python

# Modified Horovod MNIST example

import os
import sys
import time
from glow.data_loaders.read_tf_records2 import build_tfrecord_single
from glow.data_loaders.read_tf_records2 import CONF
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug


import numpy as np
import tensorflow as tf

import pdb

from glow.data_loaders.get_data import make_batch

learn = tf.contrib.learn

# Surpress verbose warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

summ_interval = 500 #episodes


def _print(*args, **kwargs):
    print(*args, **kwargs)

def sample_images(eps, sess, model, summary_writer, itr, bsize, sum_op, mode):

    y = np.zeros(bsize, dtype='int32')
    feed_dict = {
                 model.eps_std: np.ones(bsize)*eps,
                 model.train_cond: mode}

    [summary_str] = sess.run([sum_op],  feed_dict)
    summary_writer.add_summary(summary_str, itr)

def draw_samples(sess, hps, model, summary_writer, itr, traincond):
    if traincond == 1:
        sum_temp = model.x_sampled_grid_train
        bsize = hps.local_batch_train
    else:
        sum_temp = model.x_sampled_grid_test
        bsize = hps.local_batch_test

    for eps, sum in zip(*sum_temp):
        sample_images(eps, sess, model, summary_writer, itr, bsize, sum, traincond)

# ===
# Code for getting data
# ===
def get_data(hps, sess):
    if hps.image_size == -1:
        hps.image_size = {'mnist': 32, 'cifar10': 32, 'imagenet-oord': 64,
                          'imagenet': 256, 'celeba': 256, 'lsun_realnvp': 64, 'lsun': 256, 'cartgripper':64}[hps.problem]
    if hps.n_test == -1:
        hps.n_test = {'mnist': 10000, 'cifar10': 10000, 'imagenet-oord': 50000, 'imagenet': 50000,
                      'celeba': 3000, 'lsun_realnvp': 300, 'lsun': 300, 'cartgripper': 5000}[hps.problem]

    if hps.data_dir == "" and hps.problem != 'cartgripper':
        hps.data_dir = {'mnist': None, 'cifar10': None, 'imagenet-oord': '/mnt/host/imagenet-oord-tfr', 'imagenet': '/mnt/host/imagenet-tfr',
                        'celeba': '/mnt/host/celeba-reshard-tfr', 'lsun_realnvp': '/mnt/host/lsun_realnvp', 'lsun': '/mnt/host/lsun'}[hps.problem]

    if hps.problem == 'lsun_realnvp':
        hps.rnd_crop = True
    else:
        hps.rnd_crop = False

    if hps.category:
        hps.data_dir += ('/%s' % hps.category)

    # Use anchor_size to rescale batch size based on image_size
    s = hps.anchor_size
    hps.local_batch_train = hps.n_batch_train * \
        s * s // (hps.image_size * hps.image_size)
    hps.local_batch_test = {64: 50, 32: 25, 16: 10, 8: 5, 4: 2, 2: 2, 1: 1}[
        hps.local_batch_train]  # round down to closest divisor of 50
    hps.local_batch_init = hps.n_batch_init * \
        s * s // (hps.image_size * hps.image_size)

    if hps.problem in ['imagenet-oord', 'imagenet', 'celeba', 'lsun_realnvp', 'lsun']:
        hps.direct_iterator = True
        import data_loaders.get_data as v
        train_iterator, test_iterator, data_init = \
            v.get_data(sess, hps.data_dir, 1, 1, hps.pmap, hps.fmap, hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init, hps.image_size, hps.rnd_crop)

    elif hps.problem in ['mnist', 'cifar10']:
        hps.direct_iterator = False
        import data_loaders.get_mnist_cifar as v
        train_iterator, test_iterator, data_init = \
            v.get_data(hps.problem, 1, 1, hps.dal, hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init,  hps.image_size)

    elif hps.problem == 'cartgripper':
        hps.direct_iterator = True
        CONF['data_dir'] = hps.data_dir
        if 'weiss_gripper_20k' in hps.data_dir:
            CONF['sdim'] = 4
            CONF['adim'] = 5
            CONF['orig_size'] = [64, 64]
            CONF['batch_size'] = hps.local_batch_train
            train_iterator = build_tfrecord_single(CONF, 'train')
            CONF['batch_size'] = hps.local_batch_test
            test_iterator = build_tfrecord_single(CONF, 'test')
        else:

            from glow.data_loaders.base_dataset import BaseVideoDataset
            dataset = BaseVideoDataset(hps.data_dir, hps.local_batch_train)

            train_iterator = dataset.get_iterator('env/image_view0/encoded', 'train')
            test_iterator = dataset.get_iterator('env/image_view0/encoded', 'test')

        data_init = make_batch(sess, train_iterator, hps.local_batch_train, hps.local_batch_init)

    else:
        raise Exception()

    return train_iterator, test_iterator, data_init


def process_results(results):
    stats = ['loss', 'bits_x', 'bits_y', 'pred_loss']
    assert len(stats) == results.shape[0]
    res_dict = {}
    for i in range(len(stats)):
        res_dict[stats[i]] = "{:.4f}".format(results[i])
    return res_dict


def main(hps):

    # Initialize Horovod.

    if hps.eager == 1:
        tf.enable_eager_execution()

    # Create tensorflow session
    sess = tensorflow_session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6010')

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
    import glow.model as model
    model = model.model(sess, hps, train_iterator, test_iterator, data_init)
    summary_writer = tf.summary.FileWriter(hps.logdir, graph=sess.graph, flush_secs=10)

    _print(hps)
    _print('Starting training. Logging to', logdir)
    _print('epoch n_processed n_images ips dtrain dtest dsample dtot train_results test_results msg')

    # Train
    # tf.add_check_numerics_ops()
    sess.graph.finalize()
    n_processed = 0
    n_images = 0
    train_time = 0.0

    print('saving initial weights')
    model.save(logdir+"model_initial_weights.ckpt")

    for epoch in range(0, hps.epochs):
        t = time.time()

        print('starting epoch: ', epoch)
        for it in range(hps.train_its):
            tot_iter = it + epoch*hps.train_its

            if it % summ_interval == 0:
                print('--------------------')
                print('sampling images')
                draw_samples(sess, hps, model, summary_writer, tot_iter, traincond=1)

            # Set learning rate, linearly annealed from 0 in the first hps.epochs_warmup epochs.
            lr = hps.lr * min(1., n_processed /
                              (hps.n_train * hps.epochs_warmup))

            sum_str, stats = model.train(lr)

            if it % summ_interval == 0:
                summary_writer.add_summary(sum_str, tot_iter)

            if it % 100 == 0:
                print('itr {} starting epoch {}, results [local_loss, bits_x, bits_y, pred_loss]: {}'.format(it, epoch, stats))

            # Images seen wrt anchor resolution
            n_processed += hps.n_batch_train
            # Actual images seen at current resolution
            n_images +=  hps.local_batch_train

            if it % hps.save_interval == 0:
                # Save checkpoint
                model.save(logdir + "/model_{}.ckpt".format(it))

        dtrain = time.time() - t
        train_time += dtrain

        if epoch % hps.epochs_full_valid == 0:

            t = time.time()
            # model.polyak_swap()

            # Full validation run
            print('##############')
            print('testing model')
            test_loss, sum_str, _ = model.test()
            summary_writer.add_summary(sum_str, epoch)


    _print("Finished!")

# Get number of training and validation iterations


def get_its(hps):
    # These run for a fixed amount of time. As anchored batch is smaller, we've actually seen fewer examples
    train_its = int(np.ceil(hps.n_train / (hps.n_batch_train )))
    test_its = int(np.ceil(hps.n_test / (hps.n_batch_train )))
    train_epoch = train_its * hps.n_batch_train

    # Do a full validation run
    print(hps.n_test, hps.local_batch_test)
    assert hps.n_test % (hps.local_batch_test) == 0
    full_test_its = hps.n_test // (hps.local_batch_test)

    print("Train epoch size: " + str(train_epoch))
    return train_its, test_its, full_test_its


'''
Create tensorflow session with horovod
'''


def tensorflow_session():
    # Init session and params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Pin GPU to local rank (one GPU per process)
    config.gpu_options.visible_device_list = str(0)
    sess = tf.Session(config=config)
    return sess


if __name__ == "__main__":

    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager", type=int, default=0, help="use eager")

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
    parser.add_argument("--seq_len", type=int, default=30,
                        help="length of videosequnces to load")

    # New dataloader params
    parser.add_argument("--fmap", type=int, default=1,
                        help="# Threads for parallel file reading")
    parser.add_argument("--pmap", type=int, default=16,
                        help="# Threads for parallel map")

    # Optimization hyperparams:
    parser.add_argument("--n_train", type=int, default=1.8e6, help="Train epoch size")
    parser.add_argument("--n_test", type=int, default=9e4, help="Valid epoch size")
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
    parser.add_argument("--save_interval", type=int,
                        default=50, help="Epochs between save")
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
    parser.add_argument("--cond", type=str,
                        default=0, help="conditional model")
    parser.add_argument("--condactnorm", type=str,
                        default=1, help="whether to use actnorm in the conditional preprocessor")

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
    parser.add_argument("--use_lu_decomp", type=int, default=0,
                        help="whether to use LU-Decomposition: 0=dont use, 1=use")

    hps = parser.parse_args()  # So error if typo
    main(hps)
