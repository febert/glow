import tensorflow as tf
import pdb

import glow.tfops as Z
import glow.optim as optim
import numpy as np
from tensorflow.contrib.framework.python.ops import add_arg_scope

from glow.tfops import actnorm_center

'''
f_loss: function with as input the (x,y,reuse=False), and as output a list/tuple whose first element is the loss.
'''




def abstract_model_xy(sess, hps, feeds, train_iterator, test_iterator, data_init, lr, f_loss, train=True):

    # == Create class with static fields and methods
    class m(object):
        pass
    m.sess = sess
    m.feeds = feeds
    m.lr = lr

    # === Loss and optimizer
    loss_train, stats_train = f_loss(train_iterator)
    #tf.add_check_numerics_ops()
    all_params = tf.trainable_variables()

    if train:
        if hps.gradient_checkpointing == 1:
            from glow.memory_saving_gradients import gradients
            gs = gradients(loss_train, all_params)
        else:
            gs = tf.gradients(loss_train, all_params)

        optimizer = {'adam': optim.adam, 'adamax': optim.adamax,
                     'adam2': optim.adam2}[hps.optimizer]

        train_op, polyak_swap_op, ema = optimizer(
            all_params, gs, alpha=lr, hps=hps)

    if hps.direct_iterator:
        train_sum_op = tf.summary.scalar('train_loss', loss_train)
        m.train = lambda _lr: sess.run([train_op, train_sum_op, stats_train], {lr: _lr})[1:]
    else:
        def _train(_lr):
            _x = train_iterator()
            train_sum_op = tf.summary.scalar('train_loss', loss_train)
            return sess.run([train_op, train_sum_op, stats_train], {feeds['x']: _x, lr: _lr})[1]
        m.train = _train

    m.polyak_swap = lambda: sess.run(polyak_swap_op)

    # === Testing
    loss_test, stats_test = f_loss(test_iterator, reuse=True)
    if hps.direct_iterator:
        test_sum_op = tf.summary.scalar('test_loss', loss_test)
        m.test = lambda: sess.run([loss_test, test_sum_op, stats_test])
    else:
        def _test():
            _x, _y = test_iterator()
            return sess.run([loss_test, stats_test], {feeds['x']: _x,
                                         feeds['y']: _y})
        m.test = _test

    # === Saving and restoring
    saver = tf.train.Saver(max_to_keep=3)
    if train:
        saver_ema = tf.train.Saver(ema.variables_to_restore())
        m.save_ema = lambda path: saver_ema.save(
            sess, path, write_meta_graph=False)
        m.save = lambda path: saver.save(sess, path, write_meta_graph=False)
    m.restore = lambda path: saver.restore(sess, path)

    # === Initialize the parameters
    if hps.restore_path != '':
        m.restore(hps.restore_path)
    else:
        with Z.arg_scope([Z.get_variable_ddi, Z.actnorm], init=True):   # TODO: add argscoping for
            results_init = f_loss(None, reuse=True)

        sess.run(tf.global_variables_initializer())
        if hps.cond == 1:
            sess.run(results_init, {feeds['x']: data_init['x'],
                                    feeds['cond']: data_init['cond']})
        else:
            sess.run(results_init, {feeds['x']: data_init['x']})

    return m


def codec(hps):

    def cond_encoder(cond, hps):
        def split_squeeze(z):
            n_z = Z.int_shape(z)[3]
            z1 = z[:, :, :, :n_z // 2]
            z1 = Z.squeeze2d(z1)
            return z1

        h = tf.concat(tf.unstack(cond, axis=1), -1)

        cond_all = [h]
        for i in range(hps.n_levels-1):
            with tf.variable_scope("cond_enc/level{}".format(i)):
                h = revnet2d_cond(h, hps)
                h = split_squeeze(h)

            cond_all.append(h)
        return cond_all

    def encoder(z, cond_all, objective):
        print('building encoder')
        z_all = []
        for i in range(hps.n_levels):
            z_shape = z.get_shape().as_list()
            print('encoder layer {} shape {}, num_dim {}'.format(i, z.get_shape(), z_shape[1]*z_shape[2]*z_shape[3]))
            if cond_all is None:
                z, objective = revnet2d(str(i), z, objective, hps)
            else:
                #tf.add_check_numerics_ops()
                z, objective = revnet2d(str(i), z, objective, hps, cond=cond_all[i])

            if i < hps.n_levels-1:
                z, z_all, objective = split2d("pool"+str(i), z, z_all, objective=objective)
        z_all.append(z)
        return z, z_all, objective

    def decoder(z, cond_all, eps_std, z_all=None):
        print('building decoder')

        for i in reversed(range(hps.n_levels)):
            z_shape = z.get_shape().as_list()
            print('decoder layer {} shape {}, num_dim {}'.format(i, z.get_shape(), z_shape[1]*z_shape[2]*z_shape[3]))
            if i < hps.n_levels-1:
                z = split2d_reverse("pool"+str(i), z, z_all, eps_std=eps_std)
            if cond_all is not None:
                z, _ = revnet2d(str(i), z, 0, hps, reverse=True, cond=cond_all[i])
            else:
                z, _ = revnet2d(str(i), z, 0, hps, reverse=True)

        return z

    return encoder, cond_encoder, decoder


def prior(name, hps, bsize):

    with tf.variable_scope(name):
        n_z = hps.top_shape[-1]

        h = tf.zeros([bsize]+hps.top_shape[:2]+[2*n_z])
        if hps.learntop:
            h = Z.conv2d_zeros('p', h, 2*n_z)

        pz = Z.gaussian_diag(h[:, :, :, :n_z], h[:, :, :, n_z:])

    def logp(z1):
        objective = pz.logp(z1)
        return objective

    def sample(eps_std=None):
        if eps_std is not None:
            z = pz.sample2(pz.eps * tf.reshape(eps_std, [-1, 1, 1, 1]))
        else:
            z = pz.sample

        return z

    return logp, sample


def model(sess, hps, train_iterator, test_iterator, data_init, train=True):

    ncontxt = 2
    # Only for decoding/init, rest use iterators directly
    if hps.eager == 0:
        with tf.name_scope('input'):
            X = tf.placeholder(tf.uint8, [None,  hps.image_size, hps.image_size, 3], name='image')
            Cond = tf.placeholder(tf.uint8, [None, ncontxt, hps.image_size, hps.image_size, 3], name='cond')
            Y = tf.placeholder(tf.int32, [None], name='label')
            lr = tf.placeholder(tf.float32, None, name='learning_rate')

    else:
        lr = hps.lr
        traj = train_iterator.get_next()
        X = traj[:, 2]
        Cond = traj[:, :2]

    encoder, cond_encoder, decoder = codec(hps)
    hps.n_bins = 2. ** hps.n_bits_x

    def preprocess(x):
        x = tf.cast(x, 'float32')
        if hps.n_bits_x < 8:
            x = tf.floor(x / 2 ** (8 - hps.n_bits_x))
        x = x / hps.n_bins - .5
        return x

    def postprocess(x):
        return tf.cast(tf.clip_by_value(tf.floor((x + .5)*hps.n_bins)*(256./hps.n_bins), 0, 255), 'uint8')

    def _f_loss(x, reuse=False, cond=None):

        with tf.variable_scope('model', reuse=reuse):
            #tf.add_check_numerics_ops()

            objective = tf.zeros_like(x, dtype='float32')[:, 0, 0, 0]

            z = preprocess(x)
            z = z + tf.random_uniform(tf.shape(z), 0, 1./hps.n_bins)

            objective += - np.log(hps.n_bins) * np.prod(Z.int_shape(z)[1:])

            # Encode
            z = Z.squeeze2d(z, 2)  # > 16x16x12
            #tf.add_check_numerics_ops()

            if cond is not None:
                cond = preprocess(cond)
                cond = Z.squeeze2d(cond, 2)  # > 16x16x12
                cond_all = cond_encoder(cond, hps)
            else:
                cond_all = None


            z, z_all, objective = encoder(z, cond_all, objective)

            hps.top_shape = Z.int_shape(z)[1:]

            # Prior
            logp, _ = prior("prior", hps, tf.shape(x)[0])
            objective += logp(z)

            # Generative loss
            nobj = - objective
            bits_x = nobj / (np.log(2.) * int(x.get_shape()[1]) * int(
                x.get_shape()[2]) * int(x.get_shape()[3]))  # bits per subpixel

            bits_y = tf.zeros_like(bits_x)
            classification_error = tf.ones_like(bits_x)

            #tf.add_check_numerics_ops()

        return bits_x, bits_y, classification_error

    # === Sampling function
    def f_decode(cond, eps_std):
        with tf.variable_scope('model', reuse=True):
            _, sample = prior("prior", hps, hps.local_batch_train)
            z = sample(eps_std)

            if cond is not None:
                cond = preprocess(cond)
                cond = Z.squeeze2d(cond, 2)  # > 16x16x12
                cond_all = cond_encoder(cond, hps)
            else:
                cond_all = None

            z = decoder(z, cond_all, eps_std)
            z = Z.unsqueeze2d(z, 2)  # 8x8x12 -> 16x16x3
            x = postprocess(z)
        return x

    def f_loss(iterator, reuse=False):
        if hps.direct_iterator and iterator is not None:
            traj = iterator.get_next()

            start_ind = tf.random_uniform((),0, hps.seq_len-ncontxt, dtype=tf.int32)
            x = traj[:, start_ind + ncontxt]
            if hps.cond == 1:
                cond = traj[:, start_ind:start_ind+ncontxt]
                cond = tf.reshape(cond, [-1, ncontxt, x.get_shape()[1], x.get_shape()[2], 3])
            else:
                cond = None
        else:
            if hps.cond == 1:
                x, cond = X, Cond
            else:
                x, cond = X, None

        bits_x = _f_loss(x, reuse, cond)

        local_loss = bits_x

        stats = [local_loss, bits_x]
        global_stats = Z.allreduce_mean(
            tf.stack([tf.reduce_mean(i) for i in stats]))

        return tf.reduce_mean(local_loss), global_stats


    def build_exmp_encoder(x_pl, reuse=False):
        """
        Encode a batch of examples into latents
        :param iterator:
        :return: batch of latents
        """

        z = preprocess(x_pl)
        # z = z + tf.random_uniform(tf.shape(z), 0, 1. / hps.n_bins)

        with tf.variable_scope('model', reuse=reuse):
            # Encode
            z = Z.squeeze2d(z, 2)  # > 16x16x12
            objective = tf.zeros_like(z, dtype='float32')[:, 0, 0, 0]
            z, z_all, objective = encoder(z, objective)

        return z_all


    def build_exmp_decoder(z_all, cond):
        """
        Decode (possibly interpolated) latent into image
        """
        with tf.variable_scope('model', reuse=True):
            z_start = z_all.pop(-1)
            cond = Z.squeeze2d(cond, 2)  # > 16x16x12
            cond_all = cond_encoder(cond)
            z = decoder(z_start, cond_all, z_all=z_all, eps_std=0)
            z = Z.unsqueeze2d(z, 2)
            x = postprocess(z)
        return x

    feeds = {'x': X, 'cond':Cond}
    m = abstract_model_xy(sess, hps, feeds, train_iterator,
                          test_iterator, data_init, lr, f_loss, train=train)

    # === Decoding functions
    m.eps_std = tf.placeholder(tf.float32, [None], name='eps_std')
    m.train_cond = tf.placeholder(tf.int64, (), 'traincond')

    if hps.cond == 1:
        ###################
        # cond_data = tf.cond(m.train_cond > 0,
        #                # if 1 use trainigbatch else validation batch
        #                train_iterator.get_next,
        #                test_iterator.get_next)[0]

        cond_data = train_iterator.get_next()[0]


        #pick the first timestep
        cond_data = cond_data[:, :ncontxt]
    else:
        cond_data = None

    x_sampled = f_decode(cond_data, m.eps_std)

    def m_decode(_eps_std):
        return m.sess.run(x_sampled, {m.eps_std: _eps_std})
    m.decode = m_decode

    m.x_sampled = x_sampled
    m.x_sampled_grid_train = get_grid(x_sampled, 1, hps)
    # m.x_sampled_grid_test = get_grid(x_sampled, 0, hps)

    m.build_exmp_encoder = build_exmp_encoder
    m.build_exmp_decoder = build_exmp_decoder
    return m

def get_grid(x_sampled, traincond, hps):
    im_height, im_width = x_sampled.get_shape().as_list()[1:3]

    if traincond == 1:
        bsize = hps.local_batch_train
        suf = 'train'
    else:
        bsize = hps.local_batch_test
        suf = 'test'

    width = int(np.floor(np.sqrt(bsize)))  # result width
    height = int(bsize / width)  # result height

    x_sampled = tf.reshape(x_sampled, [bsize, im_height, im_width, 3])
    x_sampled = tf.unstack(x_sampled, axis=0)
    rows = []
    for r in range(height):
        rows.append(tf.concat(x_sampled[r * width:(r + 1) * width], axis=1))
    image = tf.concat(rows, axis=0)[None]

    temperatures = [0., .25, .5, .6, .7, .8, .9, 1.]
    summaries = [tf.summary.image('temp{}_{}'.format(eps, suf), image) for eps in temperatures]
    return temperatures, summaries


def checkpoint(z, cond, logdet):
    zshape = Z.int_shape(z)
    num_zvals = zshape[1]*zshape[2]*zshape[3]
    z = tf.reshape(z, [-1, num_zvals])

    logdet = tf.reshape(logdet, [-1, 1])
    if cond is not None:
        condshape = Z.int_shape(cond)
        num_condvals = condshape[1]*condshape[2]*condshape[3]
        cond = tf.reshape(cond, [-1, num_condvals])

        combined = tf.concat([z, cond, logdet], axis=1)
        tf.add_to_collection('checkpoints', combined)
        logdet = combined[:, -1]    # why don't we take the original z, cond and logdet here????
        z = tf.reshape(combined[:, :num_zvals], [-1, zshape[1], zshape[2], zshape[3]])
        cond = tf.reshape(combined[:, num_zvals:num_zvals + num_condvals], [-1, condshape[1], condshape[2], condshape[3]])
    else:
        combined = tf.concat([z, logdet], axis=1)
        tf.add_to_collection('checkpoints', combined)
        logdet = combined[:, -1]    # why don't we take the original z, cond and logdet here????
        z = tf.reshape(combined[:, :num_zvals], [-1, zshape[1], zshape[2], zshape[3]])
    return z, cond, logdet


def checkpoint_cond(cond):
    condshape = Z.int_shape(cond)
    num_condvals = condshape[1]*condshape[2]*condshape[3]
    cond = tf.reshape(cond, [-1, num_condvals])
    tf.add_to_collection('checkpoints', cond)
    cond = tf.reshape(cond, [-1, condshape[1], condshape[2], condshape[3]])
    return cond


@add_arg_scope
def revnet2d(name, z, logdet, hps, reverse=False, cond=None):
    with tf.variable_scope(name):
        if not reverse:
            for i in range(hps.depth):
                z, cond, logdet = checkpoint(z, cond, logdet)
                z, logdet = revnet2d_step(str(i), z, cond, logdet, hps, reverse)
            z, cond, logdet = checkpoint(z, cond, logdet)
        else:
            for i in reversed(range(hps.depth)):
                z, logdet = revnet2d_step(str(i), z, cond, logdet, hps, reverse)
    return z, logdet


def revnet2d_cond(h, hps):
    for d in range(hps.depth):
        with tf.variable_scope("d{}".format(d)):
            n_out = int(h.get_shape()[3])
            width = hps.width
            h = tf.nn.relu(Z.conv2d("l_1", h, width))
            h = tf.nn.relu(Z.conv2d("l_2", h, width, filter_size=[1, 1]))
            h = Z.conv2d_zeros("l_last", h, n_out)

            if hps.condactnorm == 1:
                h = Z.actnorm("actnorm_cond_d{}".format(d), h)
                # with tf.variable_scope("actnorm_cond_output_d{}".format(d)):
                #     h = tf.identity(h, name="actnorm_cond_output_d{}".format(d))
                #     h = tf.print(h, [tf.reduce_sum(h[0])])

        h = checkpoint_cond(h)
    return h

# Simpler, new version


@add_arg_scope
def revnet2d_step(name, z, cond, logdet, hps, reverse):
    with tf.variable_scope(name):

        shape = Z.int_shape(z)
        n_z = shape[3]
        assert n_z % 2 == 0

        if not reverse:

            z, logdet = Z.actnorm("actnorm", z, logdet=logdet)

            if hps.flow_permutation == 0:
                z = Z.reverse_features("reverse", z)
            elif hps.flow_permutation == 1:
                z = Z.shuffle_features("shuffle", z)
            elif hps.flow_permutation == 2:
                z, logdet = invertible_1x1_conv(hps, "invconv", z, logdet)
            else:
                raise Exception()

            z1 = z[:, :, :, :n_z // 2]
            z2 = z[:, :, :, n_z // 2:]

            if hps.flow_coupling == 0:
                z2 += f("f1", z1, cond, hps.width, n_z//2)
            elif hps.flow_coupling == 1:
                h = f("f1", z1, cond, hps.width, n_z)
                shift = h[:, :, :, 0::2]
                #scale = tf.exp(h[:, :, :, 1::2])
                scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.) + 1e-10
                z2 += shift
                z2 *= scale
                logdet += tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
            else:
                raise Exception()

            z = tf.concat([z1, z2], 3)

        else:

            z1 = z[:, :, :, :n_z // 2]
            z2 = z[:, :, :, n_z // 2:]

            if hps.flow_coupling == 0:
                z2 -= f("f1", z1, cond, hps.width, n_z//2)
            elif hps.flow_coupling == 1:
                h = f("f1", z1, cond, hps.width, n_z)
                shift = h[:, :, :, 0::2]
                #scale = tf.exp(h[:, :, :, 1::2])
                scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.) + 1e-10
                z2 /= scale
                z2 -= shift
                logdet -= tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
            else:
                raise Exception()

            z = tf.concat([z1, z2], 3)

            if hps.flow_permutation == 0:
                z = Z.reverse_features("reverse", z, reverse=True)
            elif hps.flow_permutation == 1:
                z = Z.shuffle_features("shuffle", z, reverse=True)
            elif hps.flow_permutation == 2:
                z, logdet = invertible_1x1_conv(hps,
                    "invconv", z, logdet, reverse=True)
            else:
                raise Exception()

            z, logdet = Z.actnorm("actnorm", z, logdet=logdet, reverse=True)

    return z, logdet


def f(name, h, cond, width, n_out=None):
    if cond is not None:
        h = tf.concat([h, cond], axis=3)
    n_out = n_out or int(h.get_shape()[3])
    with tf.variable_scope(name):
        h = tf.nn.relu(Z.conv2d("l_1", h, width))
        h = tf.nn.relu(Z.conv2d("l_2", h, width, filter_size=[1, 1]))
        h = Z.conv2d_zeros("l_last", h, n_out)
    return h


def f_resnet(name, h, width, n_out=None):
    n_out = n_out or int(h.get_shape()[3])
    with tf.variable_scope(name):
        h = tf.nn.relu(Z.conv2d("l_1", h, width))
        h = Z.conv2d_zeros("l_2", h, n_out)
    return h

# Invertible 1x1 conv


@add_arg_scope
def invertible_1x1_conv(hps, name, z, logdet, reverse=False):


    if not hps.use_lu_decomp:  # Set to "False" to use the LU-decomposed version

        with tf.variable_scope(name):

            shape = Z.int_shape(z)
            w_shape = [shape[3], shape[3]]

            # Sample a random orthogonal matrix:
            w_init = np.linalg.qr(np.random.randn(
                *w_shape))[0].astype('float32')

            w = tf.get_variable("W", dtype=tf.float32, initializer=w_init)

            #dlogdet = tf.linalg.LinearOperator(w).log_abs_determinant() * shape[1]*shape[2]
            dlogdet = tf.cast(tf.log(abs(tf.matrix_determinant(
                tf.cast(w, 'float64')))), 'float32') * shape[1]*shape[2]

            if not reverse:

                _w = tf.reshape(w, [1, 1] + w_shape)
                z = tf.nn.conv2d(z, _w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet += dlogdet

                return z, logdet
            else:

                _w = tf.matrix_inverse(w)
                _w = tf.reshape(_w, [1, 1]+w_shape)
                z = tf.nn.conv2d(z, _w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet -= dlogdet

                return z, logdet

    else:
        # LU-decomposed version
        shape = Z.int_shape(z)
        with tf.variable_scope(name):

            dtype = 'float64'

            # Random orthogonal matrix:
            import scipy
            np_w = scipy.linalg.qr(np.random.randn(shape[3], shape[3]))[
                0].astype('float32')

            np_p, np_l, np_u = scipy.linalg.lu(np_w)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(abs(np_s))
            np_u = np.triu(np_u, k=1)

            p = tf.get_variable("P", initializer=np_p, trainable=False)
            l = tf.get_variable("L", initializer=np_l)
            sign_s = tf.get_variable(
                "sign_S", initializer=np_sign_s, trainable=False)
            log_s = tf.get_variable("log_S", initializer=np_log_s)
            #S = tf.get_variable("S", initializer=np_s)
            u = tf.get_variable("U", initializer=np_u)

            p = tf.cast(p, dtype)
            l = tf.cast(l, dtype)
            sign_s = tf.cast(sign_s, dtype)
            log_s = tf.cast(log_s, dtype)
            u = tf.cast(u, dtype)

            w_shape = [shape[3], shape[3]]

            l_mask = np.tril(np.ones(w_shape, dtype=dtype), -1)
            l = l * l_mask + tf.eye(*w_shape, dtype=dtype)
            u = u * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
            w = tf.matmul(p, tf.matmul(l, u))

            if True:
                u_inv = tf.matrix_inverse(u)
                l_inv = tf.matrix_inverse(l)
                p_inv = tf.matrix_inverse(p)
                w_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))
            else:
                w_inv = tf.matrix_inverse(w)

            w = tf.cast(w, tf.float32)
            w_inv = tf.cast(w_inv, tf.float32)
            log_s = tf.cast(log_s, tf.float32)

            if not reverse:

                w = tf.reshape(w, [1, 1] + w_shape)
                z = tf.nn.conv2d(z, w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet += tf.reduce_sum(log_s) * (shape[1]*shape[2])

                return z, logdet
            else:

                w_inv = tf.reshape(w_inv, [1, 1]+w_shape)
                z = tf.nn.conv2d(
                    z, w_inv, [1, 1, 1, 1], 'SAME', data_format='NHWC')
                logdet -= tf.reduce_sum(log_s) * (shape[1]*shape[2])

                return z, logdet


@add_arg_scope
def split2d(name, z, z_all=None, objective=0.):
    with tf.variable_scope(name):
        n_z = Z.int_shape(z)[3]
        z1 = z[:, :, :, :n_z // 2]
        z2 = z[:, :, :, n_z // 2:]
        pz = split2d_prior(z1)
        objective += pz.logp(z2)
        z1 = Z.squeeze2d(z1)
        if z_all is not None:
            z_all.append(z2)
            return z1, z_all, objective
        else:
            return z1, objective


@add_arg_scope
def split2d_reverse(name, z, z_all, eps_std=None):
    with tf.variable_scope(name):
        z1 = Z.unsqueeze2d(z)

        if z_all is not None:
            z2 = z_all.pop(-1)
        else:
            pz = split2d_prior(z1)
            z2 = pz.sample
            if eps_std is not None:
                z2 = pz.sample2(pz.eps * tf.reshape(eps_std, [-1, 1, 1, 1]))

        z = tf.concat([z1, z2], 3)
        return z


@add_arg_scope
def split2d_prior(z):
    n_z2 = int(z.get_shape()[3])
    n_z1 = n_z2
    h = Z.conv2d_zeros("conv", z, 2 * n_z1)

    mean = h[:, :, :, 0::2]
    logs = h[:, :, :, 1::2]
    return Z.gaussian_diag(mean, logs)
