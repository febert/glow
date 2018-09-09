import glow.model as model
import tensorflow as tf
import numpy as np
import os

def setup_embedding(conf, gpu_id = 0):
    """
    Setup up the network for control
    :param conf_file:
    :return: function which predicts a batch of whole trajectories
    conditioned on the actions
    """
    if gpu_id == None:
        gpu_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print('using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"])

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    g_predictor = tf.Graph()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph= g_predictor)
    with sess.as_default():
        with g_predictor.as_default():

            x1_pl = tf.placeholder(dtype=tf.float32, shape=[48,64])
            lt1 = model.build_exmp_encoder(x1_pl, reuse=True)

    def get_embedding(image):
        feed_dict = {
            x1_pl:image
        }
        z_all = sess.run(lt1, feed_dict=feed_dict)
        return z_all

    return get_embedding