"""
Contains the graph definition of the GQN encoding stack.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from .gqn_params import GQN_DEFAULT_CONFIG
from .gqn_utils import broadcast_pose

def patch_encoder(frames: tf.Tensor, poses: tf.Tensor, scope="PatchEncoder"):
  """
  Feed-forward convolutional architecture.
  Only for Seq attention model
  """
  # print("??????????",tf.shape(frames))
  with tf.variable_scope(scope):
    endpoints = {}
    net = tf.layers.conv2d(frames, filters=32, kernel_size=2, strides=2,
                           padding="VALID", activation=tf.nn.relu)
    skip1 = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
                             padding="SAME", activation=None)
    net = tf.layers.conv2d(net, filters=32, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    net = net + skip1
    net = tf.layers.conv2d(net, filters=64, kernel_size=2, strides=2, padding="VALID", activation=tf.nn.relu)
    skip2 = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
                             padding="SAME", activation=None)
    net = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    net = net + skip2

    net = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    net = tf.layers.conv2d(net, filters=64, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    return net, endpoints


def patcher(patch_dic: tf.Tensor,keys:tf.Tensor, state:tf.Tensor, scope="Patcher"):
  """
  state key will be computed here
  """
  _BATCH_SIZE = GQN_DEFAULT_CONFIG.BATCH_SIZE
  _IMG_NUM = GQN_DEFAULT_CONFIG.CONTEXT_SIZE * 64
  # keys = f(state)

  state_keys = tf.layers.conv2d(state, filters=64, kernel_size=1, strides=1,padding="VALID", activation=tf.nn.relu)
  state_keys = tf.reshape(state_keys, [_BATCH_SIZE, -1, 64]) # 36 x 64 x 64
  state_keys = tf.tile(state_keys, [_IMG_NUM,1,1]) #1280(36) x 64 x 64

  keys =tf.tile(keys, [_IMG_NUM,1,1,1])
  packed_keys = tf.reshape(keys, [-1,64,64])

  patch_key_combine = tf.matmul(state_keys, packed_keys, transpose_b=True)
  patch_key_combine = tf.reshape(patch_key_combine, [-1,8,8,64])

  attention_softmax = tf.nn.softmax(patch_key_combine,axis=0)
  # print(">>>>>>>>>>>>>>>>>",attention_softmax.get_shape())

  patch_dic_in = tf.reshape(patch_dic, [_BATCH_SIZE, -1, 8, 8, 64])
  attention_softmax = tf.reshape(attention_softmax, [_BATCH_SIZE, -1, 8, 8, 64])
  representation = tf.reduce_sum(attention_softmax*patch_dic_in, axis=1)




  return representation
