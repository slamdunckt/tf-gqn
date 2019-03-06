"""
Contains the graph definition of the GQN encoding stack.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .gqn_utils import broadcast_pose
"""
    For minecraft sequnetial attention
    architecture as follows
        k=2 s=2 f=32
        k=3 s=1 f=32
        k=2 s=2 f=64
        k=3 s=1 f=64
        k=3 s=1 f=64
        k=3 s=1 f=64
        k=5 s=1 f=4 ->each channel generate one of the 4-logprobability map
                    MLP with 3 layer, log-softmax normalizer with one layer

"""


def patch_encoder(frames: tf.Tensor, poses: tf.Tensor, scope="PatchEncoder"):
  """
  Feed-forward convolutional architecture.
  Only for Seq attention model
  """
  with tf.variable_scope(scope):
    endpoints = {}
    net = tf.layers.conv2d(frames, filters=32, kernel_size=2, strides=2,
                           padding="VALID", activation=tf.nn.relu)
    skip1 = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
                             padding="SAME", activation=None)
    net = tf.layers.conv2d(net, filters=32, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    # TODO(ogroth): correct implementation for the skip connection?
    net = net + skip1
    net = tf.layers.conv2d(net, filters=64, kernel_size=2, strides=2, padding="VALID", activation=tf.nn.relu)

    # tile the poses to match the embedding shape
    height, width = tf.shape(net)[1], tf.shape(net)[2]
    poses = broadcast_pose(poses, height, width)

    temp = [[]]
    for i in range(height):
        for j in range(width):
            temp[i][j]=[i,j]

    new_poses = tf.convert_to_tensor(temp, dtype=tf.float32)
    new_poses = tf.reshape(1,1,1,-1)
    if(tf.shape(new_poses[3])!=2):
        print("error on patch poses!")
    context_tmp = tf.shape(poses[0])
    patch_poses=tf.tile(new_poses,[context_tmp,height,width,1])

    total_poses = tf.concat([poses, patch_poses], axis=3)

    # concatenate the poses with the embedding
    net = tf.concat([net, total_poses], axis=3)

    skip2 = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
                             padding="SAME", activation=None)
    net = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    # TODO(ogroth): correct implementation for the skip connection?
    net = net + skip2

    net = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    net = tf.layers.conv2d(net, filters=64, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    return net, endpoints


def pool_encoder(frames: tf.Tensor, poses: tf.Tensor, scope="PoolEncoder"):
  """
  Feed-forward convolutional architecture with terminal global pooling.
  """
  net, endpoints = tower_encoder(frames, poses, scope)
  with tf.variable_scope(scope):
    net = tf.reduce_mean(net, axis=[1, 2], keepdims=True)

  return net, endpoints


def maxpool_encoder(frames: tf.Tensor, poses: tf.Tensor, scope="PoolEncoder"):
  """
  Feed-forward convolutional architecture with terminal global max pooling.
  """
  net, endpoints = tower_encoder(frames, poses, scope)
  with tf.variable_scope(scope):
    net = tf.reduce_max(net, axis=[1, 2], keepdims=True)

  return net, endpoints