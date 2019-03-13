"""
Contains the graph definition of the GQN encoding stack.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from .gqn_params import GQNConfig
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

    # TODO(ogroth): correct implementation for the skip connection?
    net = net + skip1
    net = tf.layers.conv2d(net, filters=64, kernel_size=2, strides=2, padding="VALID", activation=tf.nn.relu)
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


def patcher(frames: tf.Tensor, poses: tf.Tensor, keys:tf.Tensor, state:tf.Tensor, scope="Patcher"):
  """
  Attention algorithm goes here with patching
  """
  _CONTEXT_SIZE = GQNConfig.CONTEXT_SIZE
  _DIM_C_ENC = GQNConfig.ENC_CHANNELS
  _BATCH_SIZE = GQNConfig.BATCH_SIZE
  # keys = f(state)

  state_keys = tf.layers.conv2d(state, filters=64, kernel_size=1, strides=1,padding="VALID", activation=tf.nn.relu)
  state_keys = tf.reshape(state_keys, [36, -1, 64]) # 36 x 64 x 64
  state_keys = tf.tile(state_keys, [1280,1,1])

  # patches now 1280(36) x 32 x 32 x 3
  patches=tf.extract_image_patches(images=frames, ksizes=[1,8,8,1], strides=[1,4,4,1],rates=[1,1,1,1], padding="SAME")
  patches = tf.reshape(patches, [-1,8,8,3])


  # embedding pos to patch
  net = tf.layers.conv2d(patches, filters=32, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)

  skip1 = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
                             padding="SAME", activation=None)
  net = tf.layers.conv2d(net, filters=32, kernel_size=2, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    # TODO(ogroth): correct implementation for the skip connection?
  net = net + skip1
  net = tf.layers.conv2d(net, filters=64, kernel_size=2, strides=1, padding="SAME", activation=tf.nn.relu)
  # patches now 1280(36) x 8 x 8 x 64

  # tile the poses to match the embedding shape
  poses = tf.reshape(poses, [-1,1,1,7]) # 20(36) x 1 x 1 x 7
  # print(poses.get_shape())
  poses = tf.tile(poses, [64,8,8,1]) # 1280(36) x 8 x 8 x 7

  temp = []
  for i in range(8):
    tt=[]
    for j in range(8):
        tt.append([i,j])
    temp.append(tt)
  empty=[]
  empty.append(temp)
  empty = np.array(empty)# 1 8 8 2
  new_poses = tf.convert_to_tensor(empty, dtype=tf.float32)
  # new_poses = tf.reshape(-1,8,8,2)

  patch_poses=tf.tile(new_poses,[46080,1,1,1]) #1280(36) x 8 x 8 x 2
  total_poses = tf.concat([poses, patch_poses], axis=3) #1280(36) x 8 x 8 x 9
  # print(">>>>>>>>>>>>>>>>",total_poses.get_shape()) #1280(36)x8x8x9

  # concatenate the poses with the embedding
  net = tf.concat([net, total_poses], axis=3) # 1280 x 8 x 8 x 11

  skip2 = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
                             padding="SAME", activation=None)
  net = tf.layers.conv2d(net, filters=32, kernel_size=2, strides=1,
                           padding="SAME", activation=tf.nn.relu)
  net = net + skip2
  net = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)
  net = tf.layers.conv2d(net, filters=64, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)
  keys =tf.tile(keys, [1280,1,1,1])
  packed_keys = tf.reshape(keys, [-1,64,64])
    # patch image
  # print(">>>>>>>>>>>>>>>>>",net.get_shape())
  # patched_keys=tf.reshape(state_keys, shape=[-1,1,1,64])
  # tiled_keys=tf.tile(patched_keys, [1,8,8,1]) #1280x8x8x64
    #TODO need to add attention dot product score
  # print(">>>>>>>>>>>>>>>>>",state_keys.get_shape())
  patch_key_combine = tf.matmul(state_keys, packed_keys, transpose_b=True)
  patch_key_combine = tf.reshape(patch_key_combine, [-1,8,8,64])
  # print(">>>>>>>>>>>>>>>>>",patch_key_combine.get_shape())
  attention_softmax = tf.nn.softmax(patch_key_combine,axis=0)
  # print(">>>>>>>>>>>>>>>>>",attention_softmax.get_shape())

  batch_net = tf.reshape(net, [36, -1, 8, 8, 64])
  attention_softmax = tf.reshape(attention_softmax, [36, -1, 8, 8, 64])
  representation = tf.reduce_sum(attention_softmax*batch_net, axis=1)




  return representation
