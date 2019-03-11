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
  # keys = f(state)
  keys = tf.layers.conv2d(keys, filters=64, kernel_size=1, strides=1,
  padding="VALID", activation=tf.nn.relu)
  # patches
  patches=tf.extract_image_patches(images=frames, ksizes=[1,8,8,3], strides=[1,4,4,3],rates=[1,1,1,1], padding="SAME")
  patches = tf.reshape(patches, [-1,8,8,3])
  print(">>>>>>>>>>>>Size of variable patches : ", tf.shape(patches))
  print("   has to be 1440x8x8x3")
  # embedding pos to patch
  net = tf.layers.conv2d(patches, filters=32, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)
  skip1 = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
                             padding="SAME", activation=None)
  net = tf.layers.conv2d(net, filters=32, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    # TODO(ogroth): correct implementation for the skip connection?
  net = net + skip1
  net = tf.layers.conv2d(net, filters=64, kernel_size=2, strides=1, padding="SAME", activation=tf.nn.relu)

    # tile the poses to match the embedding shape
  height, width = tf.shape(net)[1], tf.shape(net)[2]
  poses = tf.reshape(poses, [-1,1,1,7])
  poses = tf.tile(poses, [1,8,8,1])

  temp = []
  for i in range(8):
    tt=[]
    for j in range(8):
        tt.append([i,j])
    temp.append(tt)
  empty=[]
  empty.append(temp)
  empty = np.array(empty)
  print(empty.shape) # 8 8 2

  new_poses = tf.convert_to_tensor(empty, dtype=tf.float32)
  # new_poses = tf.reshape(-1,8,8,2)
  if(tf.shape(new_poses[3])!=2):
      print("error on new_poses!")
  context_tmp = tf.shape(poses[0])
  patch_poses=tf.tile(new_poses,[context_tmp,1,1,1]) #1280 x 8 x 8 x 2
  if(tf.shape(new_poses[0])!=1280):
        print("error on number of patches for patch_poses!")
  total_poses = tf.concat([poses, patch_poses], axis=3) #1280 x 8 x 8 x 9
  if(tf.shape(new_poses[3])!=9):
        print("error on last dimension for patch_poses!")


    # concatenate the poses with the embedding
  net = tf.concat([net, total_poses], axis=3) # 1280 x 8 x 8 x 11
    # if(tf.shape(new_poses[3])!=12):
    #     print("error on last dimension for net!")
  skip2 = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
                             padding="SAME", activation=None)
  net = tf.layers.conv2d(net, filters=32, kernel_size=2, strides=1,
                           padding="SAME", activation=tf.nn.relu)
  net = net + skip2
  net = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)
  net = tf.layers.conv2d(net, filters=64, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)


    # patch image

  patched_keys=tf.reshape(keys, shape=[-1,1,1,64])
  packed_keys=tf.tile(patched_keys, [1,8,8,64]) #1280x8x8x64

    #TODO need to add attention dot product score

  patch_key_combine = tf.matmul(state, packed_keys, transpose_b=True)
  attention_softmax = tf.nn.softmax(patch_key_combine,axis=0)
  representation = tf.reduce_sum(attention_softmax*net, axis=0)


  return representation
