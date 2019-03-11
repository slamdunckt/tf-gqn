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
    """
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
    """
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


def patcher(frames: tf.Tensor, poses: tf.Tensor, keys:tf.Tensor, scope="Patcher"):
  """
  Attention algorithm goes here with patching
  """
    _CONTEXT_SIZE = model_params.CONTEXT_SIZE
    _DIM_C_ENC = model_params.ENC_CHANNELS
# keys = f(state)
    keys = tf.layers.conv2d(keys, filters=64, kernel_size=1, strides=1,
      padding="VALID", activation=tf.nn.relu)
# patches
    patches=tf.extract_image_patches(images=frames, ksizes=[1,8,8,3], strides=[1,4,4,3], padding="SAME")
    patches = tf.reshape(patches, [-1,8,8,3])
    print("Size of variable patches : ", tf.shape(patches))
    print("   has to be 1440x8x8x3")
"""
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
"""
    # tile the poses to match the embedding shape
    height, width = tf.shape(net)[1], tf.shape(net)[2]
    poses = tf.reshape(poses, [-1,1,1,7])
    poses = tf.tile(poses, [1,8,8,1])

    temp = [[]]
    for i in range(height):
        for j in range(width):
            temp[i][j]=[i,j]

    new_poses = tf.convert_to_tensor(temp, dtype=tf.float32)
    new_poses = tf.reshape(1,8,8,-1)
    if(tf.shape(new_poses[3])!=2):
        print("error on patch poses!")
    context_tmp = tf.shape(poses[0])
    patch_poses=tf.tile(new_poses,[context_tmp,1,1,1]) #1280 x 8 x 8 x 2
    if(tf.shape(new_poses[0])!=1280):
        print("error on number of patches for patch_poses!")
    total_poses = tf.concat([poses, patch_poses], axis=3) #1280 x 8 x 8 x 9
    if(tf.shape(new_poses[3])!=9):
        print("error on last dimension for patch_poses!")


    # concatenate the poses with the embedding
    net = tf.concat([net, total_poses], axis=3) # 1280 x 8 x 8 x 11
    if(tf.shape(new_poses[3])!=11):
        print("error on last dimension for net!")
    # skip2 = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
    #                          padding="SAME", activation=None)
    # net = tf.layers.conv2d(net, filters=32, kernel_size=2, strides=1,
    #                        padding="SAME", activation=tf.nn.relu)
    # net = net + skip2
    # net = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
    #                        padding="SAME", activation=tf.nn.relu)
    # net = tf.layers.conv2d(net, filters=64, kernel_size=1, strides=1,
    #                        padding="SAME", activation=tf.nn.relu)


# patch image

    patched_keys=tf.reshape(keys, shape=[-1,1,1,64])
    packed_keys=tf.tile(patched_keys, [1,8,8,64]) #1280x8x8x64

    patch_key_combine = tf.matmul(patches, packed_keys, transpose_b=True)
    attention_softmax = tf.nn.softmax(patch_key_combine,axis=0)





  return net, endpoints

 def _reduce_packed_representation(enc_r_packed, model_params):
     _CONTEXT_SIZE = model_params.CONTEXT_SIZE
     _DIM_C_ENC = model_params.ENC_CHANNELS
     height, width = tf.shape(enc_r_packed)[1], tf.shape(enc_r_packed)[2]

     enc_r_unpacked = tf.reshape(
        enc_r_packed, shape=[-1, _CONTEXT_SIZE, height, width, _DIM_C_ENC])

    # add scene representations per data tuple
    enc_r = tf.reduce_sum(enc_r_unpacked, axis=1)

    return enc_r
