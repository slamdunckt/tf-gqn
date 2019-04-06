"""
Contains the GQN graph definition.
Original paper:
'Neural scene representation and rendering'
S. M. Ali Eslami, Danilo J. Rezende, Frederic Besse, Fabio Viola, Ari S. Morcos,
Marta Garnelo, Avraham Ruderman, Andrei A. Rusu, Ivo Danihelka, Karol Gregor,
David P. Reichert, Lars Buesing, Theophane Weber, Oriol Vinyals, Dan Rosenbaum,
Neil Rabinowitz, Helen King, Chloe Hillier, Matt Botvinick, Daan Wierstra,
Koray Kavukcuoglu and Demis Hassabis
https://deepmind.com/documents/211/Neural_Scene_Representation_and_Rendering_preprint.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from .gqn_params import GQN_DEFAULT_CONFIG, GQNConfig
from .gqn_encoder import tower_encoder, pool_encoder
from .gqn_encoder_attention import patch_encoder, patcher
from .gqn_draw_attention import inference_rnn, generator_rnn
from .gqn_utils import broadcast_encoding, compute_eta_and_sample_z
from .gqn_vae import vae_tower_decoder


_ENC_FUNCTIONS = {
    'pool' : pool_encoder,
    'tower' : tower_encoder,
    'patch' : patch_encoder,
}

def _pack_context(context_poses, context_frames, model_params):
  # shorthand notations for model parameters
  _DIM_POSE = model_params.POSE_CHANNELS
  _DIM_H_IMG = model_params.IMG_HEIGHT
  _DIM_W_IMG = model_params.IMG_WIDTH
  _DIM_C_IMG = model_params.IMG_CHANNELS

  # pack scene context into pseudo-batch for encoder
  context_poses_packed = tf.reshape(context_poses, shape=[-1, _DIM_POSE])
  context_frames_packed = tf.reshape(
      context_frames, shape=[-1, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])

  return context_poses_packed, context_frames_packed


def _reduce_packed_representation(enc_r_packed, model_params):
  # shorthand notations for model parameters
  _CONTEXT_SIZE = model_params.CONTEXT_SIZE
  _DIM_C_ENC = model_params.ENC_CHANNELS

  height, width = tf.shape(enc_r_packed)[1], tf.shape(enc_r_packed)[2]

  enc_r_unpacked = tf.reshape(
      enc_r_packed, shape=[-1, _CONTEXT_SIZE, height, width, _DIM_C_ENC])

  # add scene representations per data tuple
  enc_r = tf.reduce_sum(enc_r_unpacked, axis=1)

  return enc_r


def _encode_context(encoder_fn, context_poses, context_frames, model_params):
  endpoints = {}

  context_poses_packed, context_frames_packed = _pack_context(
      context_poses, context_frames, model_params)

  # define scene encoding graph psi
  enc_r_packed, endpoints_psi = encoder_fn(context_frames_packed,
                                           context_poses_packed)
  endpoints.update(endpoints_psi)
  # unpack scene encoding and reduce to single vector
  enc_r = _reduce_packed_representation(enc_r_packed, model_params)
  endpoints["enc_r"] = enc_r

  return enc_r, endpoints

def patch_image(frames: tf.Tensor, poses: tf.Tensor):
    batch_size=GQN_DEFAULT_CONFIG.BATCH_SIZE
    img_h = GQN_DEFAULT_CONFIG.IMG_HEIGHT
    img_w = GQN_DEFAULT_CONFIG.IMG_WIDTH

    empty = np.array(empty)# 1 8 8 2
    new_poses = tf.convert_to_tensor(empty, dtype=tf.float32)
    # frames = tf.reshape(frames, [batch_size,img_h,img_w,3])
    patches=tf.extract_image_patches(images=frames, ksizes=[1,8,8,1], strides=[1,4,4,1],rates=[1,1,1,1], padding="SAME")
    # 64 x 20 x batchsize
    patches = tf.reshape(patches, [-1,64,8,8,3])

    temp = []
    for i in range(64):
        ttt=[]
        for j in range(8):
            tt=[]
            for k in range(8):
                t = [i//8+1, i%8+1]
                tt.append(t)
            ttt.append(tt)
        temp.append(ttt)

    empty=[]
    empty.append(temp)
    empty = np.array(empty)
    new_poses = tf.convert_to_tensor(empty, dtype=tf.float32)
    patches = tf.concat([patches, new_poses],axis=4)
    patches = tf.reshape(patches, [-1,8,8,5])

    # embedding pos to patch
    net = tf.layers.conv2d(patches, filters=32, kernel_size=1, strides=1,
                          padding="SAME", activation=tf.nn.relu)

    skip1 = tf.layers.conv2d(net, filters=32, kernel_size=1, strides=1,
                            padding="SAME", activation=None)
    net = tf.layers.conv2d(net, filters=32, kernel_size=2, strides=1,
                          padding="SAME", activation=tf.nn.relu)

    net = net + skip1
    net = tf.layers.conv2d(net, filters=32, kernel_size=2, strides=1, padding="SAME", activation=tf.nn.relu)
    # patches now 1280(10) x 8 x 8 x 64

    # tile the poses to match the embedding shape
    poses = tf.reshape(poses, [-1,1,1,1,7]) # 20(36) x1x 1 x 1 x 7
    # print(poses.get_shape())
    poses = tf.tile(poses, [1,64,8,8,1]) # 1280(10) x 8 x 8 x 7
    poses = tf.reshape(poses, [-1,8,8,7]) # 1280(10) x8 x 8 x 7

    # concatenate the poses with the embedding
    net = tf.concat([net, poses], axis=3) # 1280 x 8 x 8 x 11

    skip2 = tf.layers.conv2d(net, filters=64, kernel_size=1, strides=1,
                            padding="SAME", activation=None)
    net = tf.layers.conv2d(net, filters=64, kernel_size=2, strides=1,
                          padding="SAME", activation=tf.nn.relu)
    net = net + skip2
    net = tf.layers.conv2d(net, filters=64, kernel_size=1, strides=1,
                          padding="SAME", activation=tf.nn.relu)
    net = tf.layers.conv2d(net, filters=64, kernel_size=1, strides=1,
                          padding="SAME", activation=tf.nn.relu)

    return net


def gqn_draw(
    query_pose: tf.Tensor, target_frame: tf.Tensor,
    context_poses: tf.Tensor, context_frames: tf.Tensor,
    model_params: GQNConfig, is_training: bool = True,
    scope: str = "GQN"):
  """
  Defines the computational graph of the GQN model.
  Arguments:
    query_pose: Pose vector of the query camera.
    target_frame: Ground truth frame of the query camera. Used in training mode
        by the inference LSTM.
    context_poses: Camera poses of the context views.
    context_frames: Frames of the context views.
    model_params: Named tuple containing the parameters of the GQN model as \
      defined in gqn_params.py
    is_training: Flag whether graph shall be created in training mode (including \
      the inference module necessary for training the generator). If set to 'False',
      only the generator LSTM will be created.
    scope: Scope name of the graph.
  Returns:
    net: The last tensor of the network.
    endpoints: A dictionary providing quick access to the most important model
      nodes in the computational graph.
  """
  # shorthand notations for model parameters
  _ENC_TYPE = model_params.ENC_TYPE
  _DIM_H_ENC = model_params.ENC_HEIGHT
  _DIM_W_ENC = model_params.ENC_WIDTH
  _DIM_C_ENC = model_params.ENC_CHANNELS
  _SEQ_LENGTH = model_params.SEQ_LENGTH

  with tf.variable_scope(scope):
    endpoints = {}

    enc_r, endpoints_enc = _encode_context(
        _ENC_FUNCTIONS[_ENC_TYPE], context_poses, context_frames, model_params)
    endpoints.update(endpoints_enc)


    enc_r_broadcast = tf.reshape(enc_r, [-1, _DIM_H_ENC, _DIM_W_ENC, _DIM_C_ENC])
    context_poses_packed, context_frames_packed= _pack_context(
        context_poses, context_frames, model_params)
    # print(context_frames_packed.get_shape()) ? 32 32 3
    # print(context_poses_packed.get_shape()) ? 7
    # print(enc_r_broadcast.get_shape()) ? 8 8 64
    # 1280(36) x 8 x 8 x 64

    patch_dic=patch_image(context_frames_packed, context_poses_packed)

    if is_training:
      mu_target, endpoints_rnn = inference_rnn(
        patch_dic=patch_dic,
        context_frames=context_frames_packed,
        context_poses=context_poses_packed,
        encoder_packed=enc_r_broadcast,
        query_poses=query_pose,
        target_frames=target_frame,
        sequence_size=_SEQ_LENGTH,
      )
    else:
      mu_target, endpoints_rnn = generator_rnn(
          patch_dic=patch_dic,
          encoder_packed=enc_r_broadcast,
          # representations=enc_r_broadcast,
          query_poses=query_pose,
          sequence_size=_SEQ_LENGTH
      )

    endpoints.update(endpoints_rnn)
    net = mu_target  # final mu tensor parameterizing target frame sampling
    return net, endpoints


def gqn_vae(
    query_pose: tf.Tensor,
    context_poses: tf.Tensor, context_frames: tf.Tensor,
    model_params: GQNConfig, scope: str = "GQN-VAE"):
  """
  Defines the computational graph of the GQN-VAE baseline model.
  Arguments:
    query_pose: Pose vector of the query camera.
    context_poses: Camera poses of the context views.
    context_frames: Frames of the context views.
    model_params: Named tuple containing the parameters of the GQN model as \
      defined in gqn_params.py
    scope: Scope name of the graph.
  Returns:
    net: The last tensor of the network.
    endpoints: A dictionary providing quick access to the most important model
      nodes in the computational graph.
  """
  with tf.variable_scope(scope):
    endpoints = {}

    enc_r, endpoints_enc = _encode_context(
        tower_encoder, context_poses, context_frames, model_params)
    endpoints.update(endpoints_enc)

    mu_z, sigma_z, z = compute_eta_and_sample_z(
        enc_r, channels=model_params.Z_CHANNELS, scope="Sample_eta")
    endpoints['mu_q'] = mu_z
    endpoints['sigma_q'] = sigma_z

    mu_target, decoder_ep = vae_tower_decoder(z, query_pose)
    endpoints.update(decoder_ep)

    net = mu_target  # final mu tensor parameterizing target frame sampling
    return net, endpoints
