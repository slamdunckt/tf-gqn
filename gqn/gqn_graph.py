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

from .gqn_params import GQNConfig
from .gqn_encoder import tower_encoder, pool_encoder
# from .gqn_draw import inference_rnn, generator_rnn
from .gqn_draw_attention import inference_rnn, generator_rnn
from .gqn_utils import broadcast_encoding, compute_eta_and_sample_z
from .gqn_vae import vae_tower_decoder


_ENC_FUNCTIONS = {
    'pool' : pool_encoder,
    'tower' : tower_encoder,
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



def _attention_unpacked_representation(enc_r_extracted, model_params,query_pose):
  _CONTEXT_SIZE = model_params.CONTEXT_SIZE

  query_pose_reshape = tf.reshape(
    query_pose, shape=[-1,1,1,1,7])
  query_pose_dense = tf.layers.dense(query_pose_reshape,units=64,activation=tf.nn.relu,name='pose_dense1')
  query_pose_dense = tf.layers.dense(query_pose_dense,units=256,activation=tf.nn.relu,name='pose_dense2')
  query_pose_dup = tf.tile(
    query_pose_dense, [1,_CONTEXT_SIZE,1,1,1])
  enc_r_combine = tf.matmul(enc_r_extracted,query_pose_dup,transpose_b=True)
  attention_softmax = tf.nn.softmax(enc_r_combine,axis=1)
  #enc_r_combine = tf.concat((enc_r_extracted,query_pose_dup),axis=-1)
  #dense1 = tf.layers.dense(enc_r_combine, units=64, activation=tf.nn.relu, name='attn_dense1')
  #dense2 = tf.layers.dense(dense1, units=8, activation=tf.nn.relu, name='attn_dense2')
  #dense3 = tf.layers.dense(dense2, units=1, name='attn_dense3')
  #attention_softmax = tf.nn.softmax(dense3,axis=1)
  return attention_softmax

def _reduce_packed_representation(enc_r_packed, model_params,query_pose):
  # shorthand notations for model parameters
  _CONTEXT_SIZE = model_params.CONTEXT_SIZE
  _DIM_C_ENC = model_params.ENC_CHANNELS
  _ENC_TYPE = model_params.ENC_TYPE

  height, width = tf.shape(enc_r_packed)[1], tf.shape(enc_r_packed)[2]

  enc_r_unpacked = tf.reshape(
      enc_r_packed, shape=[-1, _CONTEXT_SIZE, height, width, _DIM_C_ENC])

  #apply attention
  if _ENC_TYPE == 'tower':
    enc_r_pooling = tf.layers.average_pooling2d(enc_r_packed, 16, 1, name='pooling')
    enc_r_extracted = tf.reshape(
      enc_r_pooling, shape=[-1,_CONTEXT_SIZE,1,1,_DIM_C_ENC])
  elif _ENC_TYPE == 'pool':
    enc_r_extracted = enc_r_unpacked
  enc_attention = _attention_unpacked_representation(enc_r_extracted, model_params,query_pose)

  # add scene representations per data tuple with attention
  enc_r = tf.reduce_sum(enc_attention*enc_r_unpacked, axis=1)

  return enc_r


def _encode_context(encoder_fn, context_poses, context_frames, model_params,query_pose):
  endpoints = {}

  context_poses_packed, context_frames_packed = _pack_context(
      context_poses, context_frames, model_params)

  # define scene encoding graph psi
  enc_r_packed, endpoints_psi = encoder_fn(context_frames_packed,
                                           context_poses_packed)
  endpoints.update(endpoints_psi)

  # unpack scene encoding and reduce to single vector
  enc_r = _reduce_packed_representation(enc_r_packed, model_params,query_pose)
  endpoints["enc_r"] = enc_r

  return enc_r, endpoints


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
        _ENC_FUNCTIONS[_ENC_TYPE], context_poses, context_frames, model_params,query_pose)
    endpoints.update(endpoints_enc)

    # broadcast scene representation to 1/4 of targeted frame size
    if _ENC_TYPE == 'pool':
      enc_r_broadcast = broadcast_encoding(
          vector=enc_r, height=_DIM_H_ENC, width=_DIM_W_ENC)
    else:
      enc_r_broadcast = tf.reshape(enc_r, [-1, _DIM_H_ENC, _DIM_W_ENC, _DIM_C_ENC])

    # define generator graph (with inference component if in training mode)
    if is_training:
      mu_target, endpoints_rnn = inference_rnn(
          representations=enc_r_broadcast,
          query_poses=query_pose,
          target_frames=target_frame,
          sequence_size=_SEQ_LENGTH,
      )
    else:
      mu_target, endpoints_rnn = generator_rnn(
          representations=enc_r_broadcast,
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
        tower_encoder, context_poses, context_frames, model_params,query_pose)
    endpoints.update(endpoints_enc)

    mu_z, sigma_z, z = compute_eta_and_sample_z(
        enc_r, channels=model_params.Z_CHANNELS, scope="Sample_eta")
    endpoints['mu_q'] = mu_z
    endpoints['sigma_q'] = sigma_z

    mu_target, decoder_ep = vae_tower_decoder(z, query_pose)
    endpoints.update(decoder_ep)

    net = mu_target  # final mu tensor parameterizing target frame sampling
    return net, endpoints
