import tensorflow as tf

from gqn.gqn_params import GQN_DEFAULT_CONFIG as PARAMS
from gqn.gqn_attention import gqn_draw
from data_provider.gqn_tfr_provider import gqn_input_fn

import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = '/media/cylee/StorageDisk/gqn-dataset/'
# dataset = 'minecraft'
dataset = 'mazes'
# model_dir = '/home/cylee/gqn/models/tf-gqn/minecraft/'
model_dir = '/home/cylee/gqn/models/tf-gqn/maze_attention_fix/'
result_dir = './results/maze_attention_fix/'
train = 0

import sys
if len(sys.argv) > 1:
    iters = sys.argv[1]
else:
    iters = None

mode = tf.estimator.ModeKeys.TRAIN if train else tf.estimator.ModeKeys.PREDICT

example = gqn_input_fn(
    dataset=dataset,
    context_size=20,
    batch_size=13,
    custom_frame_size=32,
    root=data_dir,
    mode=mode
)

# graph definition in test mode
net, ep_gqn = gqn_draw(
    query_pose=example[0].query_camera,
    target_frame=example[1],
    context_poses=example[0].context.cameras,
    context_frames=example[0].context.frames,
    model_params=PARAMS,
    is_training=False
)

saver = tf.train.Saver()
sess = tf.Session()

# Don't run initalisers, restore variables instead
# sess.run(tf.global_variables_initializer())
if iters is None:
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    iters = latest_checkpoint.split("-")[-1]
else:
    latest_checkpoint = os.path.join(model_dir, '/model.ckpt-' + iters)
saver.restore(sess, latest_checkpoint)
print("iters = ", iters)
print(example[1])
# Run network forward, shouldn't complain about uninitialised variables
output, output_gt = sess.run([net, example[1]])

print(output.shape)
print(output_gt.shape)

outdir = os.path.join(result_dir, dataset,'mazes', 'train/' if train else '', iters)
# outdir = os.path.join(result_dir, dataset, 'minecraft', 'train/' if train else '', iters)

if not os.path.exists(outdir):
    os.makedirs(outdir)

for i in range(len(output)):
    plt.imsave('{}/{}_out.png'.format(outdir, i), output[i])
    plt.imsave('{}/{}_gt.png'.format(outdir, i), output_gt[i])
