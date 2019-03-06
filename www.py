import tensorflow as tf
from gqn.gqn_model import gqn_draw_model_fn
from gqn.gqn_params import GQN_DEFAULT_CONFIG, create_gqn_config
from data_provider.gqn_tfr_provider import gqn_input_fn
import matplotlib.pyplot as plt
import numpy as np

MODEL_DIR='/home/cylee/gqn/models'
DATA_DIR='/home/cylee/gqn/gqn-dataset'
DATASET='rooms_ring_camera'

#params = create_gqn_config()
params = GQN_DEFAULT_CONFIG

estimator = tf.estimator.Estimator(
    model_fn=gqn_draw_model_fn,
    model_dir=MODEL_DIR,
    params={'gqn_params' : params,  'debug' : False})

input_fn = lambda mode: gqn_input_fn(
        dataset=DATASET,
        #context_size=params['CONTEXT_SIZE'],
        context_size=params.CONTEXT_SIZE,
        root=DATA_DIR,
        mode=mode)

for prediction in estimator.predict(input_fn=input_fn):
    # prediction is the dict @ogroth was mentioning
    print(prediction['predicted_mean'])  # this is probably what you want to look at
    print(prediction['predicted_variance'])  # or use this to sample a noisy image
    a = prediction['predicted_mean']
    print(type(a))
    print(a.shape)
    plt.imsave('fig.png', a)
    break


