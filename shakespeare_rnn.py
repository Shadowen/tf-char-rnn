import os

import numpy as np
import tensorflow as tf

from RNNEstimator import RNNEstimator
from data.TinyShakespeareDataSet import TinyShakespeareDataSet

# Load data
dataset = TinyShakespeareDataSet()

with tf.Session() as sess:
    model_dir = 'models/shakespeare_rnn/'
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    max_timesteps = 25
    model = RNNEstimator(sess=sess, vocab=dataset.vocab, max_timesteps=max_timesteps)
    sess.run(tf.global_variables_initializer())
    if os.path.exists(model_dir):
        model.restore_pretrained(model_dir)
        global_step = sess.run(global_step_tensor)
        print('Restored model from {} at step {}'.format(model_dir, global_step))
    else:
        os.makedirs(model_dir)
        global_step = sess.run(global_step_tensor)

    # Train
    total_steps = 100000
    for step in range(global_step, total_steps):
        if step % 1000 == 0:
            primer = np.random.choice(dataset.vocab)
            print(
                'Step {}/{}: {}'.format(step, total_steps,
                                        repr(primer + '|' + ''.join(model.sample(primer=primer, sample_size=150)))))
            model.save(model_dir)
        model.train(**dataset.get_training_samples(max_timesteps=model.max_timesteps))
