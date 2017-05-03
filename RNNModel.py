import tensorflow as tf

from Model import Model
from util import lazy_property


class RNNModel(Model):
    def build_graph(self):
        one_hot_char_in = tf.one_hot(indices=self.char_in, depth=self.vocab_size, dtype=tf.float32)

        self._rnn_cell = tf.contrib.rnn.BasicRNNCell(128)
        self._rnn_sequence_length = tf.placeholder_with_default(
            tf.tile(tf.ones([1]), multiples=[self.batch_size]) * self.max_timesteps, shape=[None])
        rnn_out, self._rnn_final_state = tf.nn.dynamic_rnn(self._rnn_cell, inputs=one_hot_char_in,
                                                           sequence_length=self.rnn_sequence_length,
                                                           initial_state=self.rnn_initial_state_placeholder)

        self._char_out_log_probs = tf.layers.dense(rnn_out, units=len(self.vocab))
        self._char_out_probs = tf.nn.softmax(self._char_out_log_probs)
        self._char_out_max = tf.argmax(self._char_out_log_probs, axis=2)

    @lazy_property
    def rnn_initial_state_placeholder(self):
        return tf.placeholder_with_default(self.rnn_zero_state,
                                           shape=[None, self._rnn_cell.state_size])

    @lazy_property
    def rnn_zero_state(self):
        return self._rnn_cell.zero_state(1, tf.float32)

    @lazy_property
    def rnn_sequence_length(self):
        return self._rnn_sequence_length

    @lazy_property
    def rnn_final_state(self):
        return self._rnn_final_state

    @lazy_property
    def char_out_probs(self):
        return self._char_out_probs

    @lazy_property
    def char_out_max(self):
        return self._char_out_max

    @lazy_property
    def loss(self):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.char_target, logits=self._char_out_log_probs))

    @lazy_property
    def train_op(self):
        grads = tf.gradients(self.loss, self.trainable_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        return optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables), global_step=self.global_step)


if __name__ == '__main__':
    import os

    import numpy as np

    from RNNModel import RNNModel
    from data.TinyShakespeareDataSet import TinyShakespeareDataSet

    # Load data
    dataset = TinyShakespeareDataSet()

    with tf.Session() as sess:
        model_dir = 'models/{}/'.format(os.path.splitext(os.path.basename(__file__))[0])
        global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

        max_timesteps = 25
        model = RNNModel(sess=sess, vocab=dataset.vocab, max_timesteps=max_timesteps)
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
