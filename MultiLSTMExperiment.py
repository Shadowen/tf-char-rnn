import numpy as np
import tensorflow as tf

from Model import Model
from util import lazy_property


class ExperimentModel(Model):
    def build_graph(self):
        one_hot_char_in = tf.one_hot(indices=self.char_in, depth=self.vocab_size, dtype=tf.float32)

        self._dropout_keep_prob = tf.placeholder_with_default(1.0, shape=[], name='dropout_keep_prob')
        single_cell = lambda: tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(512),
                                                            output_keep_prob=self._dropout_keep_prob)
        self._rnn_cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(3)])
        rnn_out, self._rnn_final_state = tf.nn.dynamic_rnn(self._rnn_cell, inputs=one_hot_char_in,
                                                           sequence_length=self.rnn_sequence_length,
                                                           initial_state=self.rnn_initial_state_placeholder)

        self._char_out_log_probs = tf.layers.dense(rnn_out, units=len(self.vocab))
        self._char_out_probs = tf.nn.softmax(self._char_out_log_probs)
        self._char_out_max = tf.argmax(self._char_out_log_probs, axis=2)

    @property
    def dropout_keep_prob(self):
        return self._dropout_keep_prob

    @lazy_property
    def rnn_initial_state_placeholder(self):
        return tuple(
            tf.contrib.rnn.LSTMStateTuple(
                tf.placeholder_with_default(self.rnn_zero_state[i].c, shape=[None, self._rnn_cell.state_size[i].c]),
                tf.placeholder_with_default(self.rnn_zero_state[i].h, shape=[None, self._rnn_cell.state_size[i].h])
            ) for i in range(len(self._rnn_cell._cells))
        )

    @lazy_property
    def rnn_zero_state(self):
        return self._rnn_cell.zero_state(self.batch_size, tf.float32)

    @lazy_property
    def rnn_sequence_length(self):
        return tf.placeholder_with_default(tf.tile(tf.ones([1]), multiples=[self.batch_size]) * self.max_timesteps,
                                           shape=[None])

    @lazy_property
    def rnn_final_state(self):
        return self._rnn_final_state

    @lazy_property
    def char_out_probs(self):
        return self._char_out_probs

    @lazy_property
    def _char_out_max(self):
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

    def train(self, x, t):
        batch_size = x.shape[0]
        x_embedding = np.zeros([batch_size, self.max_timesteps])
        t_embedding = np.zeros([batch_size, self.max_timesteps])
        for b in range(batch_size):
            for i in range(self.max_timesteps):
                x_embedding[b, i] = self.char_to_ix[x[b, i]]
                t_embedding[b, i] = self.char_to_ix[t[b, i]]
        self.sess.run(self.train_op,
                      feed_dict={self.char_in: x_embedding, self.char_target: t_embedding, self.dropout_keep_prob: 0.5})


if __name__ == '__main__':
    import os
    from datasets.FullShakespeareDataSet import FullShakespeareDataSet

    # Load datasets
    dataset = FullShakespeareDataSet()

    with tf.Session() as sess:
        model_dir = 'models/{}/'.format(os.path.splitext(os.path.basename(__file__))[0])
        global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

        max_timesteps = 50
        model = ExperimentModel(sess=sess, vocab=dataset.vocab, max_timesteps=max_timesteps)
        sess.run(tf.global_variables_initializer())
        if os.path.exists(model_dir):
            model.restore_pretrained(model_dir)
            global_step_start = sess.run(global_step_tensor)
            print('Restored model from {} at step {}'.format(model_dir, global_step_start))
        else:
            os.makedirs(model_dir)
            global_step_start = sess.run(global_step_tensor)

        # Train
        total_steps = 20000
        for step in range(global_step_start, total_steps):
            if step % 100 == 0:
                primer = dataset.get_primer(length=10)
                print(
                    'Step {}/{}: {}'.format(step, total_steps,
                                            repr(''.join(primer[0]) + '|' + ''.join(
                                                model.sample(primer=primer, sample_size=150)))))
                model.save(model_dir)
            model.train(**dataset.get_training_samples(batch_size=100, max_timesteps=model.max_timesteps))

        # Sample
        print('Training complete after {}! Sampling...'.format(total_steps))
        primer = dataset.get_primer(length=10)
        print(''.join(primer[0]) + '|' + ''.join(model.sample(sample_size=1000, primer=primer)))
