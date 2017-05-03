from abc import abstractmethod, abstractproperty

import numpy as np
import tensorflow as tf

from util import lazy_property


class Model:
    def __init__(self, sess, vocab, max_timesteps):
        self.sess = sess
        self.global_step = tf.train.get_global_step()
        assert self.global_step is not None

        self.vocab = vocab
        self.char_to_ix = {ch: i for i, ch in enumerate(vocab)}
        self.ix_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.max_timesteps = max_timesteps

        self.scope_name = 'RNN_estimator'
        with tf.variable_scope(self.scope_name):
            # Inputs
            self.char_in = tf.placeholder(dtype=tf.int32, shape=[None, max_timesteps])
            self.char_target = tf.placeholder(dtype=tf.int32, shape=[None, max_timesteps])

            # Model
            self.batch_size = tf.shape(self.char_in)[0]
            self.build_graph()
            # Make sure everything exists
            for p in [self.rnn_initial_state_placeholder, self.rnn_zero_state, self.rnn_sequence_length,
                      self.rnn_final_state, self.char_out_probs, self._char_out_max, self.trainable_variables,
                      self.train_op]:
                if p is None:
                    raise NotImplementedError()

    @abstractmethod
    def build_graph(self):
        pass

    @abstractproperty
    def rnn_initial_state_placeholder(self):
        pass

    @abstractproperty
    def rnn_zero_state(self):
        pass

    @abstractproperty
    def rnn_sequence_length(self):
        pass

    @abstractproperty
    def rnn_final_state(self):
        pass

    @abstractproperty
    def char_out_probs(self):
        pass

    @abstractproperty
    def _char_out_max(self):
        pass

    @abstractproperty
    def loss(self):
        pass

    @lazy_property
    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)

    @abstractproperty
    def train_op(self):
        grads = tf.gradients(self.loss, self.trainable_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        return optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables),
                                         global_step=self.global_step)

    @lazy_property
    def saver(self):
        saver = tf.train.Saver(var_list=self.trainable_variables + [self.global_step], max_to_keep=2,
                               keep_checkpoint_every_n_hours=12)
        return saver

    def restore_pretrained(self, filepath):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(filepath))

    def save(self, filepath):
        self.saver.save(self.sess, filepath + 'model', global_step=self.global_step)

    def train(self, x, t):
        batch_size = x.shape[0]
        x_embedding = np.zeros([batch_size, self.max_timesteps])
        t_embedding = np.zeros([batch_size, self.max_timesteps])
        for b in range(batch_size):
            for i in range(self.max_timesteps):
                x_embedding[b, i] = self.char_to_ix[x[b, i]]
                t_embedding[b, i] = self.char_to_ix[t[b, i]]
        self.sess.run(self.train_op, feed_dict={self.char_in: x_embedding, self.char_target: t_embedding})

    def sample(self, sample_size, primer, temperature=1):
        actual_output = []

        # Prime...
        primer_duration = np.ones([1]) * primer.shape[1]
        x = np.zeros([1, self.max_timesteps])
        for i, p in enumerate(primer[0]):
            x[0, i] = self.char_to_ix[p]
        if temperature == 0:
            output, s = self.sess.run([self._char_out_max, self.rnn_final_state],
                                      feed_dict={self.char_in: x, self.rnn_sequence_length: primer_duration})
            prev_o = output[0][-1]
        else:
            probs, s = self.sess.run([self._char_out_probs, self.rnn_final_state],
                                     feed_dict={self.char_in: x, self.rnn_sequence_length: primer_duration})
            prev_o = np.random.choice(np.arange(self.vocab_size), p=probs[0][0])

        # Sample!
        for i in range(sample_size):
            x = np.zeros([1, self.max_timesteps])
            x[0, 0] = prev_o
            if temperature == 0:
                output, s = self.sess.run([self._char_out_max, self.rnn_final_state],
                                          feed_dict={self.char_in: x, self.rnn_initial_state_placeholder: s,
                                                     self.rnn_sequence_length: np.ones([1])})
                o = output[0][0]
            else:
                probs, s = self.sess.run([self._char_out_probs, self.rnn_final_state],
                                         feed_dict={self.char_in: x, self.rnn_initial_state_placeholder: s,
                                                    self.rnn_sequence_length: np.ones([1])})
                o = np.random.choice(np.arange(self.vocab_size), p=probs[0][0])
            actual_output.append(self.ix_to_char[o])
            prev_o = o

        return actual_output


if __name__ == '__main__':
    # Load datasets
    data = []
    with open('datasets/tiny-shakespeare.txt', 'r') as f:
        for _ in range(1):
            data.extend(list(f.readline()))

    print("Data loaded: {} characters".format(len(data)))

    with tf.Session() as sess:
        global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        max_timesteps = 8
        model = Model(sess=sess, vocab=list(set(data)), max_timesteps=max_timesteps)
        sess.run(tf.global_variables_initializer())

        # Train
        for step in range(200):
            start_idx = np.random.randint(len(data) - model.max_timesteps)
            model.train(x=data[start_idx:start_idx + model.max_timesteps],
                        t=data[start_idx + 1:start_idx + model.max_timesteps + 1])

        # Internal test - Ensure overfitting
        sample_size = 10
        assert model.sample(primer=[data[0]], sample_size=sample_size, temperature=0) == data[1:sample_size + 1]
        assert model.sample(primer=[data[0]], sample_size=sample_size, temperature=0) == \
               model.sample(primer=[data[0]], sample_size=sample_size, temperature=1e-1000)
