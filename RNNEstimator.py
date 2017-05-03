import numpy as np
import tensorflow as tf

from util import cachedProperty


class RNNEstimator():
    def __init__(self, sess, vocab, max_timesteps):
        self.sess = sess
        self.global_step = tf.train.get_global_step()
        assert self.global_step is not None

        self.vocab = vocab
        self.char_to_ix = {ch: i for i, ch in enumerate(vocab)}
        self.ix_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.max_timesteps = max_timesteps

        scope_name = 'RNN_estimator'
        with tf.variable_scope(scope_name):
            # Inputs
            self.char_in = tf.placeholder(dtype=tf.int32, shape=[None, max_timesteps])
            self.char_target = tf.placeholder(dtype=tf.int32, shape=[None, max_timesteps])

            # Model
            batch_size = tf.shape(self.char_in)[0]
            self.one_hot_char_in = tf.one_hot(indices=self.char_in, depth=self.vocab_size, dtype=tf.float32)
            rnn_cell = tf.contrib.rnn.BasicRNNCell(128)
            self.rnn_zero_state = rnn_cell.zero_state(1, tf.float32)
            self.rnn_initial_state = tf.placeholder_with_default(self.rnn_zero_state,
                                                                 shape=[None, rnn_cell.state_size])
            self.rnn_sequence_length = tf.placeholder_with_default(
                tf.tile(tf.ones([1]), multiples=[batch_size]) * self.max_timesteps,
                shape=[None])
            rnn_out, self.rnn_final_state = tf.nn.dynamic_rnn(rnn_cell, inputs=self.one_hot_char_in,
                                                              sequence_length=self.rnn_sequence_length,
                                                              initial_state=self.rnn_initial_state)
            char_out_log_probs = tf.layers.dense(rnn_out, units=len(vocab))
            self.char_out_probs = tf.nn.softmax(char_out_log_probs)
            self.char_out_max = tf.argmax(char_out_log_probs, axis=2)

            # Loss
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.char_target, logits=char_out_log_probs))

            # Optimizer
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
            grads = tf.gradients(self.loss, self.trainable_variables)
            clipped_gradients, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.train_op = optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables),
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
        x_embedding = np.zeros([1, self.max_timesteps])
        t_embedding = np.zeros([1, self.max_timesteps])
        for i in range(self.max_timesteps):
            x_embedding[0, i] = self.char_to_ix[x[i]]
            t_embedding[0, i] = self.char_to_ix[t[i]]
        self.sess.run(self.train_op, feed_dict={self.char_in: x_embedding, self.char_target: t_embedding})

    def sample(self, primer, sample_size, temperature=1):
        actual_output = []

        prev_o = self.char_to_ix[primer]
        s = self.sess.run(self.rnn_zero_state)
        for i in range(sample_size):
            x = np.zeros([1, self.max_timesteps])
            x[0, 0] = prev_o
            if temperature == 0:
                output, s = self.sess.run([self.char_out_max, self.rnn_final_state],
                                          feed_dict={self.char_in: x, self.rnn_initial_state: s,
                                                     self.rnn_sequence_length: np.ones([1])})
                o = output[0][0]
            else:
                probs, s = self.sess.run([self.char_out_probs, self.rnn_final_state],
                                         feed_dict={self.char_in: x, self.rnn_initial_state: s,
                                                    self.rnn_sequence_length: np.ones([1])})
                o = np.random.choice(np.arange(self.vocab_size), p=probs[0][0])
            actual_output.append(self.ix_to_char[o])
            prev_o = o
        return actual_output


if __name__ == '__main__':
    # Load data
    data = []
    with open('data/tiny-shakespeare.txt', 'r') as f:
        for _ in range(1):
            data.extend(list(f.readline()))

    print("Data loaded: {} characters".format(len(data)))

    with tf.Session() as sess:
        global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        max_timesteps = 8
        model = RNNEstimator(sess=sess, vocab=list(set(data)), max_timesteps=max_timesteps)
        sess.run(tf.global_variables_initializer())

        # Train
        for step in range(200):
            start_idx = np.random.randint(len(data) - model.max_timesteps)
            model.train(x=data[start_idx:start_idx + model.max_timesteps],
                        t=data[start_idx + 1:start_idx + model.max_timesteps + 1])

        # Internal test - Ensure overfitting
        sample_size = 10
        assert model.sample(primer=data[0], sample_size=sample_size, temperature=0) == data[1:sample_size + 1]
        assert model.sample(primer=data[0], sample_size=sample_size, temperature=0) == \
               model.sample(primer=data[0], sample_size=sample_size, temperature=1e-1000)
