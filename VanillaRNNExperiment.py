import tensorflow as tf

from Model import Model
from util import lazyproperty


class ExperimentModel(Model):
    def build_graph(self):
        one_hot_char_in = tf.one_hot(indices=self.char_in, depth=self.vocab_size, dtype=tf.float32)

        self._rnn_cell = tf.contrib.rnn.BasicRNNCell(128)
        self._rnn_sequence_length = tf.placeholder_with_default(
            tf.tile(tf.ones([1]), multiples=[self.batch_size]) * self.max_timesteps, shape=[None])
        self._rnn_initial_state_placeholder = tf.placeholder_with_default(self.rnn_zero_state,
                                                                          shape=[None, self._rnn_cell.state_size])
        rnn_out, self._rnn_final_state = tf.nn.dynamic_rnn(self._rnn_cell, inputs=one_hot_char_in,
                                                           sequence_length=self.rnn_sequence_length,
                                                           initial_state=self.rnn_initial_state_placeholder)

        self._char_out_log_probs = tf.layers.dense(rnn_out, units=len(self.vocab))
        self._char_out_probs = tf.nn.softmax(self._char_out_log_probs)
        self._char_out_max = tf.argmax(self._char_out_log_probs, axis=2)

    @lazyproperty
    def rnn_initial_state_placeholder(self):
        return self._rnn_initial_state_placeholder

    @lazyproperty
    def rnn_zero_state(self):
        return self._rnn_cell.zero_state(self.batch_size, tf.float32)

    @lazyproperty
    def rnn_sequence_length(self):
        return self._rnn_sequence_length

    @lazyproperty
    def rnn_final_state(self):
        return self._rnn_final_state

    @lazyproperty
    def char_out_probs(self):
        return self._char_out_probs

    @lazyproperty
    def char_out_max(self):
        return self._char_out_max

    @lazyproperty
    def loss(self):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.char_target, logits=self._char_out_log_probs))

    @lazyproperty
    def train_op(self):
        grads = tf.gradients(self.loss, self.trainable_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        return optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables), global_step=self.global_step)


if __name__ == '__main__':
    import os
    from datasets.FullShakespeareDataSet import FullShakespeareDataSet

    # Load dataset
    dataset = FullShakespeareDataSet()

    with tf.Session() as sess:
        model_dir = 'models/{}/'.format(os.path.splitext(os.path.basename(__file__))[0])
        global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

        max_timesteps = 25
        model = ExperimentModel(sess=sess, vocab=dataset.vocab, max_timesteps=max_timesteps, filepath=model_dir)
        sess.run(tf.global_variables_initializer())
        model.maybe_restore()
        global_step = sess.run(global_step_tensor)

        # Train
        num_epochs = 1000
        steps_per_epoch = 100
        total_steps = num_epochs * steps_per_epoch
        for epoch_num in range(num_epochs):
            for iteration in range(steps_per_epoch):
                if global_step % 100 == 0:
                    primer = dataset.get_primer(length=10)
                    print('Step {}/{}: {}'.format(global_step, total_steps, repr(''.join(primer[0]) + '|' + ''.join(
                        model.sample(primer=primer, sample_size=150)))))
                    model.validate(*dataset.get_validation_samples(batch_size=100, max_timesteps=model.max_timesteps))
                if global_step % 1000 == 0:
                    model.save()
                model.train(*dataset.get_training_samples(epoch=epoch_num, batch_size=100,
                                                          max_timesteps=model.max_timesteps),
                            record_summary=global_step % 100)
                global_step = sess.run(global_step_tensor)

                if global_step > total_steps:
                    break
            if global_step > total_steps:
                break

        # Sample
        print('Training complete after {} global steps! Sampling...'.format(total_steps))
        primer = dataset.get_primer(length=20)
        print(''.join(primer[0]) + '|' + ''.join(model.sample(sample_size=1000, primer=primer)))
