import tensorflow as tf
import numpy as np

# Load data
data = []
with open('data/tiny-shakespeare.txt', 'r') as f:
    for _ in range(1):
        data.extend(list(f.readline()))

print("Data loaded: {} characters".format(len(data)))

chars = list(set(data))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Create model
max_timesteps = len(data) - 1
# Inputs
char_in = tf.placeholder(dtype=tf.int32, shape=[None, max_timesteps])
char_target = tf.placeholder(dtype=tf.int32, shape=[None, max_timesteps])
# Model
batch_size = tf.shape(char_in)[0]
one_hot_char_in = tf.one_hot(indices=char_in, depth=len(chars), dtype=tf.float32)
rnn_cell = tf.contrib.rnn.BasicRNNCell(128)
rnn_zero_state = rnn_cell.zero_state(1, tf.float32)
rnn_initial_state = tf.placeholder_with_default(rnn_zero_state,
                                                shape=[None, rnn_cell.state_size])
rnn_sequence_length = tf.placeholder_with_default(tf.tile(tf.ones([1]), multiples=[batch_size]) * max_timesteps,
                                                  shape=[None])
rnn_out, rnn_final_state = tf.nn.dynamic_rnn(rnn_cell, inputs=one_hot_char_in, sequence_length=rnn_sequence_length,
                                             initial_state=rnn_initial_state)
char_out_probs = tf.layers.dense(rnn_out, units=len(chars))
char_out = tf.argmax(char_out_probs, axis=2)
# Loss
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=char_target, logits=char_out_probs))
# Optimizer
trainables = tf.trainable_variables()
grads = tf.gradients(loss, trainables)
clipped_gradients, _ = tf.clip_by_global_norm(grads, 1.0)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.apply_gradients(zip(clipped_gradients, trainables))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(200):
        batch_size = 1
        start_idx = np.random.randint(len(data) - max_timesteps, size=[batch_size])
        x = np.zeros([batch_size, max_timesteps])
        t = np.zeros([batch_size, max_timesteps])
        for batch_index, si in enumerate(start_idx):
            for i, data_index in enumerate(range(si, si + max_timesteps)):
                x[batch_index, i] = char_to_ix[data[data_index]]
                t[batch_index, i] = char_to_ix[data[data_index + 1]]
        loss_value, _ = sess.run([loss, train_op], feed_dict={char_in: x, char_target: t})
        if step % 100 == 0:
            print('Step {}\tLoss = {}'.format(step, loss_value))

    # Internal test - Check that we can generate out the same sequence
    expected_output = np.array(data[1:])
    s = sess.run(rnn_zero_state, feed_dict={char_in: x})
    output = sess.run(char_out, feed_dict={char_in: x, rnn_initial_state: s})
    assert np.all(np.array([ix_to_char[o] for o in output[0]]) == expected_output), 'Overfitting did not occur!'

    # Internal test 2 - Feed output and final state back in at every time step
    actual_output = []
    expected_output = np.array(data[1:])
    states = []
    s = sess.run(rnn_zero_state, feed_dict={char_in: x})
    prev_o = char_to_ix[data[0]]
    for i in range(max_timesteps):
        x = np.zeros([1, max_timesteps])
        x[0, 0] = prev_o
        output, probs, s = sess.run([char_out, char_out_probs, rnn_final_state],
                                    feed_dict={char_in: x, rnn_initial_state: s,
                                               rnn_sequence_length: np.ones([1])})
        o = output[0][0]
        p = probs[0][0]
        actual_output.append(ix_to_char[o])
        prev_o = o
    assert np.all(np.array(actual_output) == expected_output), 'Error in trying to sample overfit sequence'
