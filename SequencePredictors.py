import tensorflow as tf
import Neural_Net_Helpers
import numpy as np


class RecurrentPredictor:

    def __init__(self, model_file, n_inputs, n_outputs,
                 state_size=128, recurrent_layers=2, embedding_size=200):
        self.model_file = model_file
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.state_size = state_size
        self.recurrent_layers = recurrent_layers

        self.input = tf.placeholder(tf.int32, [None, None])
        self.output_labels = tf.placeholder(tf.int32, [None, None])
        self.max_memory = tf.placeholder(tf.int32)
        self.dropout_prob = tf.placeholder(tf.float32)
        self.initial_state = tf.placeholder(tf.float32, [recurrent_layers, 2, None, state_size])

        self.model, self.final_state = self.__build_model(embedding_size)
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    def train(self, train_data_file, reader, max_memory, learning_rate=0.001, batch_size=50, epochs=2, dropout_prob=0.5):
        with open(train_data_file, buffering=20000, encoding='latin-1') as f:
            total_batches = sum(1 for line in f) / batch_size / max_memory

        cost = tf.contrib.seq2seq.sequence_loss(
            tf.reshape(self.model, [batch_size, max_memory, self.n_outputs]),
            self.output_labels,
            tf.ones([batch_size, max_memory], tf.float32))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(1, epochs+1):
                epoch_cost = 0.0
                mem_x = []
                mem_y = []
                batch_x = []
                batch_y = []
                batches_run = 0
                state = np.zeros((self.recurrent_layers, 2, batch_size, self.state_size))

                with open(train_data_file, buffering=20000, encoding='latin-1') as f:
                    for line in f:
                        input_idx, label_idx = reader(line)
                        mem_x.append(input_idx)
                        mem_y.append(label_idx)
                        if len(mem_x) >= max_memory:
                            batch_x.append(mem_x)
                            batch_y.append(mem_y)
                            mem_x = []
                            mem_y = []

                        if len(batch_x) >= batch_size:
                            _, c, state = sess.run([optimizer, cost, self.final_state],
                                                   feed_dict={self.input: batch_x,
                                                              self.output_labels: batch_y,
                                                              self.max_memory: max_memory,
                                                              self.dropout_prob: dropout_prob,
                                                              self.initial_state: state})
                            epoch_cost += c
                            batches_run += 1
                            batch_x = []
                            batch_y = []
                            print('Batch run:', batches_run, '/', total_batches, '| Epoch:', epoch, '| Batch Loss:', c)

                print('Epoch', epoch, 'completed out of', epochs, 'costs:', epoch_cost, 'perplexity:', np.exp(epoch_cost / total_batches))
            self.saver.save(sess, self.model_file)

    def test(self, test_data_file, reader):
        max_memory = 400
        batch_size = 1
        cost = tf.contrib.seq2seq.sequence_loss(
            tf.reshape(self.model, [batch_size, max_memory, self.n_outputs]),
            self.output_labels,
            tf.ones([batch_size, max_memory], tf.float32))

        with tf.Session() as sess:
            try:
                self.saver.restore(sess, self.model_file)
            except Exception as e:
                print(str(e))

            total_cost = 0
            total_batches = 0
            mem_x = []
            mem_y = []
            batch_x = []
            batch_y = []
            state = np.zeros((self.recurrent_layers, 2, batch_size, self.state_size))

            with open(test_data_file, buffering=20000) as f:
                for line in f:
                    input_idx, label_idx = reader(line)
                    mem_x.append(input_idx)
                    mem_y.append(label_idx)
                    if len(mem_x) >= max_memory:
                        batch_x.append(mem_x)
                        batch_y.append(mem_y)
                        mem_x = []
                        mem_y = []

                    if len(batch_x) >= batch_size:
                        c, state = sess.run([cost, self.final_state],
                                            feed_dict={self.input: batch_x,
                                                       self.output_labels: batch_y,
                                                       self.max_memory: max_memory,
                                                       self.dropout_prob: 0.0,
                                                       self.initial_state: state})
                        total_cost += c
                        total_batches += 1
                        batch_x = []
                        batch_y = []

            print('Tested', max_memory*batch_size*total_batches, 'samples.', 'Perplexity:', np.exp(total_cost / total_batches))

    def predict(self, inputs):
        best_prediction = tf.argmax(self.model[-1], 0)
        with tf.Session() as sess:
            try:
                self.saver.restore(sess, self.model_file)
            except Exception as e:
                print(str(e))

            initial_state = np.zeros((self.recurrent_layers, 2, 1, self.state_size))
            return sess.run(best_prediction, feed_dict={self.input: [inputs],
                                                        self.max_memory: len(inputs),
                                                        self.dropout_prob: 0.0,
                                                        self.initial_state: initial_state})


    def __build_model(self, embedding_size):
        embedding = tf.Variable(initial_value=tf.random_uniform([self.n_inputs, embedding_size], -1.0, 1.0))
        input_data = tf.nn.embedding_lookup(embedding, tf.reshape(self.input, [-1]))
        input_data = tf.reshape(input_data, [-1, self.max_memory, embedding_size])

        rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.state_size)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=1.0 - self.dropout_prob)
        rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell] * self.recurrent_layers)

        initial_state = tf.unstack(self.initial_state, axis=0)
        rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(s[0], s[1]) for s in initial_state])
        outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, input_data, initial_state=rnn_tuple_state)
        outputs = tf.reshape(outputs, [-1, self.state_size])

        weights = Neural_Net_Helpers.weight_variables([self.state_size, self.n_outputs])
        biases = Neural_Net_Helpers.bias_variables([self.n_outputs])
        outputs = tf.matmul(outputs, weights) + biases

        return outputs, final_state
