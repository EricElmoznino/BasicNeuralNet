import tensorflow as tf
import Neural_Net_Helpers
import numpy as np
import time

'''Base class'''
class _Classifier:

    def __init__(self, model_name, input_shape, n_classes, overwrite=False):
        self.model_name = model_name
        self.model_path = './' + self.model_name
        self.train_path = Neural_Net_Helpers.training_dirs(model_name, overwrite=overwrite)

        self.input_shape = input_shape
        self.n_classes = n_classes

        self.input = tf.placeholder(tf.float32, shape=[None]+input_shape, name='input')
        self.output_labels = tf.placeholder(tf.int32, [None], name='labels')
        with tf.variable_scope('hyper_parameters'):
            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_probability')
        self.model = None
        self.saver = None

    def train(self, train_data_file, reader, learning_rate=0.001, batch_size=50, epochs=2, dropout_prob=0.5,
              obtain_metadata=False):
        with open(train_data_file, buffering=20000, encoding='latin-1') as f:
            total_batches = int(sum(1 for line in f) / batch_size)

        with tf.name_scope('training'):
            with tf.name_scope('cost'):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model,
                                                                               labels=self.output_labels)
                cost = tf.reduce_mean(cross_entropy)
                tf.summary.scalar('cost_summary', cost)
            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        with tf.Session() as sess:
            merged_summaries = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.train_path,
                                                 sess.graph)

            sess.run(tf.global_variables_initializer())
            start_time = time.time()
            total_steps = epochs*total_batches
            step = 0
            for epoch in range(1, epochs+1):
                epoch_loss = 1
                batch_x = []
                batch_y = []
                batches_run = 0
                with open(train_data_file, buffering=20000, encoding='latin-1') as f:
                    for line in f:
                        input_data, label = reader(line)
                        batch_x.append(input_data)
                        batch_y.append(label)
                        if len(batch_x) >= batch_size:
                            if step % max(int(total_steps/1000), 1) == 0:
                                if obtain_metadata:
                                    meta_data = tf.RunMetadata()
                                    options = tf.RunOptions()
                                else:
                                    meta_data = None
                                    options = None
                                _, c, s = sess.run([optimizer, cost, merged_summaries],
                                                   feed_dict={self.input: batch_x,
                                                              self.output_labels: batch_y,
                                                              self.dropout_prob: dropout_prob},
                                                   options=options,
                                                   run_metadata=meta_data)
                                if obtain_metadata:
                                    train_writer.add_run_metadata(meta_data, 'step%d' % step)
                                train_writer.add_summary(s, step)
                                Neural_Net_Helpers.log_step(step, total_steps, start_time, c)
                                self.saver.save(sess, self.train_path + 'model.ckpt', global_step=step)
                            else:
                                _, c = sess.run([optimizer, cost],
                                                feed_dict={self.input: batch_x,
                                                           self.output_labels: batch_y,
                                                           self.dropout_prob: dropout_prob})
                            epoch_loss += c
                            batch_x = []
                            batch_y = []
                            batches_run += 1
                            step += 1

                Neural_Net_Helpers.log_epoch(epoch, epochs, epoch_loss)
            self.saver.save(sess, self.train_path + 'model.ckpt')

    def test(self, test_data_file, reader, label_reader=None, sprite_reader=None):
        with tf.name_scope('performance'):
            correct = tf.nn.in_top_k(self.model, self.output_labels, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        with tf.Session() as sess:
            try:
                self.saver.restore(sess, self.train_path + 'model.ckpt')
            except Exception as e:
                print(str(e))

            accuracies = []
            batch_size = 100
            batch_inputs = []
            labels = []
            with open(test_data_file, buffering=20000) as f:
                for line in f:
                    input_data, label = reader(line)
                    batch_inputs.append(input_data)
                    labels.append(label)

                    if len(batch_inputs) >= batch_size:
                        acc = sess.run(accuracy, feed_dict={self.input: batch_inputs,
                                                            self.output_labels: labels,
                                                            self.dropout_prob: 0.0})
                        accuracies.append(acc)
                        batch_inputs = []
                        labels = []

            total_accuracy = sum(accuracies)/float(len(accuracies))
            print('Tested', len(accuracies)*batch_size, 'samples.', 'Accuracy:', total_accuracy)

            # TensorBoard embeddings
            with open(test_data_file, buffering=2000) as f:
                feed_dict = []
                all_inputs = []
                inputs = []
                labels = []
                for line in f:
                    input_data, label = reader(line)
                    all_inputs.append(input_data)
                    inputs.append(input_data)
                    labels.append(label)
                    if len(inputs) >= batch_size:
                        feed_dict.append({self.input: inputs, self.dropout_prob: 0.0})
                        inputs = []
            if not label_reader:
                label_reader = self.default_label_reader
            Neural_Net_Helpers.embed(sess, self.model, feed_dict, self.model_path, labels,
                                     label_reader, all_inputs, sprite_reader=sprite_reader)
            print('Created embeddings for TensorBoard visualization')

    def predict(self, input_data):
        with tf.name_scope('prediction'):
            best_prediction = tf.nn.top_k(self.model, 1)
        with tf.Session() as sess:
            try:
                self.saver.restore(sess, self.train_path + 'model.ckpt')
            except Exception as e:
                print(str(e))

            _, predictions = sess.run(best_prediction, feed_dict={self.input: input_data,
                                                                  self.dropout_prob: 0.0})
            return np.reshape(predictions, (-1)).tolist()

    def default_label_reader(self, label):
        return str(label)

class BasicClassifier(_Classifier):

    def __init__(self, model_name, input_size, n_classes, hidden_layers=[100, 100], overwrite=False):
        _Classifier.__init__(self, model_name, [input_size], n_classes, overwrite=overwrite)
        self.model = self.__build_model([input_size] + hidden_layers + [n_classes])
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    def __build_model(self, layer_sizes):
        with tf.variable_scope('model'):
            model = self.input
            for i in range(1, len(layer_sizes)):
                with tf.variable_scope('layer_' + str(i)):
                    if i == len(layer_sizes) - 1:
                        model = tf.nn.dropout(model, 1.0 - self.dropout_prob)
                    input_size = layer_sizes[i-1]
                    output_size = layer_sizes[i]
                    weights = Neural_Net_Helpers.weight_variables([input_size, output_size])
                    biases = Neural_Net_Helpers.bias_variables([output_size])
                    model = tf.matmul(model, weights) + biases
                    if i != len(layer_sizes) - 1:
                        model = tf.nn.relu(model)
        return model

class ConvolutionClassifier(_Classifier):
    def __init__(self, model_name, width, height, input_channels, n_classes,
                 strides=[[1,1],[1,1]], windows=[[5,5],[5,5]],
                 channels=[32, 64], max_pools=[[2,2],[2,2]],
                 dense_layers=[1024], overwrite=False):
        _Classifier.__init__(self, model_name, [width, height, input_channels], n_classes, overwrite=overwrite)
        self.model = self.__build_model(strides, windows, channels, max_pools, dense_layers)
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    def __build_model(self, strides, windows, channels, max_pools, dense_layers):
        with tf.variable_scope('model'):
            model = self.input
            remaining_width = self.input_shape[0]
            remaining_height = self.input_shape[1]
            n_inputs = self.input_shape[2]

            # build the convolution/pooling part of the network
            conv_layer = 0
            with tf.variable_scope('convolutions'):
                for stride, window, n_outputs, pool_size in zip(strides, windows, channels, max_pools):
                    with tf.variable_scope('convolution_' + str(conv_layer)):
                        model = self.__convolve(model, window, n_inputs, n_outputs, stride)
                        model = model + Neural_Net_Helpers.bias_variables([n_outputs])
                        model = tf.nn.relu(model)
                    with tf.variable_scope('max_pool_' + str(conv_layer)):
                        model = self.__pool(model, pool_size)
                    n_inputs = n_outputs
                    remaining_width = remaining_width / pool_size[0]
                    remaining_height = remaining_height / pool_size[1]
                    conv_layer += 1

            # build the densely connected part of the network
            n_conv_outputs = int(remaining_width*remaining_height*channels[-1])
            layer_sizes = [n_conv_outputs] + dense_layers + [self.n_classes]
            with tf.variable_scope('densely_connected'):
                model = tf.reshape(model, [-1, n_conv_outputs])
                for i in range(1, len(layer_sizes)):
                    with tf.variable_scope('layer_' + str(i)):
                        if i == len(layer_sizes) - 1:
                            model = tf.nn.dropout(model, 1.0 - self.dropout_prob)
                        input_size = layer_sizes[i-1]
                        output_size = layer_sizes[i]
                        weights = Neural_Net_Helpers.weight_variables([input_size, output_size])
                        biases = Neural_Net_Helpers.bias_variables([output_size])
                        model = tf.matmul(model, weights) + biases
                        if i != len(layer_sizes) - 1:
                            model = tf.nn.relu(model)

        return model

    def __convolve(self, x, window, n_inputs, n_outputs, stride):
        weights = Neural_Net_Helpers.weight_variables(window + [n_inputs] + [n_outputs])
        stride = [1] + stride + [1]
        return tf.nn.conv2d(x, weights, stride, padding='SAME')

    def __pool(self, x, pool_size):
        stride = [1] + pool_size + [1]
        return tf.nn.max_pool(x, ksize=stride, strides=stride, padding='SAME')

class RecurrentClassifier(_Classifier):
    def __init__(self, model_name, input_size, memory_size, n_classes,
                 state_size=128, recurrent_layers=2, overwrite=False):
        _Classifier.__init__(self, model_name, [memory_size, input_size], n_classes, overwrite=overwrite)
        self.model = self.__build_model(state_size, recurrent_layers)
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    def __build_model(self, state_size, recurrent_layers):
        with tf.variable_scope('model'):
            model = self.input

            with tf.variable_scope('recurrent') as rec:
                def create_rnn():
                    cell = tf.contrib.rnn.BasicLSTMCell(state_size)
                    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0-self.dropout_prob)
                    return cell
                rnn_cell = tf.contrib.rnn.MultiRNNCell([create_rnn() for _ in range(recurrent_layers)])
                outputs, _ = tf.nn.dynamic_rnn(rnn_cell, model, dtype=tf.float32, scope=rec)
                outputs = tf.unstack(outputs, self.input_shape[0], axis=1)
                model = outputs[-1]

            with tf.variable_scope('softmax'):
                weights = Neural_Net_Helpers.weight_variables([state_size, self.n_classes])
                biases = Neural_Net_Helpers.bias_variables([self.n_classes])
                model = tf.matmul(model, weights) + biases

        return model




