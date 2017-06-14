import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import scipy
import scipy.misc
import time
import os
import shutil
import webbrowser
from subprocess import Popen, PIPE

def weight_variables(shape):
    initial = tf.truncated_normal_initializer(stddev=0.1)
    return tf.get_variable('weights', shape=shape,
                           initializer=initial)

def bias_variables(shape):
    initial = tf.constant_initializer(0.1)
    return tf.get_variable('biases', shape=shape,
                           initializer=initial)

def embed(session, model, feed_dicts, model_path, labels,
          label_reader, inputs, sprite_reader=None):
    embedding_vals = np.ndarray([0, model.shape[1]])
    for feed_dict in feed_dicts:
        embedding_val = session.run(model, feed_dict=feed_dict)
        embedding_vals = np.concatenate((embedding_vals, embedding_val), axis=0)
    with tf.variable_scope('embedding'):
        embedding_var = tf.get_variable('embedding_var', shape=[embedding_vals.shape[0], embedding_vals.shape[1]],
                                        initializer=tf.constant_initializer(embedding_vals))
    session.run(embedding_var.initializer)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    metadata_dir_path = os.path.join(model_path, 'metadata')
    makedir(metadata_dir_path, overwrite=True)
    metadata_file_path = os.path.join(metadata_dir_path, 'metadata.tsv')
    embedding.metadata_path = metadata_file_path

    # data labels
    with open(metadata_file_path, 'w') as f:
        for label in labels:
            f.write(label_reader(label) + '\n')

    # data images
    if sprite_reader != None:
        images = []
        for input in inputs:
            images.append(sprite_reader(input))
        images = np.array(images)

        sprite_file_path = os.path.join(metadata_dir_path, 'sprite.png')
        embedding.sprite.image_path = sprite_file_path
        embedding.sprite.single_image_dim.extend([images.shape[1], images.shape[2]])

        create_tiled_sprite(images, sprite_file_path)

    writer = tf.summary.FileWriter(os.path.join(model_path, 'train'))
    saver = tf.train.Saver([embedding_var])

    projector.visualize_embeddings(writer, config)
    saver.save(session, os.path.join(model_path, 'train', 'embedding_model.ckpt'))


def create_tiled_sprite(data, sprite_file_path):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)
    # Inverting the colors seems to look better for MNIST
    # data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    sprite = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    sprite = sprite.reshape((n * sprite.shape[1], n * sprite.shape[3]) + sprite.shape[4:])
    sprite = (sprite * 255).astype(np.uint8)

    scipy.misc.imsave(sprite_file_path, sprite)

def variable_summaries(var, scope):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        with tf.name_scope(scope):
          mean = tf.reduce_mean(var)
          tf.summary.scalar('mean', mean)
          with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
          tf.summary.scalar('stddev', stddev)
          tf.summary.scalar('max', tf.reduce_max(var))
          tf.summary.scalar('min', tf.reduce_min(var))
          tf.summary.histogram('histogram', var)

def log_step(step, total_steps, start_time, loss):
    progress = int(step/float(total_steps) * 100)

    seconds = time.time() - start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    print(str(progress) + '%\t|\t',
          int(h), 'hours,', int(m), 'minutes,', int(s), 'seconds\t|\t',
          'Step:', step, '/', total_steps, '\t|\t',
          'Loss:', loss)

def log_epoch(epoch, total_epochs, epoch_loss):
    print('Epoch', epoch, 'completed out of', total_epochs,
          '\t|\tLosses:', epoch_loss, '\n')

def training_dirs(model_name, overwrite=False):
    root_path = './' + model_name + '/'
    makedir(root_path, overwrite=overwrite)

    train_path = root_path + 'train/'
    makedir(train_path, overwrite=overwrite)
    return train_path

def makedir(path, overwrite=False):
    if overwrite and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)


def int_to_sparse(idx, size):
    sparse = np.zeros(size)
    sparse[idx] = 1
    return sparse.tolist()

def open_tensorboard(model_name):
    tensorboard = Popen(['tensorboard', '--logdir=~/Dropbox/Programming/Python/BasicNeuralNet/'
              + model_name + '/train'], stdout=PIPE, stderr=PIPE)
    time.sleep(5)
    webbrowser.open('http://0.0.0.0:6006')
    while input('Press <q> to quit') != 'q':
        continue
    tensorboard.terminate()