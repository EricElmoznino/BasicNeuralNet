import Classifiers
import SequencePredictors
import numpy as np

def string_to_params(line):
    input = [float(s) for s in line.split(',')[0].split(':')]
    label_idx = int(line.split(',')[1])
    return input, label_idx

def string_to_img_params(line):
    input = [float(s) for s in line.split(',')[0].split(':')]
    input = np.reshape(input, (28, 28, 1)).tolist()
    label_idx = int(line.split(',')[1])
    return input, label_idx

def string_to_rec_params(line):
    input = [float(s) for s in line.split(',')[0].split(':')]
    input = np.reshape(input, (28, 28)).tolist()
    label_idx = int(line.split(',')[1])
    return input, label_idx

def string_to_seq_params(line):
    input = int(line.split(',')[0])
    label = int(line.split(',')[1])
    return input, label

def basic_list_to_img(input_data):
    return np.reshape(input_data, (28, 28)).tolist()

def twoD_list_to_img(input_data):
    return np.reshape(input_data, (28, 28)).tolist()

def rec_list_to_img(input_data):
    return input_data

def numpy_to_csv(input, label, file_name):
    with open(file_name, 'w') as f:
        for x, y in zip(input, label):
            line = ':'.join([str(x_) for x_ in x]) + ',' + str(y) + '\n'
            f.write(line)

# The following blocks create, train, and test individual models, as
# well as use them to make predictions. The classifiers were trained on the MNIST
# dataset and the sequence predictor was trained on the PTB dataset. To run and experiment,
# download the datasets and format them appropriately (according to the 'reader' functions
# above that are fed to the net during training), then uncomment the block corresponding
# to the model you want to train. 

# neural_net = Classifiers.BasicClassifier('BasicClassifier', 784, 10, overwrite=True)
# neural_net.train('train.csv', string_to_params)
# neural_net.test('test.csv', string_to_params, sprite_reader=basic_list_to_img)
# testing = [[num%10 for num in range(784)], [num%9 for num in range(784)]]
# print(neural_net.predict(testing))

# neural_net = Classifiers.ConvolutionClassifier('ConvolutionClassifier', 28, 28, 1, 10, overwrite=True)
# neural_net.train('train.csv', string_to_img_params, batch_size=128, epochs=10)
# neural_net.test('test.csv', string_to_img_params, sprite_reader=twoD_list_to_img)
# testing = [[[[0 for elem in range(1)] for col in range(28)] for row in range(28)]]
# print(neural_net.predict(testing))

# neural_net = Classifiers.RecurrentClassifier('RecurrentClassifier', 28, 28, 10, recurrent_layers=3, overwrite=True)
# neural_net.train('train.csv', string_to_rec_params, batch_size=128, epochs=10, obtain_metadata=False)
# neural_net.test('test.csv', string_to_rec_params, sprite_reader=rec_list_to_img)
# testing = [[[col%2 for col in range(28)] for row in range(28)]]
# print(neural_net.predict(testing))

# neural_net = SequencePredictors.RecurrentPredictor('./seqmodel.ckpt', 10000, 10000, state_size=650, recurrent_layers=3)
# neural_net.train('train_seq.csv', string_to_seq_params, 35, batch_size=20, epochs=40, dropout_prob=0.65)
# neural_net.test('test_seq.csv', string_to_seq_params)
# print(neural_net.predict([2, 4, 76, 12]))

