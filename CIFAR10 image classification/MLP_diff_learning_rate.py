"""
Ayoung Kang
"""


from __future__ import division
from __future__ import print_function

import sys
try:
   import _pickle as pickle
except:
   import pickle

import numpy as np
import matplotlib.pyplot as plt
import csv


# This is a class for a LinearTransform layer which takes an input
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
        # DEFINE __init function
        self.W = W
        self.b = b

    def forward(self, x):
        # DEFINE forward function
        # dim of Z: (# of samples x size of hidden layers)
        self.x = x
        z = np.dot(x, self.W) + self.b
        return z

    def backward(self, grad_output, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
        # DEFINE backward function
        # dE/dx = dE/dz1 * dz1/dx = grad_output * W
        # dE/dw = dE/dz1 * dz1/dw = grad_output * x
        # dE/db = dE/dz1 * dz1/db = grad_output * [1...1]T = sum of the rows

        dw = np.dot(self.x.T, grad_output)
        db = np.sum(grad_output, axis=0, keepdims=True)
        dx = np.dot(grad_output, self.W.T)
        return dw, db, dx

# ADD other operations in LinearTransform if needed

# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
        # DEFINE forward function
        self.x = x
        self.relu_output = x * (x > 0)
        return self.relu_output

    def backward(self, grad_output, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
        # DEFINE backward function
        # dE/dz1 = dE/dz2 * g'(z1)
        derivative = self.relu_derivative(self.relu_output)
        dz = grad_output * derivative

        return dz

    def relu_derivative(self, x):
        x[x > 0] = 1
        x[x <= 0] = 0
        return x

# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def forward(self, x):
        # DEFINE forward function
        self.sigmoid_output = 1.0 / (1.0 + np.exp(-x))
        return self.sigmoid_output

    def backward(self, y, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
        # DEFINE backward function
        self.y = y
        dz2 = self.sigmoid_output - y
        return dz2

# ADD other operations and data entries in SigmoidCrossEntropy if needed


# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
        # INSERT CODE for initializing the network
        self.W1 = np.random.randn(input_dims, hidden_units) * 0.01
        self.b1 = np.random.randn(1, hidden_units) * 0.01
        self.W2 = np.random.randn(hidden_units, 1) * 0.01
        self.b2 = np.random.randn(1,1) * 0.01

        self.l1 = LinearTransform(self.W1, self.b1)
        self.l2 = LinearTransform(self.W2, self.b2)

        self.relu = ReLU()
        self.sce = SigmoidCrossEntropy()
        self.y_pred = 0
        self.prob = 0

        self.x_batch = 0
        self.y_batch = 0

        self.W1_momentum = 0
        self.W2_momentum = 0
        self.b1_momentum = 0
        self.b2_momentum = 0


    def train(self, x_batch, y_batch, learning_rate, momentum,l2_penalty):
        # INSERT CODE for training the network
        # Forward
        self.x_batch = x_batch
        self.y_batch = y_batch

        z1 = self.l1.forward(self.x_batch)
        z1_relu = self.relu.forward(z1)
        z2 = self.l2.forward(z1_relu)
        z2_sigmoid = self.sce.forward(z2)
        self.prob = z2_sigmoid
        self.y_pred = np.round(z2_sigmoid)

        # backward
        dz2 = self.sce.backward(y_batch)
        dw2, db2, dx2 = self.l2.backward(dz2)
        dz1 = self.relu.backward(dx2)
        dw1, db1, dx1 = self.l1.backward(dz1)

        # calculate momentum and weight decay regularization
        self.W1_momentum = momentum * self.W1_momentum - learning_rate * (dw1 + l2_penalty * self.W1)
        self.b1_momentum = momentum * self.b1_momentum - learning_rate * db1
        self.W2_momentum = momentum * self.W2_momentum - learning_rate * (dw2 + l2_penalty * self.W2)
        self.b2_momentum = momentum * self.b2_momentum - learning_rate * db2

        # update rule for each parameter
        self.W1 += self.W1_momentum
        self.b1 += self.b1_momentum
        self.W2 += self.W2_momentum
        self.b2 += self.b2_momentum

        loss = self.calculate_loss(self.y_batch, self.prob, l2_penalty)
        acc = self.cal_accuracy(self.y_batch, self.y_pred)
        return loss, acc


    def evaluate(self, x_batch, y_batch):
        # INSERT CODE for testing the network
        z1 = self.l1.forward(x_batch)
        z1_relu = self.relu.forward(z1)
        z2 = self.l2.forward(z1_relu)
        z2_sigmoid = self.sce.forward(z2)
        prob = z2_sigmoid
        y_pred = np.round(z2_sigmoid)

        loss = self.calculate_loss(y_batch, prob, l2_penalty)
        acc = self.cal_accuracy(y_pred, y_batch)
        return loss, acc


    # ADD other operations and data entries in MLP if needed
    def calculate_loss(self, y_batch, prob, l2_penalty):

        loss = -(y_batch * np.log(prob + 1e-15) + (1 - y_batch) * np.log(1 - prob + 1e-15))
        regularization = 0.5 * l2_penalty * (np.linalg.norm(self.W1) ** 2 + np.linalg.norm(self.W2) ** 2)
        loss += regularization

        return np.sum(loss)

    def cal_accuracy(self, y_batch, y_pred):
        accuracy = (y_pred == y_batch).mean()
        #print(type(accuracy))
        return accuracy * 100.


def normalize(x):
    x -= np.mean(x, axis=0)
    x /= np.ptp(x, axis=0)
    return x


def random_permute(x, y):
    n = np.shape(x)[0]
    idx = np.random.permutation(n)
    x = x[idx]
    y = y[idx]
    return x, y



if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = data[b'train_data']
    train_y = data[b'train_labels']
    test_x = data[b'test_data']
    test_y = data[b'test_labels']

    # visualize
    #for i in range(9):
    #    plt.subplot(330 + 1 + i)
    #    data = np.reshape(train_x[i], (32, 32, 3), order='F')
    #    plt.imshow(np.transpose(data, (1, 0, 2)), interpolation='bicubic')
    #plt.show()

    num_examples, input_dims = train_x.shape
    # INSERT YOUR CODE HERE
    train_x = train_x.astype(float)
    train_x = normalize(train_x)
    test_x = test_x.astype(float)
    test_x = normalize(test_x)

    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    #np.random.seed(1)
    num_epochs = 100
    batch_size = 64
    num_batches = num_examples // batch_size
    #print(num_batches)
    num_test_examples = np.shape(test_x)[0]
    num_val_batches = num_test_examples // batch_size
    #hidden_units_list = [32, 64, 128, 256, 512]
    hidden_units = 128

    #learning_rate = 0.001
    learning_rate_list = [0.005, 0.001, 0.0005, 0.0001]
    momentum = 0.8
    l2_penalty = 0

    test_acc_results = []
    for i in range(len(learning_rate_list)):
        learning_rate = learning_rate_list[i]
        mlp = MLP(input_dims, hidden_units)
        total_train_loss, total_train_acc = [], []
        total_val_loss, total_val_acc = [], []

        for epoch in range(num_epochs):
            # INSERT YOUR CODE FOR EACH EPOCH HERE
            bat_train_loss, bat_train_acc = [], []
            total_loss = 0
            train_x, train_y = random_permute(train_x, train_y)
            for b in range(num_batches):
                train_loss_, train_acc_ = mlp.train(train_x[b*batch_size:(b+1)*batch_size, : ],
                                                    train_y[b*batch_size:(b+1)*batch_size, : ], learning_rate, momentum, l2_penalty)
                bat_train_loss.append(train_loss_)
                bat_train_acc.append(train_acc_)

                total_loss += train_loss_

                print(
                    '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                        epoch + 1,
                        b + 1,
                        total_loss / ((b+1)*batch_size),
                    ),
                    end='',
                )
                sys.stdout.flush()

            total_train_loss.append(bat_train_loss)
            total_train_acc.append(bat_train_acc)

            # validation
            bat_val_loss, bat_val_acc = [], []
            test_x, test_y = random_permute(test_x, test_y)
            for b in range(num_val_batches):
                val_loss_, val_acc_ = mlp.evaluate(test_x[b*batch_size:(b+1) * batch_size, :],
                                                   test_y[b*batch_size:(b+1) * batch_size, :])

                bat_val_loss.append(val_loss_)
                bat_val_acc.append(val_acc_)

            total_val_loss.append(bat_val_loss)
            total_val_acc.append(bat_val_acc)

            print()
            print('    Train Loss: {:.3f}    Avg. train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
                np.sum(total_train_loss[-1]),
                np.sum(total_train_loss[-1])/num_examples,
                np.mean(total_train_acc[-1]),
            ))
            print('    Test Loss: {:.3f}     Avg. test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
                np.sum(total_val_loss[-1]),
                np.sum(total_val_loss[-1])/num_test_examples,
                np.mean(total_val_acc[-1]),
            ))

        print()
        print('Best train accuracy: {:.2f}    Best test accuracy: {:.2f}'.format(
            np.max(np.mean(total_train_acc, axis=1)),
            np.max(np.mean(total_val_acc, axis=1)),
        ))


        #train_acc = np.mean(total_train_acc, axis=1)
        val_acc = np.mean(total_val_acc, axis=1)
        test_acc_results.append(val_acc)

    with open('accuracy_learning_rate.csv', 'w', newline='') as f:
        write = csv.writer(f)
        write.writerows(test_acc_results)





