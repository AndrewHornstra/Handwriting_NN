from handwritten_digits import Network
from mnist import load
import numpy as np
from typing import List


def vectorize(i: int) -> List:
    vec = np.zeros((10, 1))
    vec[i] = 1
    return vec


epochs = 60
mini_batch_size = 20
eta = 0.4
NN_struct = []
input_layer = 784
output_layer = 10
hidden_layers = [100, 123]
NN_struct.append(input_layer)
for layer in hidden_layers:
    NN_struct.append(layer)
NN_struct.append(output_layer)
train_images, train_labels, test_images, test_labels = load()
train_labels = np.array([vectorize(i) for i in train_labels])
test_labels = np.array([vectorize(i) for i in test_labels])
train_data = np.array(list(zip(train_images, train_labels)))
test_data = np.array(list(zip(test_images, test_labels)))
network = Network([784, 100, 10])
network.train_SGD(train_data, epochs, mini_batch_size,
                  eta, test_data=test_data)
print(f"Saving object via pickle with parameters:\nepochs - {epochs}\n"
      f"minibatch size - {mini_batch_size}\neta - {eta}")
network.save(f"SNN_{epochs}_{mini_batch_size}_{eta}_HL_{hidden_layers}.obj")
print("Done.")
