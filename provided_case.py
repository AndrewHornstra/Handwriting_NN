import pickle
from mnist import load
import numpy as np
from typing import List
import handwritten_digits


# Load the same data for the provided case. This code is unneccessary for your
# own formatted images.
def vectorize(i: int) -> List:
    vec = np.zeros((10, 1))
    vec[i] = 1
    return vec


train_images, train_labels, test_images, test_labels = load()
train_labels = np.array([vectorize(i) for i in train_labels])
test_labels = np.array([vectorize(i) for i in test_labels])
train_data = np.array(list(zip(train_images, train_labels)))
test_data = np.array(list(zip(test_images, test_labels)))

# Load the provided NN object.
with open("SNN_1_20_0.4_HL_[100].pkl", 'rb') as f:
    new_object = pickle.load(f)

# Print the results. In this provided case, the NN was trained once and had
# around 50% accuracy. The proper digits represented in test_data[0][0] is
# stored in test_data[0][1]. In the provided case, 7.
print("The NN sees:", new_object.identify(test_data[0][0]))
print("The correct digit is:", test_data[0][1].argmax())
