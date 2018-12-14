import numpy as np
import pickle
from typing import List, Tuple


class Network:
    def __init__(self, sizes: List) -> None:
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1).astype(np.float64)
                       for y in sizes[1:]]
        self.weights = [np.random.randn(y, x).astype(np.float64)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z: List) -> List:
        z = np.clip(z, -500, 500)
        return 1.0/(1.0 + np.exp(-z))

    def feedforward(self, a: List) -> List:
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid((np.dot(w, a) + b))
        return a

    def train_SGD(self, training_data: List, epochs: int, mini_batch_size: int,
                  eta: float, test_data: List = None) -> None:
        if test_data is not None:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data is not None:
                print(f"Epoch {j+1}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j+1} complete.")

    def update_mini_batch(self, mini_batch: List, eta: float) -> None:
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnd for nb, dnd in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch)*nw)
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)*nb)
                       for b, nb in zip(self.biases, nabla_b)]

    def vectorize(self, i: int) -> List:
        vec = np.zeros((10, 1))
        vec[i] = 1
        return vec

    def evaluate(self, test_data: List) -> int:
        results = np.array([(self.vectorize(self.feedforward(x).argmax(0)), y)
                            for x, y in test_data])
        return sum([int(np.all(x == y)) for x, y in results])

    def sigmoid_prime(self, z: float) -> float:
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def backprop(self, x: List, y: List) -> Tuple[List[float], List[float]]:
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        z = []
        a = [x]
        for b, w in zip(self.biases, self.weights):
            z.append(np.dot(w, a[-1]) + b)
            a.append(self.sigmoid(z[-1]))
        delta = self.cost_derivative(a[-1], y) * self.sigmoid_prime(z[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, a[-2].transpose())
        for l in range(2, self.num_layers):
            sp = self.sigmoid_prime(z[-l])
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, a[-l-1].transpose())
        return nabla_b, nabla_w

    def cost_derivative(self, output: List, y: List) -> List:
        return (output - y)

    def save(self, filename: str) -> None:
        with open(f'{filename}', 'wb') as f:
            pickle.dump(self, f)

    def identify(self, x: List) -> int:
        return self.feedforward(x).argmax(0)[0]
