# Handwriting_NN
Another neural network done in Python 3.7. Run on Windows 10 and MacOS 10.14 Mojave. Commented version forthcoming. Based heavily on the code from Michael Nielson's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com) book online.


## Dependencies
Python 3.7
Pickle 4.0
NumPy 1.15.4
Typing 3.6.6 (dependency easily removed if necessary)


### Requirements
To run, this relies on the mnist.py script being run and stored in the same directory as handwritten_digits.py. This was is code written by [hsjeong5](https://github.com/hsjeong5/) located in their [MNIST-for-Numpy](https://github.com/hsjeong5/MNIST-for-Numpy) respository. Prior to running, the code must be altered to have the data formatted for the neural network. The altered code is in the MNIST folder.

### Running
To run, put all of the .py files in this repository into the same directory then run mnist.py followed by create_nn.py. The parameters can be altered as needed.

The neural network will be saved as a file with the .pkl extension which can be loaded using
```
import pickle
import handwritten_digits

with open(filename, 'rb') as f:
  new_object = pickle.load(f)
```
with all of its behaviors preserved. The line `import handwritten_digits` is necessary so `pickle.load(f)` can reconstruct the object. The file name is generated automatically based on your parameters as is. This is done in the create_nn.py file, but uses the save method in the handwriten_digits.py file.

To use the network to identify an image `x`, run
```
new_object.identify(x)
```
A provided case called provided_use.py is included. It has an example for how to use the loaded object.
