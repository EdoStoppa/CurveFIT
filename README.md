# CurveFIT

A simple Neural Network project that will fit the pseudo-random generated data using a Fully Connected Neural Network (with everything implemented from scratch)

## Description

This simple project aims at creating a neural network that is able to fit a curve on any data in 2 dimensions. The Network, the learning algorithm (Gradient Descent with Backpropagation) and the loss function (MSE) are all implemented from scratch using only NumPy.

<p float="left">
  <img src="/imgs/fit0.png" width="400" />
  <img src="/imgs/fit2.png" width="400" />
</p>

## Network Architecture
The network does a simple regression given a single input to know the value of the value that should be associated. So, given that a point p is formed by two elements p<sub>x</sub> and p<sub>y</sub>, the input will be p<sub>x</sub>, and the output will be a value that should be as close as possible to p<sub>y</sub>.

To achieve that, the network is composed of a hidden layer with 24 perceptrons, and an output layer with a single perceptron. The activation function used is tanh (given the simplicity of the network, the vanishing gradient problem is not a issue).

The architecture is the following:
<p float="center"> <img src="/imgs/nn.png" width="400" /> </p>

## Getting Started

### Python and Libraries

* Tested using Python 3.8
* NumPy v. 1.20.3
* Matplotlib 3.4.2

### How to Run

If on a Unix environment use
```
python3 CurveFIT.py
```
If on a Windows environment use
```
python CurveFIT.py
```

### Modifying data

* Modify the method called generate_data()
