import numpy as np

inputs = np.array([
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -.8]
])

weights = np.array([
    [.2, .8, -.5, 1],
    [.5, -.91, .26, -.5],
    [-.26, -.27, .17, .87]
])

biases = np.array([2, 3, .5])

weights2 = np.array([
    [0.1, -.14, .5],
    [-.5, 0.12, -.33],
    [-.44, .73, -.13]
])

biases2 = [-1, 2,-0.5]

layer1_output = np.dot(inputs, weights.T) + biases
layer2_output = np.dot(layer1_output, weights2.T) + biases2

print(layer2_output)
