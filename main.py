import numpy as np

inputs = np.array([1, 2, 3, 2.5])

weights = np.array([
    [.2, .8, -.5, 1],
    [.5, -.91, .26, -.5],
    [-.26, -.27, .17, .87]
])

biases = np.array([2, 3, .5])


output = np.dot(weights, inputs) + biases

print(output)