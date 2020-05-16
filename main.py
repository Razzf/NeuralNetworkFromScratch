import numpy as np

_inputs = np.array([1,2,3])
weights = np.array([.5, -.4, .3])
bias = 2

output = _inputs[0] * weights[0] + _inputs[1] * weights[1] + _inputs[2] * weights[2] + bias
print(output)