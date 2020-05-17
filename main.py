import numpy as np

_inputs = np.array([1, 2, 3, 2.5])


weights1 = np.array([.2, .8, -.5, 1])
weights2 = np.array([.5, -.91, .26, -.5])
weights3 = np.array([-.26, -.27, .17, .87])

bias1 = 2
bias2 = 3
bias3 = .5

output = np.array([
    np.dot(_inputs, weights1) + bias1,
    np.dot(_inputs, weights2) + bias2,
    np.dot(_inputs, weights3) + bias3
    ]) 
    
print(output)