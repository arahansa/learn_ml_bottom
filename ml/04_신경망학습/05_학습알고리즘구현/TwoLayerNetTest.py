
import numpy as np
from two_layer_net import TwoLayerNet
from common.functions import *
from common.gradient import numerical_gradient


network = TwoLayerNet(input_size=3, hidden_size=3, output_size=3,weight_init_std=1.0)

testarray = np.array([
    [1.0,2.0,3.0],
    [4.0,5.0,6.0],
    [7.0,8.0,9.0]
])

t2 = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])


t = np.array([
    [1.0,0.0,0.0],
    [1.0,0.0,0.0],
    [1.0,0.0,0.0]
])

network.setParam(testarray, t2)

print("network w2 : ", network.params['W2'])

print("---- gradient start ----")
grad = network.numerical_gradient(testarray, t)
for key in ('W1', 'b1', 'W2', 'b2'):
    print(key," : ", grad[key])


