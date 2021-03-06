import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x,y)
plt.ylim(-0.1, 1.1) #y 축 범위 지정
plt.show()

def relu(x):
    return np.maximum(0,x)