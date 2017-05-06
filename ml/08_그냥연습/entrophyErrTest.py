import numpy as np
from two_layer_net import TwoLayerNet
from common.functions import *

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        print("t max ")
        t = t.argmax(axis=1)
        print("after t : ", t)


    batch_size = y.shape[0]
    print("y shape: ", y.shape, y.shape[0]," , arrange : ", np.arange(batch_size))
    print("arrange :", [np.arange(batch_size), t])
    print("---")
    yt = y[np.arange(batch_size),t]
    print(" y np arrange :  ", yt)
    print("y log :", np.log(yt))
    print("np sum  :", np.sum(np.log(y[np.arange(batch_size), t])))
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


x = np.array([
    [0.00235587, 0.04731414,  0.95032999],
    [0.00235587, 0.04731414,  0.95032999],
    [0.00235587, 0.04731414,  0.95032999]
])
t = np.array([
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0]
])

print("cee :", cross_entropy_error(x, t))