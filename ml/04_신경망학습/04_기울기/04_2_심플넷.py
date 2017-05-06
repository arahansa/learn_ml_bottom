
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화

    def setW(self, W):
        self.W = W

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        print("x :", x)
        print("t :", t)
        z = self.predict(x)
        y = softmax(z)

        loss = cross_entropy_error(y, t)
        print("y : ", y, ", return loss :", loss)
        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()
net.setW(np.array([
    [1, 2, 3],
    [1, 2, 3]
]))

print("가중치 매개변수 net : \n", net.W, "\n")

x = np.array([1, 2])
p = net.predict(x)
print("p : \n", p)
print("최대값의 인덱스 :", np.argmax(p))

t = np.array([0, 0, 1])
print("손실 함수 값  \n", net.loss(x,t))

def f(W):
    return net.loss(x,t)
#f = lambda w: net.loss(x, t)

def numerical_gradient(f, x):
    print("numerric gradient...?")
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    print("직접 호출..? f? ")
    f(np.array([3, 4]))


    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    print("it~~ : ", it);
    print("size : ", x.size)

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        print("현재 x [idx] :" , x[idx], idx, ", float:", tmp_val)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)

        print("fxh2 : ", fxh2)
        # 여기가 grad[idx] 에 미분값을 넣어준다..
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        print("grad[idx] :", grad[idx])
        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad


dW = numerical_gradient(f, net.W)

print("dW 기울기~ \n",dW)