# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 정규분포로 초기화

    def setW(self, W):
        print("set W ")
        self.W = W

    def predict(self, x):
        print("self W :", self.W)
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)

        loss = cross_entropy_error(y, t)
        print("y : ", y, ", return loss :", loss)
        return loss


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()
net.setW(np.array([
    [0.47355232, 0.9977393, 0.84668094],
    [0.85557411, 0.03563661, 0.69422093]
]))

print("가중치 매개변수 net : \n", net.W, "\n")

x = np.array([0.6, 0.9])
p = net.predict(x)
print("p : \n", p)
print("최대값의 인덱스 :", np.argmax(p))

t = np.array([0, 0, 1])
print("손실 함수 값  \n", net.loss(x, t))


def f(W):
    return net.loss(x, t)


# f = lambda w: net.loss(x, t)

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

        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)

        # 여기가 grad[idx] 에 미분값을 넣어준다..
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        print("현재 x [idx] :", x[idx], idx, ", float:", tmp_val)
        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad


dW = numerical_gradient(f, net.W)

print("dW 기울기~ \n", dW)