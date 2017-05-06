# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)


print(softmax(np.array([[1,2],[3,4]])))

t = np.array([[1,2,3],[3,4,5], [5,6, 7]])
print("t dim :", t.ndim)
print("exp : ", np.exp(t))
print("sum : ", np.sum(np.exp(t), axis=0))

print("---")
t = np.array([[8,8,8], [8,8,8], [8,8,8]])
a = np.array([1,2,4])
c = t/a
print("c:", c)
print("shape a ", a.shape)
print("ndim a :", a.ndim)

ttt = np.array([
    [[1,2],[3,4]],
    [[1,2],[3,4]]
])
print("ttt ndim :", ttt.ndim)

t = np.array([[6,2,3],[3,4,5], [5,6,7]])
aMax = np.max(t, axis=0)
print("aMax : ", aMax)

aMax = np.max(np.array([[5,4],[3,8]]), axis=0)
print("aMax : ", aMax)


ac1 = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

ac2 = np.array([
    [1]
])

print("ac1 T:", ac1.T)

print("x + y : ", ac1 + ac2)

print("---")
m = np.array([
    [1010, 1000, 990]
])
print("m ndim : ", m.ndim)
print("sf : ", softmax(m))

