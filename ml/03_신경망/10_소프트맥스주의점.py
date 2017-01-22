import numpy as np


def softmax(a):
    exp_a = np.exp(a)  # 지수함수
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([1010, 1000, 990])

print("--- soft max --- ")
print(softmax(a)) # 소프트 맥스 함수의 계산

c = np.max(a)
print(a-c)

print('after sofmax')
print(softmax(a-c))



