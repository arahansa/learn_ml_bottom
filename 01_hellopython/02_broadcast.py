

import numpy as np


# 차원을 알아서 좀 바꿔줌.
x = np.array([1,2,3])
print(x/2)


print("broad cast")
A = np.array([[1,2], [3,4]])

B = np.array([10,20])

print(A*B)


print("원소 접근...")
X = np.array([[51,55], [14,19], [0,4]])
print(X)

print("X[0]", X[0])
print("X[0][1],", X[0][1])

print("np flat")
X = X.flatten()
print("after flat ", X)
