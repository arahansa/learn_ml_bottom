
import numpy as np

def numeric_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x) # x 와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val+h
        fxh1 = f(x)

        x[idx] = tmp_val -h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

def function_2(x):
    return x[0]**2 + x[1]**2

print(numeric_gradient(function_2, np.array([3.0, 4.0])))
print(numeric_gradient(function_2, np.array([0.0, 2.0])))
print(numeric_gradient(function_2, np.array([3.0, 2.0])))
