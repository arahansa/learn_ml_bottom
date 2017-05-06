import numpy as np

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

# 2일 확률이 높다고 추정했을 때
print(mean_squared_error(np.array(y),np.array(t)))

# 예 7일 확률이 높다고 추정했을 때
y = [ 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(y),np.array(t)))



def cross_entropy_error(y,t):
    print("t :", t, "y : ", y)
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print("---")
print(cross_entropy_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))


def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    print("t :", t , "y : ", y)
    batch_size = y.shape[0]
    delta = 1e-7
    print("batch size :", batch_size)
    return -np.sum( t * np.log(y+ delta)) / batch_size

print("---")
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))

y = [[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0], [0.1, 0.05, 0.3, 0.3, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]]
t = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0,0,1,0,0,0,0,0,0,0]]
print(cross_entropy_error(np.array(y), np.array(t)))


print("----clear----")
print("y shape", np.array(y).shape[0])

# 정답 레이블이 원핫인코딩이 ㄴ아니라 2, 7 등의 숫자 레이블로 주어졌을 때의 경우
# def cross_entropy_error(y,t):
#     if y.dim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#
#     batch_size = y.shape[0]
#     return -np.sum( np.log(y[np.arange(batch_size), t])) / batch_size
