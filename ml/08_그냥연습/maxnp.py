import numpy as np


testarray = np.array([
    [4,2,10,10],
    [10,5,6,3],
    [7,1,9,5]
])

print("size : ", testarray.size)

# axis=0 일 때는 세로로 해서 긴거 가져온다
t = testarray.argmax(axis=0)
print("argmax : ", t)

# axis=1 일 때는 가로로 해서 긴거 가져온다
t = testarray.argmax(axis=1)
print("argmax axis 1: ", t)

t = np.array([
    [0, 1, 0, 0]
])

t = t.argmax(axis=1)
print("t :", t)

batch_size = testarray.shape[0]
print("batch size :", batch_size)
print(np.arange(batch_size))