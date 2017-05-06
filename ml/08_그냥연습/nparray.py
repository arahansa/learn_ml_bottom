import numpy as np


t = np.array([
    [1,4,3],
    [4,5,3],
    [8,2,7],
])

t1 = t[[0,1,2],[1,1,0]]
print("t1:", t1)
print("log :", np.log(t1))

print("max : ", np.max(t, axis=1))


