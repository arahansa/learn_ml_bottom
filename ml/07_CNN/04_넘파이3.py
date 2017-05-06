import numpy as np
from common.util import im2col


data2 = \
[[
    [
        [1,2],[3,4],[5,6]
    ]

    ]]


data = np.array(data2)
print("---- data shape --- ")
print(data.shape)
print("--------------------")
print(data)
print("-----")
im2 = im2col(data, 2, 2, 1, 0)
print("---- : im2 shape : ----")
print(im2.shape)
print("------")
print(im2)


filter = \
[
    [1,1],
    [1,1]
]
npf = np.array(filter)
print("filter shape :", npf.shape)
res = npf.reshape(1, -1).T
print("filter reshape --- ")
print(res.shape)
print("filter content --- ")
print(res)

print("-------")
print(np.dot(im2, res))



