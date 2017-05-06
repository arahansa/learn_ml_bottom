import numpy as np
from common.util import im2col


data2 = \
[
    [
        [
            [1,2,3],
            [4,5,6]
        ],
        [
            [7,8,9],
            [10,11,12]
        ]

    ],
[
        [
            [13,14,15],
            [16,17,18]
        ],
        [
            [19,20,21],
            [21,22,23]
        ]

    ]
]


data = np.array(data2)
print(data.shape)
print("-----")
print(data)
print("-----")
im2 = im2col(data, 2, 2, 1, 0)
print("---- : im2 shape : ----")
print(im2.shape)
print("------")
print(im2)
