import sys,os
import numpy as np
sys.path.append(os.pardir)

from common.util import im2col


# 데이터수, 채널수, 높이, 너비
x1 = np.random.rand(1,3,7,7)

col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)


x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)



x3 = np.random.rand(2,3,3,2)
x3 = np.dot(x3, 100)
x3 = np.divide(x3, 10)
print("---x3---")
print(x3)

col3 = im2col(x3, 2, 2, stride=1, pad=0)
print("--- x3 im2col --- ")
print(col3.shape)
print(col3)