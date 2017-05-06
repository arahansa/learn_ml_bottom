# coding: utf-8
import sys, os
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.functions import *

testarray = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])


print("testarray  : ", testarray)
print("testarray shape  : ", testarray.shape)
sum = np.sum(testarray, axis=0)
print("sum : ", sum)
print("sum : ", sum.shape)

sum = np.sum(testarray)
print("sum no axis : ", sum)


print("1- : ", 1-testarray)

print("sigmoid grad " , sigmoid_grad(testarray))


testarray = np.array([
    [1],
    [4],
    [7]
])

sum = np.sum(testarray, axis=0)
print("sum : ", sum)
print("sum : ", sum.shape)

testarray2 = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

testarray3 = np.array([
    [1,4,7]
])

print("minus : ", testarray2-testarray)
print("minus : ", testarray2-testarray3)

testarray2 = np.array([
    [2,4,8],
    [16,32,64],
    [128,256,512]
])

testarray3 = np.array([
    [1,2,4]
])

print("divide : ", testarray2 / testarray3)