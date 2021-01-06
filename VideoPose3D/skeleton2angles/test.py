'''
Author: Boris.Peng
Date: 2020-10-21 10:00:27
LastEditors: Boris.Peng
LastEditTime: 2020-10-21 10:57:30
'''
from ctypes import *
import numpy as np

imit = cdll.LoadLibrary('./libske2ang.so')

dataIn = np.array(range(1,52), dtype=c_double)
dataIn = np.log(dataIn)*0.1 + 0.1
dataOut = np.zeros(21, dtype=c_double)
dblP = POINTER(c_double)

a = dataIn.ctypes.data_as(dblP)
b = dataOut.ctypes.data_as(dblP)

imit.skeleton2angles(a,b)

res = []
for i in range(21):
    res.append(b[i])
print(res)