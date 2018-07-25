# encoding:utf-8
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.c_[a,b]

print(np.r_[a,b])
print(c)
print(np.c_[c,a])
print(np.abs(0-1))
A = np.arange(1,5).reshape(2,2)
B = np.mat(A)
print("%d Times, Cost is %f" % (1, 0.05))