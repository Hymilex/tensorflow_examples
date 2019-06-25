# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 08:28:05 2019

@author: Administrator
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

# Step.1 设置 Eager Api
print('Setting Eager mode... ')
tf.enable_eager_execution()
tfe = tf.contrib.eager
# Step.2 定义一个张量
print("Define constant tensors")
a = tf.constant(2)
print('a=%i'%a)
b = tf.constant(3)
print('b=%i'%b)
# Step.3 运行不需要tf.Session的操作运算
print("Running operations, without tf.Session")
c = a + b
print("a+b=%i"%c)
d = a * b
print("a*b=%i"%d)
#Step.4 完全兼容Numpy
print("Mixing operations with Tensors and Numpy Arrays")
a = tf.constant([[2.,1.],
                 [1.,0.]],dtype=tf.float32)
print("Tensor:\n a=%s"%a)

b = np.array([[3.,0.],
              [5.,1.]],dtype=np.float32)
print("NumpyArray:\n b = %s" % b)

c = a + b
print("a+b=%s"%c)

d = a*b
print("a*b=%s"%d)
# Step.5 通过Tensor迭代
print("Iterate through Tensor 'a':")
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])
print("Iterate finished!")
