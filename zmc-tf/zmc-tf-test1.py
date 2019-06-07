#%% [markdown]
# # Tensorflow ZMCintegral Tests
# 
# Testing of the tensorflow version of ZMCintegral.

import math
import time
import numpy as np
import tensorflow as tf
import ZMCintegral

def testfoo(x):
    return tf.math.sin(x[0]+x[1]+x[2]+x[3])

MC = ZMCintegral.MCintegral(testfoo, [[0,2], [1,4], [2,3], [4,6]])

start = time.time()
result = MC.evaluate()
end = time.time()

print('=====================================')
print('Result = ', result[0])
print(' Error = ', result[1])
print('=====================================')
print('Time = ', end-start, 'seconds')
