import math
import numpy as np
# from numba import cuda
import ZMCIntegral
import numba

@numba.cuda.jit(device=True)
def testfoo(x):
	f = numba.cuda.shared.array(10,dtype=numba.types.float32)
	for i in range(0,10):
		f[i] = math.sin(x[0]+x[1])
	return f[0]


x = np.linspace(0,10,1)
print('Creating ZMCintegral object')
MC = ZMCIntegral.MCintegral(testfoo, [[0,2],[0,1]])
print('====================================================')
print('Evaluating..')
results = MC.evaluate()

print('result =', result[0])
