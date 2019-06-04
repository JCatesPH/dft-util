import math
import numpy as np
# from numba import cuda
import ZMCIntegral
import numba

@numba.cuda.jit(device=True)
def testfoo(x):
	N = 3
	f = numba.cuda.shared.array(N,dtype=numba.types.float64)
	for i in range(0,N):
		f[i] = math.sin(x[0]+x[1])
	return f[0] + f[N]


x = np.linspace(0,10,1)
print('Creating ZMCintegral object')
MC = ZMCIntegral.MCintegral(testfoo, [[0,2],[0,1]])
print('====================================================')
print('Evaluating..')
results = MC.evaluate()

print('result = ', results[0])
print('error = ', results[1])

print('Complete!')