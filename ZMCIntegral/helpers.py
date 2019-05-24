import math
import time
import numpy as np
from numba import cuda

# Helper function to set qx
def setqx(qxi):
	global qx
	qx = qxi
	return

# Helper function to get qx
@cuda.jit(device=True)
def getqx():
    return qx

# Helper function to set qx
def setr(ri):
	global r
	r = ri
	return

# Helper function to get qx
@cuda.jit(device=True)
def getr():
    return r

@cuda.jit(device=True)
def my_heaviside(z):
    # Wrote this Heaviside expression with it cast in cuda to avoid error below.
    if z <= 0 :
	    return 0
    else :
	    return 1

@cuda.jit(device=True)
def my_Besselv(v, z):
    # WILL NOT WORK IF v IS NOT AN INTEGER
    # Conditional to handle case of negative v.
    if(v < 0):
        v = abs(v)
        resultsign = (-1) ** v
    else:
        resultsign = 1
    result = 0
    # Loop to construct Bessel series sum.
    for n in range(0,20):
        sign = (-1)**n
        exp = 2 * n + v
        term = z ** exp
        r = n + v + 1
        if(r == 0):
            r = 1e-15
        gamma = int(math.gamma(r))
        factorial = int(math.gamma(n+1))
        twoexp = 2 ** exp
        denom = twoexp * factorial * gamma
        term = term / denom * sign
        # print('for ', n, ': ',term)
        result = result + term

    return result * resultsign

@cuda.jit(device=True)
def my_Bessel(z):
    # CHANGE FOR v4: Changing polynomial evaluation for efficiency (SEE Horner's Algorithm). Extending number of terms.
    # Carrying the series to eight terms ensures that the error in the series is < machine_epsilon when z < 1.
    # Approximately: z1 <~ 2.1, z2 <~ 3.33  implies  error <~ 2.15E-6
    val = z**2 / 4 * (-1 + z**2 / 16 * (1 + z**2 / 36 * (-1 + z**2 / 64 * (1 + z**2 / 100 * (-1 + z**2 / 144 * (1 + z**2 / 196 * (-1 + z**2 / 256)))))))
    return val + 1
