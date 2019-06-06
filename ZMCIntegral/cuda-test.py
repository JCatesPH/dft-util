import math
import time

import numpy as np
import numba

# @numba.cuda.jit(device=True)
def nestedfoo(x):
    tmp = 0
    for i in range(1,21):
        for j in range(1,21):
            for k in range(1,21):
                tmp += math.sin(i*x) * math.sin(j*x) * math.sin(k*x)

    return tmp


nestedfoo(0)

start = time.time()
print('Result:', nestedfoo(5))
end = time.time()

print('Time = ', end-start)

@numba.cuda.jit
def nestedfoo2(x):
    tmp = 0
    for i in range(1,21):
        for j in range(1,21):
            for k in range(1,21):
                tmp += math.sin(i*x) * math.sin(j*x) * math.sin(k*x)

    return tmp


an_array = np.array(1)
an_array[0] = 5
threadsperblock = 16
blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock
nestedfoo2[blockspergrid, threadsperblock](an_array)