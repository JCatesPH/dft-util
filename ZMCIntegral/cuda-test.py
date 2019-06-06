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



# Create the data array - usually initialized some other way
data = numpy.ones(256)

# Set the number of threads in a block
threadsperblock = 32 

# Calculate the number of thread blocks in the grid
blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock

# Now start the kernel
nestedfoo2[blockspergrid, threadsperblock](data)

# Print the result
print(data)