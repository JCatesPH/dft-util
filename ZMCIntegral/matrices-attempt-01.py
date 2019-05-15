#!/home/jmcates/miniconda3/envs/zmcint/bin/python
# coding: utf-8

# # Proper interpreter:
# /share/apps/python_shared/3.6.5/bin/python

# # Testing example from github to see ZMCIntegral is working correctly.
# https://github.com/Letianwu/ZMCintegral
#
# It is the integration of this function:
# https://github.com/Letianwu/ZMCintegral/blob/master/examples/example01.png?raw=true


import math
from numba import cuda
import ZMCintegral
import time
import numpy as np
