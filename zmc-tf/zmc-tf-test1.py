#%% [markdown]
# # Tensorflow ZMCintegral Tests
# 
# Testing of the tensorflow version of ZMCintegral.

import math
import time
import numpy as np
import tensorflow as tf
import ZMCintegral

#from tensorflow_on_slurm import tf_config_from_slurm

#cluster, my_job_name, my_task_index = tf_config_from_slurm(ps_number=1)
#cluster_spec = tf.train.ClusterSpec(cluster)
#server = tf.train.Server(server_or_cluster_def=cluster_spec,
#                         job_name=my_job_name,
#                         task_index=my_task_index)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def testfoo(x):
    return tf.sin(x[0]+x[1]+x[2]+x[3])

MC = ZMCintegral.MCintegral(testfoo, [[0,2], [1,4], [2,3], [4,6]])

start = time.time()
result = MC.evaluate()
end = time.time()

print('=====================================')
print('Result = ', result[0])
print(' Error = ', result[1])
print('=====================================')
print('Time = ', end-start, 'seconds')
