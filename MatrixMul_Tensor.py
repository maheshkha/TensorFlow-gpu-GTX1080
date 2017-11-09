# import tensorflow, numpy packages
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

device_name = sys.argv[1]  # Selecting device from cmd line. Options: gpu or cpu
shape = (int(sys.argv[2]), int(sys.argv[2])) # size of input matrix to multiply

# selecting device
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1) # fill random values
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix)) # matrix mul
    sum_operation = tf.reduce_sum(dot_operation)							# reduce sum	


startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print(result)

# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 5)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)

print("\n" * 5)