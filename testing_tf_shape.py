import numpy as np 
import tensorflow as tf 

x= np.arange(0,36)

x=x.reshape((6,6))

tensor_input=tf.reshape(x,[-1,6,6,1])

pool=tf.layers.max_pooling2d(inputs=tensor_input,pool_size=2,strides=2)

print(pool.get_shape())