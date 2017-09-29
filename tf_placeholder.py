import tensorflow as tf 
import numpy as np
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

ini = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(ini)
  #print(sess.run(y))  # ERROR: will fail because x was not fed.
  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.