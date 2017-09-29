import tensorflow as tf 
import numpy as np 

def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)
    return output

x_data = np.linspace(-1,1,300, dtype = np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise


xs = tf.placeholder(tf.float32, [None,1])
ys = tf.placeholder(tf.float32, [None,1])


l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

ini = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(ini)
    for step in range(1000):
        sess.run(train_step, feed_dict = {xs:x_data, ys:y_data}) #小批量训练，提升效率
        if step % 50 == 0:
            print(step, sess.run(loss, feed_dict = {xs:x_data, ys:y_data}))