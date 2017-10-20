import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function = None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
        with tf.name_scope('Wx_plus_x'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)
        return output

x_data = np.linspace(-1,1,300, dtype = np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None,1], name = 'x_input')
    ys = tf.placeholder(tf.float32, [None,1], name = 'y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function = None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices = [1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

ini = tf.global_variables_initializer()

# #样本画图
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data, y_data)
# plt.ion()
# plt.show()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/",sess.graph)
    sess.run(ini)
    for step in range(1001):
        sess.run(train_step, feed_dict = {xs:x_data, ys:y_data}) #小批量训练，提升效率
        if step % 50 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction,feed_dict={xs:x_data, ys:y_data})
            # print(step, sess.run(loss, feed_dict = {xs:x_data, ys:y_data}))
            
            # lines = ax.plot(x_data, prediction_value,'r-', lw=5)
            # plt.pause(0.1)

