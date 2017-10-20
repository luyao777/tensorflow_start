import tensorflow as tf
import matplotlib.pyplot as plt
learning_rate = 0.01

x_train = tf.placeholder(tf.float32,shape = None)
y_train = tf.placeholder(tf.float32,shape = None)

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

h = x_train * W + b
cost = tf.reduce_mean(tf.square(y_train - h))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

Wlist = []
blist = []

with tf.Session() as sess:
    ini = tf.global_variables_initializer()
    sess.run(ini)
    for step in range(2001):
        cost_,W_,b_,_ = sess.run([cost,W,b,train], feed_dict={x_train:[1.,2.,3.], y_train:[2.,4.,6.] })
        if step % 20 == 0:
            print('step', step, '  cost:', cost_,' W:', W_, ' b:', b_)
        Wlist.append(W_)
        blist.append(b_)
#    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
#                 feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})

x = [var for var in range(len(Wlist))]
print(x)
plt.figure("line")
plt.plot(x,Wlist)
plt.plot(x,blist)
plt.show()
