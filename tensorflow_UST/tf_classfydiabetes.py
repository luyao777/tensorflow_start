import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
print(x_data.shape)
print(y_data.shape)


# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

optimize = tf.train.AdamOptimizer(learning_rate).minimize(cost)


predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


ini = tf.global_variables_initializer()

costlist = []
accuracylist = []

with tf.Session() as sess:
    sess.run(ini)
    for step in range(2001):
        W_,b_,cost_,optimize_,accuracy_ = sess.run([W,b,cost,optimize,accuracy],feed_dict={X:x_data, Y:y_data})
        costlist.append(cost_)
        accuracylist.append(accuracy_)
        if step % 20 == 0:
            print(step,':',accuracy_)

x = [var for var in range(len(costlist))]
plt.plot(x,costlist,label = 'cost')
plt.plot(x,accuracylist,label = 'accuracy')
plt.show()

