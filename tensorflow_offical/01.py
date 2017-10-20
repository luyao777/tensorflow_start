
'''
Python 程序生成了一些三维数据, 然后用一个平面拟合它.
'''
import tensorflow as tf
import numpy as np
# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300
# 构造一个线性模型
#
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b
# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
# 启动图 (graph)
sess = tf.Session()
sess.run(init)
# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
# 得到最佳拟合结果 W: [[0.100 0.200]], b: [0.300]
'''
 [[ 0.34953234  0.52099639]] [-0.01467655]
20 [[ 0.19349834  0.31485686]] [ 0.19103672]
40 [[ 0.13384378  0.24076653]] [ 0.2609885]
60 [[ 0.11217624  0.2145327 ]] [ 0.28603593]
80 [[ 0.1043684  0.2051914]] [ 0.2950021]
100 [[ 0.10156518  0.20185629]] [ 0.29821128]
120 [[ 0.10056044  0.20066407]] [ 0.29935983]
'''