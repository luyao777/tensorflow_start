'''
为了取回操作的输出内容, 可以在使用Session 对象的run() 调用 执行图时, 传入一些 tensor, 这些 tens
or 会帮助你取回结果
'''
import tensorflow as tf 
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)
with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)
# 输出:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]
