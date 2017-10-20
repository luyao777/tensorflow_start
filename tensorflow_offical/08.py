import tensorflow as tf 
# Create some variables.
v1 = tf.Variable([2], name="v1")
v2 = tf.Variable([3], name="v2")


# v1 = tf.Variable([1], name="v1")
# v2 = tf.Variable([2] name="v2")

# # Add an op to initialize the variables.
# init_op = tf.global_variables_initializer()
# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()
# # Later, launch the model, initialize the variables, do some work, save the
# # variables to disk.
# with tf.Session() as sess:
#     sess.run(init_op)
#     # Do some work with the model.

#     # Save the variables to disk.
#     save_path = saver.save(sess, "./model.ckpt")
#     print("Model saved in file: ", save_path)

saver = tf.train.Saver()
# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "./model.ckpt")
    print ("Model restored.")
    print(sess.run(v1))