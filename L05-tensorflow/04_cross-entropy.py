# Solution is available in the other "solution.py" tab
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

log = tf.log(softmax)
sum_arg = tf.multiply(tf.multiply(tf.cast(-1, dtype=tf.float32), one_hot), log)
cross_entropy = tf.reduce_sum(sum_arg)

# Print cross entropy from session
with tf.Session() as sess:
    output = sess.run(cross_entropy, feed_dict={softmax: softmax_data, \
                                                one_hot: one_hot_data})

print(output)
