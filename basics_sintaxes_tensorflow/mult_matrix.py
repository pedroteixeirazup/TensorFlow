import tensorflow as tf

a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[-1, 3], [4, 2]])

multiplicacao = tf.matmul(a, b)

with tf.Session() as sess:
    print(sess.run(a))
    print('\n')
    print(sess.run(b))
    print('\n')
    print(sess.run(multiplicacao))