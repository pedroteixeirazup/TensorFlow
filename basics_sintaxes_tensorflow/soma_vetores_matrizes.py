import tensorflow as tf 

a = tf.constant([9, 8, 7], name='a')
b = tf.constant([1, 2, 3], name='b')

soma = a + b

with tf.Session() as sess:
    print(sess.run(soma))


a1 = tf.constant([[1, 2, 3], [4, 5, 6]], name='a1')
b1 = tf.constant([[1, 2, 3], [4, 5, 6]], name='b1') 

soma1 = tf.add(a1, b1)

with tf.Session() as sess:
    print(sess.run(soma1))

a2 = tf.constant([[1,2,3], [4,5,6]])
b2 = tf.constant([[1], [2]])
soma2 = tf.add(a2,b2)

with tf.Session() as sess:
    print(sess.run(soma2))