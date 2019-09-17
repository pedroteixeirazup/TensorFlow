import tensorflow as tf 


# a = tf.add(2, 2, name = 'add')
# b = tf.multiply(a, 3, name = 'mult1')
# c = tf.multiply(b, a, name = 'mult2')

tf.reset_default_graph()

with tf.name_scope('Operacoes'):
    with tf.name_scope('Escopo_A'):
        a = tf.add(2, 2, name = 'add')
    with tf.name_scope('Escopo_B'):
        b = tf.multiply(a, 3, name = 'mult1')
        c = tf.multiply(b, a, name = 'mult2')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('output', sess.graph)
    print(sess.run(c))
    write.close()