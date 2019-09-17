import tensorflow as tf 

valor1 = tf.constant(15, name = 'valor1')


soma = tf.Variable(valor1 + 5, name='valor1')

# print(soma)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    s = sess.run(soma)


# print(s)

vetor = tf.constant([5, 10, 15], name='vetor')

print(vetor)

soma = tf.Variable(vetor + 5, name='soma')

init1 = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init1)
    print(sess.run(soma))


valor = tf.Variable(0, name='valor')

init2 = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init2)
    for i in range(5):
        valor = valor + 1
        print(sess.run(valor))