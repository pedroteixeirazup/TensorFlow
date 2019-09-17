import tensorflow as tf 

valor1 =  tf.constant(2)
valor2 = tf.constant(3)

soma  = valor1  + valor2


print(soma)

with tf.Session() as sess:
    s = sess.run(soma)
    
print(s)

texto1 = tf.constant('Texto 1')
texto2 = tf.constant('Texto 2')

with tf.Session() as sess:
    con = sess.run(texto1 + texto2)

print(con)
