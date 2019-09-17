from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity (tf.compat.v1.logging.ERROR)

mnist = input_data.read_data_sets('mnist/', one_hot = True)

X_treinamento = mnist.train.images
Y_treinamento = mnist.train.labels
X_teste = mnist.test.images
Y_teste = mnist.test.labels

# X_treinamento.shape
# Y_treinamento.shape


# plt.imshow(X_treinamento[102].reshape((28,28)), cmap = 'gray')
# plt.title('Classe: ' + str(np.argmax(Y_treinamento[102])))
# plt.show()

X_batch, Y_batch = mnist.train.next_batch(64)
# X_batch.shape

neuronios_entrada = X_treinamento.shape[1]
# neuronios_entrada

neuronios_oculta1 = int((X_treinamento.shape[1] + Y_treinamento.shape[1]) / 2)
# print(neuronios_oculta)
neuronios_oculta2 = neuronios_oculta1
neuronios_oculta3 = neuronios_oculta1
neuronios_saida =  Y_treinamento.shape[1]

#784 -> 397 -> 397 -> 397 -> 10

#pesos
W = {
    'oculta1': tf.Variable(tf.random_normal([neuronios_entrada, neuronios_oculta1])), #784*397
    'oculta2': tf.Variable(tf.random_normal([neuronios_oculta1, neuronios_oculta2])), #397*397
    'oculta3': tf.Variable(tf.random_normal([neuronios_oculta2, neuronios_oculta3])), #397*397
    'saida': tf.Variable(tf.random_normal([neuronios_oculta3, neuronios_saida]))
    }

b = { #BIAS
    'oculta1': tf.Variable(tf.random_normal([neuronios_oculta1])),
    'oculta2': tf.Variable(tf.random_normal([neuronios_oculta2])),
    'oculta3': tf.Variable(tf.random_normal([neuronios_oculta3])),
    'saida': tf.Variable(tf.random_normal([neuronios_saida]))
    }

xph = tf.placeholder('float', [None, neuronios_entrada])
yph = tf.placeholder('float', [None, neuronios_saida])

def mlp(x, W, bias):
    camada_oculta1 = tf.nn.relu(tf.add(tf.matmul(x, W['oculta1']), bias['oculta1']))
    camada_oculta2 = tf.nn.relu(tf.add(tf.matmul(camada_oculta1, W['oculta2']), bias['oculta2']))
    camada_oculta3 = tf.nn.relu(tf.add(tf.matmul(camada_oculta2, W['oculta3']), bias['oculta3']))
    camada_saida = tf.add(tf.matmul(camada_oculta3, W['saida']), bias['saida'])
    return camada_saida

modelo= mlp(xph, W,b)
erro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = modelo, labels = yph))
otimizador = tf.train.AdadeltaOptimizer(learning_rate = 0.0001).minimize(erro)

previsoes = tf.nn.softmax(modelo)
previsoes_corretas = tf.equal(tf.argmax(previsoes, 1), tf.argmax(yph, 1))
taxa_acerto = tf.reduce_mean(tf.cast(previsoes_corretas, tf.float32))

#AJUSTE DOS PESOS
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoca in range(5000):
        X_batch, Y_batch = mnist.train.next_batch(128)
        _, custo = sess.run([otimizador, erro], feed_dict = {xph: X_batch, yph: Y_batch})
        if epoca % 100 == 0: 
            acc = sess.run([taxa_acerto], feed_dict = {xph: X_batch, yph: Y_batch})
            print('Época: ' + str((epoca + 1)) + ' erro: ' + str(custo) + ' acc: ' + str(acc))

    print('Treinamento concluido')
    print(sess.run(taxa_acerto, feed_dict = {xph: X_teste, yph: Y_teste}))

print('ALOOO')
tf.compat.v1.logging.set_verbosity(old_v)