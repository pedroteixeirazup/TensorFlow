import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity (tf.compat.v1.logging.ERROR)

mnist = input_data.read_data_sets('mnist/', one_hot = False)

X_treinamento = mnist.train.images
Y_treinamento = mnist.train.labels
X_testes = mnist.test.images
Y_testes = mnist.test.labels

print()

Y_treinamento = np.asarray(Y_treinamento,dtype = np.int32)
Y_testes = np.asarray(Y_testes, dtype = np.int32)
# plt.imshow(X_treinamento[2].reshape(28,28))
# plt.title('Classe: ' + str(Y_treinamento[2]))
# plt.show()

def cria_rede(features, labels, mode):
    #batch_size, largura, altura, canais
    entrada = tf.reshape(features['X'], [-1, 28, 28, 1])

    #recebe [batch_size, 28, 28, 1]
    #retorna [batch_size, 28, 28, 32]
    convolucao1 = tf.layers.conv2d(inputs = entrada, filters = 32, kernel_size = [5,5], activation = tf.nn.relu, padding = 'same')

    #recebe [batch_size, 28, 28, 32]
    #retorna [batch_size, 14, 14, 32]
    pooling1 = tf.layers.max_pooling2d(inputs = convolucao1, pool_size = [2,2], strides = 2)

    #recebe [batch_size, 14, 14, 32]
    #retorna [batch_size, 14, 14, 64]
    convolucao2 = tf.layers.conv2d(inputs = pooling1, filters = 64, kernel_size = [5,5], activation = tf.nn.relu, padding = 'same')

    #recebe [batch_size, 14, 14, 64]
    #retorna [batch_size, 7, 7, 64]
    pooling2 = tf.layers.max_pooling2d(inputs = convolucao2, pool_size = [2,2], strides = 2)

    #recebe [batch_size, 7, 7, 64]
    #retorna [batch_size, 3136]
    flattening = tf.reshape(pooling2, [-1, 7 * 7 * 64])

    # 3126 (entradas) -> 1024 (oculta) -> 10 (saida)
    # recebe [batch_size, 3136]
    # retornar [batchsize, 1024]
    densa = tf.layers.dense(inputs = flattening, units = 1024, activation = tf.nn.relu)

    # dropout
    dropout = tf.layers.dropout(inputs = densa, rate = 0.2, training = mode == tf.estimator.ModeKeys.TRAIN)

    # recebe [batch_size, 1024]
    # retorna [batch_size, 10]
    saida = tf.layers.dense(inputs = dropout, units = 10)

    previsoes = tf.argmax(saida, axis = 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = previsoes)

    erro = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = saida)

    if mode == tf.estimator.ModeKeys.TRAIN:
        otimizador = tf.train.AdamOptimizer(learning_rate = 0.001)
        treinamento = otimizador.minimize(erro, global_step = tf.train.get_global_step()) 
        return tf.estimator.EstimatorSpec(mode = mode, loss = erro, train_op = treinamento)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics_ops = {'accuracy': tf.metrics.accuracy(labels = labels, predictions = previsoes)}
        return tf.estimator.EstimatorSpec(mode = mode, loss = erro, eval_metric_ops = eval_metrics_ops)

classificador = tf.estimator.Estimator(model_fn = cria_rede)

funcao_treinamento = tf.estimator.inputs.numpy_input_fn(x = {'X': X_treinamento}, y = Y_treinamento, batch_size = 128, num_epochs = None, shuffle = True)

classificador.train(input_fn = funcao_treinamento, steps = 200)

funcao_teste = tf.estimator.inputs.numpy_input_fn(x = {'X': X_testes}, y = Y_testes, num_epochs = 1, shuffle = False)

resultados = classificador.evaluate(input_fn = funcao_teste)
print(resultados)

X_imagem_teste = X_testes[2] #Mude aqui para mudar o valor da previsao

X_imagem_teste = X_imagem_teste.reshape(1,-1)
print("Reshape: " + str(X_imagem_teste.shape))

funcao_previsao = tf.estimator.inputs.numpy_input_fn(x = {'X': X_imagem_teste}, shuffle = False)
pred = list(classificador.predict(input_fn = funcao_previsao))

print(pred[0])
plt.imshow(X_imagem_teste.reshape(28,28), cmap='gray')
plt.title('Classe prevista: ' + str(pred[0]))
plt.show()

tf.compat.v1.logging.set_verbosity(old_v)