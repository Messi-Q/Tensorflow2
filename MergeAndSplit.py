import numpy as np
import tensorflow as tf

x1 = [[[1, 2, 4, 6, 9]], [[5, 1, 9, 10, 27]]]
x2 = [[[7, 2, 3, 1, 10]], [[6, 8, 2, 5, 17]]]
x3 = [[[10, 12, 13, 11, 10]], [[26, 28, 22, 25, 37]]]

x11 = np.array(x1, dtype='float32')
x22 = np.array(x2, dtype='float32')
x33 = np.array(x3, dtype='float32')

print(x11.shape)

feature2vector = tf.keras.layers.Dense(5, activation=tf.nn.relu, dtype='float32')
mergevec = tf.keras.layers.Concatenate(axis=1)
flattenvec = tf.keras.layers.Flatten()

x1_1 = feature2vector(x11)
x2_1 = feature2vector(x22)
x3_1 = feature2vector(x33)

x = mergevec([x1_1, x2_1, x3_1])
x = flattenvec(x)
vecvalue = x.numpy()

value = np.hsplit(vecvalue, 3)

print(x)
print(value)
