import tensorflow as tf
from align import *

opt = tf.optimizers.Adam(1.0)

train_iter = 10
t = tf.constant(one_hot_str("AATTTCCGG"))
#y_init = tf.ones_like(t)
y_init = tf.constant(one_hot_str("TTTCCCCGG"))
y = tf.Variable(y_init)

for i in range(train_iter):
    with tf.GradientTape() as tape:
        ym = tf.nn.softmax(y)
        _, loss = align(ym, t, gamma = 2.0, epsilon = 0.1)

    print("Loss:", loss.numpy())
    opt.minimize(loss, [y], tape = tape)

ym = tf.nn.softmax(y)
print(ym.numpy())
print(prob_to_str(ym))
