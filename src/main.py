import tensorflow as tf
from align import *

opt = tf.optimizers.Adam(1.0)

train_iter = 10
t_str = "AATTTCCGG"
y_str = "TTTCCCCGG"

print("Target:", t_str)
print("Initial:", y_str)

t = tf.constant(one_hot_str(t_str))
#y_init = tf.ones_like(t)
y_init = tf.constant(one_hot_str(y_str))
y = tf.Variable(y_init)

for i in range(train_iter):
    with tf.GradientTape() as tape:
        ym = tf.nn.softmax(y)
        _, loss = align(ym, t, gamma = 2.0, epsilon = 0.1)

    print("Loss:", loss.numpy())
    opt.minimize(loss, [y], tape = tape)

ym = tf.nn.softmax(y)
print(ym.numpy())
print("Final:", prob_to_str(ym))
