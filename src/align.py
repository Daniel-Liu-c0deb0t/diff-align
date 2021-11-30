import tensorflow as tf

GAP = "$"
ALPHA = ["A", "C", "G", "T", "$"]

def align(y, t, gamma, epsilon = 1.0):
    n = t.shape[0]
    m = y.shape[0]
    dp = tf.TensorArray(tf.float32, size = (n + 1) * (m + 1), clear_after_read = False)

    def idx(i, j):
        return j * (n + 1) + i

    for j in range(m + 1):
        for i in range(n + 1):
            if i == 0 and j == 0:
                dp = dp.write(idx(i, j), 0.0)
            elif i == 0:
                dp = dp.write(idx(i, j), float("inf"))
            elif j == 0:
                dp = dp.write(idx(i, j), dp.read(idx(i - 1, j)) + gamma)
            else:
                s = tf.stack([
                    dp.read(idx(i - 1, j - 1)) + cost(t[i - 1], y[j - 1]),
                    dp.read(idx(i, j - 1)) + cost(one_hot_char(GAP), y[j - 1]),
                    dp.read(idx(i - 1, j)) + gamma,
                ])

                if epsilon == 0.0:
                    dp = dp.write(idx(i, j), tf.reduce_min(s))
                else:
                    dp = dp.write(idx(i, j), epsilon * -tf.math.reduce_logsumexp(-s / epsilon))

    dp = tf.transpose(tf.reshape(dp.stack(), (m + 1, n + 1)))
    return dp, dp[n][m]

def cost(t, y):
    l = tf.losses.categorical_crossentropy(t, y)
    return l

def one_hot_str(s):
    a = [ALPHA.index(c) for c in s]
    return tf.one_hot(a, len(ALPHA))

def one_hot_char(c):
    return tf.one_hot(ALPHA.index(c), len(ALPHA))

def prob_to_str(y):
    a = tf.argmax(y, axis = 1).numpy()
    return "".join((ALPHA[i] for i in a))
