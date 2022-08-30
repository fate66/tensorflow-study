import tensorflow as tf
from tensorflow import keras

a = tf.linspace(-6, 6, 10)
print(a)
# 将分类输出值控制在0到1之间
print(tf.sigmoid(a))



# 一般神经层的最后一层返回的值 被称为 logits，最后一层不加激活函数
logits = tf.random.uniform([1, 10], minval=-1, maxval=2)

# 将分类输出值控制在0到1之间 并且各个类相加为1
prob = tf.nn.softmax(logits, axis=1)
print(prob)
print(tf.reduce_sum(prob, axis=1))


# -1 到 1  用 tf.tanh  