
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

# axis 指定在哪个维度进行处理 center 值得是 偏移量，也就是平移 scale值得是  缩放量 ，也就是是否缩小
# trainable 指的是 是否反向传播优化 ， 训练时为true，可以解决 当卷积层堆叠过多时导致的 精度下降问题
# 类似于执行 x = x/w + b   w就是scale，缩放，b就是center 平移
# 使用这个 可以更快的搜索到最优的参数，收敛更快 ，调参时  变化没那么快，参数更加稳定
#主要作用是 将某个维度的数据 趋近于某个范围
#实际应用中，这一层可以使参数调整过程中  参数的变化不会过于敏感

x = tf.random.normal([2,4,4,3], mean=1, stddev=0.5)
net = layers.BatchNormalization(axis=3)
out = net(x, training=True) #训练集 
net.variables

