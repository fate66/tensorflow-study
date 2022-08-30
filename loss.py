# 损失函数

import tensorflow as tf
from tensorflow import keras


# MSE函数，均方差  loss=（真实值-输出值）平方 相加
# 模拟真实y, 一共5行数据源，one-hot编码为4个种类
y = tf.constant([1,2,3,0,2])
y = tf.one_hot(y, 4)
y = tf.cast(y, dtype=tf.float32)
print(y)

# 模拟神经层输出值 5行 4列
out = tf.random.normal([5, 4]) 

loss = tf.reduce_mean(tf.losses.MSE(y, out))
print(loss)

# 交叉熵，也是一种loss函数，主要也是为了衡量真实y与推测y之间 的距离
# 第一个参数是真实y，第二个参数是预测出来的y的概率。因为预测的值每个都是0.25， 所以这个loss值很大,
# 预测值最好直接来自于 logits， 如果预测值来自于 经过softmax处理过的 logits，那么 from_logits 为false
# 但是不推荐这种做法，据说数据会不稳定
tf.losses.categorical_crossentropy([0,1,0,0], [0.25,0.25,0.25,0.25], from_logits=True)
# 第一个参数是真实y，第二个参数是预测出来的y的概率。因为预测的值第三个是0.8，并且预测正确， 所以这个loss值很小
tf.losses.categorical_crossentropy([0,0,1,0], [0,0,0.8,0.2])



# 分类问题，优先选用交叉熵，再用MSE

