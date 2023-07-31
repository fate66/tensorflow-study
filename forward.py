import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x, y),(x_test, y_test)  = datasets.mnist.load_data()

x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
x_test = tf.convert_to_tensor(x_test) / 255.
y_test = tf.convert_to_tensor(y_test)
print(x.shape, x.dtype, y.shape, y.dtype)
print(tf.reduce_max(x), tf.reduce_min(x), tf.reduce_max(y), tf.reduce_min(y))

train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
text_db = tf.data.Dataset.from_tensor_slices(x_test, y_test).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)

#根据公式 out = relu(relu(X@W1+b1)@W2+b2) 进行定义整个训练过程
# 输入的值，是根据输入的x来定，中间值随便定，输出值是根据一共有几个分类而定
# 对于以上的分类问题，最初输入值是 784， 最终输出值是 10， 784是像素，10是一共有10类
#  整个过程： 
#  [b, 784] => [b, 256] => [b, 128] => [b, 10]
#  对于w的变化过程，就是从784-》256-》128-》10 这几个范围内
#  中间的值 256 128 是随意定的

w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))  # 输入 784， 输出256
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1)) # 输入 256， 输出128
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1)) # 输入 128， 输出10
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3
for epoch in range(60):
    for step, (x, y) in enumerate(train_db): 
        # x: [128, 28, 28]
        # y: [128]
        # [b, 28, 28] => [b, 28*28] 换形状
        x = tf.reshape(x, [-1, 28 * 28])
        with tf.GradientTape() as tape: # tf.Variable
            # [b, 28*28]@[784,256] + [256] => [b, 256] + [256] =》 [b, 256] + [b , 256]
            # 本质就是在计算 这个公式  h1 = x@w1 + b2
            h1 = x@w1 + b1
            h1 = tf.nn.relu(h1)
            # [b, 256] => [b, 128]
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            # [b, 128] => [b, 10]
            out = h2@w3 + b3

            #  接下来是 计算误差 loss  和梯度下降 
            # out: [b, 10]
            # y: [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10)
            # mse = mean(sum(y-out)^2)
            # [b, 10]
            loss = tf.square(y_onehot - out)
            # mean: scalar
            loss = tf.reduce_mean(loss)
        # compute gradients
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # print(grads)
        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

