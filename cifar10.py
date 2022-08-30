#自定义model层和layer层，自定义模型和神经节点
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

def preprocess(x, y) :
    # x 数据集可以调整为 [-1, 1]
    x = 2 * tf.cast(tf.convert_to_tensor(x), dtype=tf.float32) / 255. - 1.
    y = tf.cast(tf.convert_to_tensor(y), dtype=tf.int32)
    return x, y


#[50k,32,32,3]  [10k,1]
(x, y), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
# 压缩维度 [50k]
y = tf.squeeze(y)
# [10k]
y_val = tf.squeeze(y_val)
#[10k, 10]
y = tf.one_hot(y, depth=10)
y_val = tf.one_hot(y_val, depth=10) # [10k, 10]
print('datasets:', x.shape, y.shape, x_val.shape, y_val.shape, x.min(), x.max())

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.map(preprocess).shuffle(10000).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(128)



sample = next(iter(train_db))
print('batch:', sample[0].shape, sample[1].shape)


class MyDense(layers.Layer):

    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_weight('w', [inp_dim, outp_dim])
        # self.bias = self.add_variable('w', [outp_dim])

    def call(self, inputs, trainiint=None):

        return inputs @ self.kernel

class MyNetwork(tf.keras.Model):
    # 定义model
    def __init__(self):
        super(MyNetwork, self).__init__()

        self.fc1 = MyDense(32*32*3, 256)
        self.fc2 = MyDense(256, 256)
        self.fc3 = MyDense(256, 256)
        self.fc4 = MyDense(256, 256)
        self.fc5 = MyDense(256, 10)

    def call (self, inputs, training=None):
        x = tf.reshape(inputs, [-1, 32*32*3])
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        return self.fc5(x)



network = MyNetwork()
lr=1e-3
# optimizer 梯度优化 1e-3  或者 1e-2
# loss 
# metrics  存储的正确率
network.compile(optimizer=tf.keras.optimizers.Adam(lr),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )
# train_db 数据源
# ecochs 迭代几次
# validation_data 测试数据集
# validation_freq ，每训练一次，测试一次
network.fit(train_db, epochs=15, validation_data=test_db, validation_freq=1)


# 再次测试模型
network.evaluate(test_db)
#保存
network.save_weights('ckpt/weights.ckpt')
del network
print('saved to ckpt/weights.ckpt')


network = MyNetwork()
network.compile(optimizer=optimizers.Adam(lr),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
                #加载
network.load_weights('ckpt/weights.ckpt')
print('loaded weights from file.')
network.evaluate(test_db)