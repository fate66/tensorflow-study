import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

def preprocess(x, y) :
    # x 数据集可以调整为 [-1, 1]
    x = 2 * tf.cast(tf.convert_to_tensor(x), dtype=tf.float32) / 255. - 1.
    y = tf.cast(tf.convert_to_tensor(y), dtype=tf.int32)
    return x, y


#[50k,32,32,3]  [10k,1]
(x, y), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()

print(y_val)
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


# 定义卷积层，64 为通道数 kernel_size（观察点） 一般为 [1, 1] [3, 3] [5, 5]
# padding same 自动添加0  保证输出 与 输出 的 size相同
conv_layers = [
    #unit 1 [b, 32, 32, 3] -> [b, 16, 16, 64]
    layers.Conv2D(64, kernel_size=[3, 3], strides=2, padding="same", activation=tf.nn.relu),
    # 降维， 会将输入的size缩小2，   [b, 16, 16, 64] -> [b, 8, 8, 64]
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    #unit 2  [b, 8, 8, 64] -> [b, 4, 4, 128]
    layers.Conv2D(128, kernel_size=[3, 3], strides=2,padding="same", activation=tf.nn.relu),
    # 降维， 会将输入的size缩小2，  [b, 4, 4, 128] -> [b, 2, 2, 128]
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    #unit 5  [b, 2, 2, 128] -> [b, 2, 2, 256]
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    # 降维， 会将输入的size缩小2， [b,  2, 2, 256] -> [b,  1, 1, 256]
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    # 摊平
    layers.Flatten(),
    layers.Dense(128, activation=tf.nn.relu),
    # 最后一个节点的输出必须和分类一致
    layers.Dense(10)
]
model = Sequential(conv_layers)
model.build(input_shape=[None, 32, 32, 3])
model.summary()

# optimizer 梯度优化 1e-3  或者 1e-2
# loss 
# metrics  实时查看测试的正确率
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )
model.fit(train_db, epochs=15, validation_data=test_db, validation_freq=1)
