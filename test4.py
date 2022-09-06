#  10 个时尚类别的 60,000 个 28x28 灰度图像的数据集，以及一个包含 10,000 个图像的测试集
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

def preprocess(x, y) :
    # x 数据集可以调整为 [-1, 1]
    x = 2 * tf.cast(tf.convert_to_tensor(x), dtype=tf.float32) / 255. - 1.
    # x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

#[60k,28,28,3]  [60k]
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_test = tf.squeeze(y_test)
y_train = tf.squeeze(y_train)
y_test = tf.one_hot(y_test, depth=10) # [10k, 10]
y_train = tf.one_hot(y_train, depth=10)

print('datasets:', x_train.shape, y_train.shape, x_test.shape, y_test.shape, tf.reduce_max(x_train), tf.reduce_min(x_train))
print(y_test)

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.map(preprocess).shuffle(10000).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(128)


sample = next(iter(train_db))
print('batch:', sample[0].shape, sample[1].shape)
print('batch_val', sample[0], sample[1])


# 定义卷积层，64 为通道数 kernel_size（观察点） 一般为 [1, 1] [3, 3] [5, 5]
# padding same 自动添加0  保证输出 与 输出 的 size相同
conv_layers = [
    #unit 1 [b, 32, 32, 3] -> [b, 16, 16, 64]
    layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation=tf.nn.relu),
    #unit 2  [b, 16, 16, 64] -> [b, 8, 8, 128]
    layers.Conv2D(128, kernel_size=3, strides=2,padding="same", activation=tf.nn.relu),
    # 降维， 会将输入的size缩小2，  [b, 8, 8, 128] -> [b, 4, 4, 128]
    layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

    #unit 5  [b, 4, 4, 128] -> [b, 2, 2, 256]
    layers.Conv2D(256, kernel_size=3, strides=2, padding="same", activation=tf.nn.relu),
    # 降维， 会将输入的size缩小2， [b,  2, 2, 256] -> [b,  1, 1, 256]
    layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
    # 摊平
    layers.Flatten(),
    layers.Dense(128, activation=tf.nn.relu),
    # 最后一个节点的输出必须和分类一致
    # 逻辑回归 
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
history = model.fit(train_db, epochs=500, validation_data=test_db, validation_freq=1)

# 再次测试模型
# model.evaluate(test_db)
#保存 训练时使用
# model.save_weights('ckpt/weights.ckpt')
# tf.saved_model.save(model, './model')
model.save('my_model')
# print('saved to my_model')

# reconstructed_model = keras.models.load_model("my_model")

# test_input = tf.ones([1,32,32,3])

# Let's check:
# print(model.predict(test_input), reconstructed_model.predict(test_input))


# print(history)
