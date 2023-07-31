#  10 个时尚类别的 60,000 个 28x28 灰度图像的数据集，以及一个包含 10,000 个图像的测试集
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import matplotlib.pyplot as plt


# [60k,28,28,3]  [60k]
# [60k,28,28,3]  [60k]
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
# y_test = tf.squeeze(y_test)
# y_train = tf.squeeze(y_train)
# y_test = tf.one_hot(y_test, depth=10) # [10k, 10]
# y_train = tf.one_hot(y_train, depth=10)

print('datasets:', x_train.shape, y_train.shape, x_test.shape, y_test.shape, tf.reduce_max(x_train), tf.reduce_min(x_train))

model = tf.keras.applications.Xception(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(400, 400, 3),
    pooling=None,
    classes=5,
    classifier_activation='softmax')

model.build(input_shape=[None, 400, 400, 3])
model.summary()

# optimizer 梯度优化 1e-3  或者 1e-2
# loss 
# metrics  实时查看测试的正确率
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=tf.losses.CategoricalCrossentropy(),
                metrics=['accuracy']
                )
history = model.fit(train_db, epochs=2, validation_data=test_db, validation_freq=1)

# 再次测试模型
# model.evaluate(test_db)
#保存 训练时使用
model.save_weights('ckpt9-2/weights.ckpt')
# tf.saved_model.save(model, './model')
# model.save('my_model')
# print('saved to my_model')

# reconstructed_model = keras.models.load_model("my_model")

# test_input = tf.ones([1,32,32,3])

# Let's check:
# print(model.predict(test_input), reconstructed_model.predict(test_input))


# print(history)
