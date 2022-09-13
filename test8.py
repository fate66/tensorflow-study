#  10 个时尚类别的 60,000 个 28x28 灰度图像的数据集，以及一个包含 10,000 个图像的测试集
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
tf.random.set_seed(2345)

def load_img(type):
    train_db = tf.keras.utils.image_dataset_from_directory(
        directory='/data/tf-python/data/datasets',
        label_mode='categorical',
        image_size=(400, 400),
        batch_size=64,
        seed=1024,
        subset=type,
        validation_split=0.2,
        follow_links=True)  
    normalization_layer = layers.Rescaling(1./255)
    return train_db.map(lambda x, y: (normalization_layer(x), y))


#[b,400,400,3]  [b, 5]
train_db = load_img('training')
test_db = load_img('validation')

sample = next(iter(test_db))
print('batch:', sample[0].shape, sample[1].shape)
# print('batch_val', sample[0], sample[1])

conv_layers = [
    #unit 1 [b, 400,400,3] -> [b, 200, 200, 64]
    layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation=tf.nn.relu),
    layers.Dropout(0.5),
     #unit 1 [b, 200, 200, 64] -> [b, 100, 100, 64]
    layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
    #unit 2  [b, 100, 100, 64] -> [b, 50, 50, 128]
    layers.Conv2D(128, kernel_size=3, strides=2,padding="same", activation=tf.nn.relu),
    # layers.Dropout(0.5),
    # 降维， 会将输入的size缩小2，  [b, 50, 50, 128] -> [b, 25, 25, 128]
    layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

    #unit 5  [b,25, 25, 128] -> [b, 5, 5, 256]
    layers.Conv2D(256, kernel_size=3, strides=5, padding="same", activation=tf.nn.relu),
    layers.Dropout(0.5),
    # 降维， 会将输入的size缩小2， [b,  2, 2, 256] -> [b,  1, 1, 256]
    layers.MaxPool2D(pool_size=2, strides=5, padding='same'),
    # 摊平
    layers.Flatten(),
    layers.Dense(128, activation=tf.nn.relu),

    layers.Dropout(0.5),
    # 最后一个节点的输出必须和分类一致
    # 逻辑回归 
    layers.Dense(5)

]
model = Sequential(conv_layers)
model.build(input_shape=[None, 400, 400, 3])
model.summary()

# optimizer 梯度优化 1e-3  或者 1e-2
# loss 
# metrics  实时查看测试的正确率
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )

# model.load_weights('ckpt/weights.ckpt')

history = model.fit(train_db, epochs=50, validation_data=test_db, validation_freq=1)

# 再次测试模型
model.evaluate(test_db)
#保存 训练时使用
model.save_weights('ckpt8/weights.ckpt')
# tf.saved_model.save(model, './model')
# model.save('my_model')
# print('saved to my_model')

# reconstructed_model = keras.models.load_model("my_model")

# test_input = tf.ones([1,32,32,3])

# Let's check:
# print(model.predict(test_input), reconstructed_model.predict(test_input))


# print(history)
