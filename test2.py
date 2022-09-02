# 逻辑回归

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

def preprocess(x, y) :
    # x 数据集可以调整为 [-1, 1]
    # x = 2 * tf.cast(tf.convert_to_tensor(x), dtype=tf.float32) / 255. - 1.
    x = tf.cast(tf.convert_to_tensor(x), dtype=tf.float32) / 255.
    y = tf.cast(tf.convert_to_tensor(y), dtype=tf.int32)
    return x, y


#[50k,32,32,3]  [10k,1]
(x, y), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()

# 压缩维度 [50k]
y = tf.squeeze(y)
# [10k]
# y_val = tf.squeeze(y_val)
#[10k, 10]
# y = tf.one_hot(y, depth=2)
# y_val = tf.one_hot(y_val, depth=1) # [10k, 10]
y_val = []
y = []
for num in range(0,50000): 
    y.append([1, 0])
for num in range(0,10000): 
    y_val.append([1, 0])

y_val = tf.convert_to_tensor(y_val)
y = tf.convert_to_tensor(y)
print('datasets:', x.shape, y.shape, x_val.shape, y_val.shape)

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.map(preprocess).shuffle(10000).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(128)


sample = next(iter(train_db))
print('batch:', sample[0].shape, sample[1].shape)
# print('batch_val', sample[1])


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
    # 逻辑回归 
    layers.Dense(2, activation=tf.nn.sigmoid)

]
model = Sequential(conv_layers)
model.build(input_shape=[None, 32, 32, 3])
model.summary()

# optimizer 梯度优化 1e-3  或者 1e-2
# loss 
# metrics  实时查看测试的正确率
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
#                 loss=tf.losses.CategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy']
#                 )
# history = model.fit(train_db, epochs=5, validation_data=test_db, validation_freq=1)

# 再次测试模型
# model.evaluate(test_db)
#保存 训练时使用
# network.save_weights('ckpt/weights.ckpt')
# tf.saved_model.save(model, './model')
# model.save('my_model')
# print('saved to my_model')

# reconstructed_model = keras.models.load_model("my_model")

# test_input = tf.ones([1,32,32,3])

# Let's check:
# print(model.predict(test_input), reconstructed_model.predict(test_input))


# print(history)

optimizer = optimizers.Adam(1e-3)

for epoch in range(5):

    for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                # [128, 32, 32, 3] => [128,  10]
                logits = model(x)
                loss = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss,  model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, 'loss:',  float(loss))
    total_num = 0
    total_correct = 0
    for x, y in test_db:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            # prob2 = tf.nn.sigmoid(logits)
            # print('prob2:', prob, prob2)
            print(prob.shape)
            pred = tf.argmax(prob, axis=1)
            print(pred.shape)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_num += x.shape[0]
            total_correct += int(correct)
    acc = total_correct / total_num
    print(epoch, 'acc', acc)


# model = Sequential.