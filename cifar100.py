#3层全连接层 和 13层卷积层
# 模型是 100类 60k张图
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

# tf.random.set_seed(2345)

# 定义卷积层，64 为通道数 kernel_size（观察点） 一般为 [1, 1] [3, 3] [5, 5]
# padding same 自动添加0  保证输出 与 输出 的 size相同
conv_layers = [
    #unit 1 [b, 32, 32, 3] -> [b, 32, 32, 64]
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    # [b, 32, 32, 64] -> [b, 32, 32, 64]
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    # 降维， 会将输入的size缩小2，   [b, 32, 32, 64] -> [b, 16, 16, 64]
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    #unit 2  [b, 16, 16, 64] -> [b, 16, 16, 128]
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    # [b, 16, 16, 128] -> [b, 16, 16, 128]
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    # 降维， 会将输入的size缩小2，  [b, 16, 16, 128] -> [b, 8, 8, 128]
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    #unit 3  [b, 8, 8, 128] -> [b,  8, 8, 256]
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    # [b, 8, 8, 256] -> [b,  8, 8, 256]
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    # 降维， 会将输入的size缩小2 [b, 8, 8, 256] -> [b,  4, 4, 256]
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    #unit 4  [b,  4, 4, 256] -> [b,  4, 4, 512]
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    # [b,  4, 4, 512] -> [b,  4, 4, 512]
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    # 降维， 会将输入的size缩小2 [b,  4, 4, 512] -> [b,  2, 2, 512]
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    #unit 5  [b,  2, 2, 512] -> [b,  4, 4, 512]
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    # [b,  2, 2, 512] -> [b,  4, 4, 512]
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    # 降维， 会将输入的size缩小2， [b,  2, 2, 512] -> [b,  1, 1, 512]
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
]


def preprocess (x, y) :
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# (x, y), (x_test, y_test) = datasets.cifar100.load_data()
# x = x[0:500]
# y = y[0:500]
# x_test = x_test[0:100]
# y_test = y_test[0:100]

print(x.shape, y.shape, x_test.shape, y_test.shape)
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(10000).map(preprocess).batch(64)


test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(64)

sample = next(iter(train_db))
print(sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

def main() :
      # 定义卷积层，卷积层主要是 为了将size降维，channel升维 
   conv_net = Sequential(conv_layers)

   # 定义全连接层  主要是为了 将 channel 将维
   fc_net = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        # 最后一个节点的输出必须和分类一致
        layers.Dense(10, activation=tf.nn.relu)
   ])

   conv_net.build(input_shape=[None, 32, 32, 3])
   fc_net.build(input_shape=[None, 512])

   optimizer = optimizers.Adam(1e-3)
#    x = tf.random.normal([4, 32, 32, 3])
#    out = conv_net(x)
#    print(out.shape)
#    [1, 2] + [3, 4] => [1,2,3,4]
   variables = conv_net.trainable_variables + fc_net.trainable_variables
   for epoch in range(50):

        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                # [64, 32, 32, 3] => [64,  1, 1, 512]
                out = conv_net(x)
                out = tf.reshape(out, [-1, 512])
                logits = fc_net(out)
                # y_onehot = tf.one_hot(y, depth=100)
                y_onehot = tf.one_hot(y, depth=10)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
            # if step % 100 == 0:
            print(epoch, step, 'loss:',  float(loss))

        total_num = 0
        total_correct = 0
        for x, y in test_db:
            out = conv_net(x)
            out = tf.reshape(out, [-1, 512])
            logits = fc_net(out)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_num += x.shape[0]
            total_correct += int(correct)
        acc = total_correct / total_num
        print(epoch, 'acc', acc)

if __name__ == '__main__':
   main()