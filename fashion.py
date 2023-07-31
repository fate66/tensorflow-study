import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

# 

def preprocess(x, y):

    x = tf.convert_to_tensor(x)
    x = tf.cast(x, dtype=tf.float32)
    x = x / 255.
    y = tf.convert_to_tensor(y)
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape, y.shape)

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(128)


db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(128)

db_iter = iter(db)
sample = next(db_iter)
print(sample[0].shape, sample[1])



model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)
])

model.build(input_shape=[None, 28*28])

model.summary()
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# summary_writer = tf.summary.create_file_writer('logs/'+current_time)

lr=1e-3
# lr=0.0001 梯度 优化
optimizer = optimizers.Adam(lr)

def main() :
    for epoch in range(30):
        for step, (x, y) in enumerate(db):
            x = tf.reshape(x, [-1, 28 * 28])
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss_ce = tf.reduce_mean(loss_ce)
            grads = tape.gradient(loss_ce, model.trainable_variables)
            # 梯度优化器
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0 :
                print(epoch, step, "loss:", float(loss_ce), "loss2:", float(loss_mse))
        total_correct = 0
        total_num = 0
        for x,y in db_test:
             x = tf.reshape(x, [-1, 28 * 28])
             logits = model(x)
             prob = tf.nn.softmax(logits, axis=1)
             pred = tf.argmax(prob, axis=1)
             pred = tf.cast(pred, dtype=tf.int32)
             correct = tf.equal(pred, y)
             correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
             total_correct += int(correct)
             total_num += x.shape[0]
        acc = total_correct / total_num
        print(epoch, '正确率：', acc)
    return

if __name__ == '__main__':
    main()    