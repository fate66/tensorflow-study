#5个图片分类
import tensorflow as tf
import os
import time
# import tensorflowjs as tfjs

tf.random.set_seed(2345)

print(tf.__version__)


def load_img(type):
    train_db = tf.keras.utils.image_dataset_from_directory(
        directory='/Users/ft/tensorflow-study/data/datasets',
        label_mode='categorical',
        image_size=(400, 400),
        batch_size=150,
        seed=1024,
        subset=type,
        validation_split=0.2,
        follow_links=True)
    print(train_db.class_names)

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    return train_db.map(lambda x, y: (normalization_layer(x), y))


def load_one_img(dir):
    image = tf.keras.preprocessing.image.load_img(dir)
    return tf.keras.preprocessing.image.img_to_array(image)


def create_img(dir, name):
    image = tf.keras.preprocessing.image.load_img(dir + name)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    # input_arr = np.array([input_arr])  # Convert single image to a batch.
    out = tf.image.resize_with_pad(
        image=input_arr, target_height=400, target_width=400)
    dirs = path = dir+'resize/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    tf.keras.utils.save_img(dirs+name, x=out)


def defModel():
    #[b,400,400,3]  [b, 5]
    train_db = load_img('training')
    test_db = load_img('validation')

    sample = next(iter(test_db))
    print('batch:', sample[0].shape, sample[1].shape)
    # print('batch_val', sample[0], sample[1])

    conv_layers = [
        # accuracy: 0.9979 - val_accuracy: 0.9183
        # layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation=tf.nn.relu),
        # layers.Conv2D(128, kernel_size=3, strides=2,padding="same", activation=tf.nn.relu),
        # layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, kernel_regularizer=tf.keras.regularizers.l2(
        ), padding="same", activation=tf.nn.relu),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, kernel_regularizer=tf.keras.regularizers.l2(
        ), padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
        tf.keras.layers.Dropout(0.5),

        # layers.Dropout(0.5),
        # [b, 100, 100, 128] -》[b, 50, 50, 128]
        # unit 2  [b, 100, 100, 64] -> [b, 50, 50, 128]
        #
        # 降维， 会将输入的size缩小2，  [b, 50, 50, 128] -> [b, 25, 25, 128]
        # layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

        # unit 5  [b,25, 25, 128] -> [b, 5, 5, 256]
        # layers.Conv2D(256, kernel_size=3, strides=5, padding="same", activation=tf.nn.relu),
        # layers.Dropout(0.5),
        # 降维， 会将输入的size缩小2， [b,  2, 2, 256] -> [b,  1, 1, 256]
        # layers.MaxPool2D(pool_size=2, strides=5, padding='same'),
        # 摊平
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        # tf.keras.layers.Dropout(0.5),
        # 最后一个节点的输出必须和分类一致
        # 逻辑回归
        tf.keras.layers.Dense(5)
    ]
    model = tf.keras.Sequential(conv_layers)
    model.build(input_shape=[None, 400, 400, 3])
    model.summary()

    # optimizer 梯度优化 1e-3  或者 1e-2
    # loss
    # metrics  实时查看测试的正确率
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
                  )

    model.load_weights('ckpt8/weights.ckpt')

    history = model.fit(train_db, epochs=2,
                        validation_data=test_db, validation_freq=1)

    # 再次测试模型
    model.evaluate(test_db)
    # 保存 训练时使用
    model.save_weights('ckpt8/weights.ckpt')
    tf.saved_model.save(model, "model8")
    # model.save('model8')
    # tfjs.converters.save_keras_model(model, 'model8js')

    # print('saved to my_model')

    # print(history)


def predict():
    # db = load_img('training')

    # '/data/tf-python/data/'+alcohol+'/
    # 酒
    dir = 'alcohol'
    # 玉
    dir2 = 'jade'
    # 首饰
    dir3 = 'jewellery'
    # 字画
    dir4 = 'paint'
    # 瓷器
    dir5 = 'porcelain'
    image = load_one_img(
        '/Users/ft/tensorflow-study/data/test/resize/porcelain01.jpeg') / 255.
    image = tf.expand_dims(image, axis=0)
    print('image:', image.shape, tf.reduce_max(image), tf.reduce_min(image))
    start = time.time()
    model = tf.saved_model.load("model8")
    print(model(image))
    print(time.time() - start)


# for name in os.listdir('/data/tf-python/data/test/'):
#     create_img('/data/tf-python/data/test/', name)


defModel()
# predict()
