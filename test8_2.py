#  10 个时尚类别的 60,000 个 28x28 灰度图像的数据集，以及一个包含 10,000 个图像的测试集
import tensorflow as tf
from tensorflow import keras
import os
import time
# import tensorflowjs as tfjs
import keras_tuner as kt

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


def defModel(hp):
    hp_units = hp.Int('units', min_value=32, max_value=64, step=32)
    num_filters_top_layer = hp.Choice(
        'num_filters_top_layer', values=[10, 64], default=10)
    filters_layer_1 = hp.Choice(
        'filters_layer_1', values=[64, 128], default=64)

    conv_layers = [
        # accuracy: 0.9979 - val_accuracy: 0.9183
        # layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation=tf.nn.relu),
        # layers.Conv2D(128, kernel_size=3, strides=2,padding="same", activation=tf.nn.relu),
        # layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
        tf.keras.layers.Conv2D(num_filters_top_layer, kernel_size=3, strides=2, kernel_regularizer=tf.keras.regularizers.l2(
        ), padding="same", activation=tf.nn.relu),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters_layer_1, kernel_size=3, strides=2, kernel_regularizer=tf.keras.regularizers.l2(
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
        tf.keras.layers.Dense(units=hp_units, activation=tf.nn.relu),
        # tf.keras.layers.Dropout(0.5),
        # 最后一个节点的输出必须和分类一致
        # 逻辑回归
        tf.keras.layers.Dense(5)
    ]
    model = tf.keras.Sequential(conv_layers)
    model.build(input_shape=[None, 400, 400, 3])
    model.summary()

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
                  )

    # model.load_weights('ckpt8/weights.ckpt')
    return model


def start():
    #[b,400,400,3]  [b, 5]
    train_db = load_img('training')
    test_db = load_img('validation')

    sample = next(iter(test_db))
    print('batch:', sample[0].shape, sample[1].shape)
    # print('batch_val', sample[0], sample[1])
    tuner = kt.Hyperband(defModel,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='intro_to_kt')

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)
    tuner.search(train_db, epochs=5, validation_data=test_db,
                 validation_freq=1, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

#  history = model.fit(train_db, epochs=500, validation_data=test_db, validation_freq=1)

    print('units:', best_hps.get('units'),
          'learning_rate:', best_hps.get('learning_rate'))
    model = tuner.hypermodel.build(best_hps)
    # model.load_weights('ckpt6/weights.ckpt')

    history = model.fit(train_db, epochs=5,
                        validation_data=test_db, validation_freq=1)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    # 再次测试模型
    # model.evaluate(test_db)
    # 保存 训练时使用
    model.save_weights('ckpt8/weights.ckpt')
    model.save('model8')
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
    model = keras.models.load_model("model8")
    print(model.predict(image))
    print(time.time() - start)


# for name in os.listdir('/data/tf-python/data/test/'):
#     create_img('/data/tf-python/data/test/', name)


start()
# predict()
