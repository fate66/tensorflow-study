import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import keras_tuner as kt
import json
import os

def load_img(dir, name):
    image = tf.keras.preprocessing.image.load_img(dir + name)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    # input_arr = np.array([input_arr])  # Convert single image to a batch.
    out = tf.image.resize_with_pad(image=input_arr,target_height=400,target_width=400)
    dirs = path=dir+'resize/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    tf.keras.utils.save_img(dirs+name, x=out)


def down(list, dir) :
    for index, url in enumerate(list):
        print(f"处理第{index}张")
        path_to_downloaded_file = tf.keras.utils.get_file(
            fname=str(index)+'.jpg',
            origin=url,
            cache_subdir=dir,
            cache_dir='/data/tf-python/data/'
        )
        load_img('/data/tf-python/data/'+dir+'/', str(index)+'.jpg')



# 油画
def painting8000() :
    with open("./data/painting8000.json",'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'paint')

# 瓷器
def porcelain10000() :
    with open("./data/porcelain10000.json",'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'porcelain')


# 珠宝
def jewellery5000() :
    with open("./data/jewellery5000.json",'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'jewellery')

        # 酒
def alcohol8000() :
    with open("./data/alcohol8000.json",'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'alcohol')

        # 玉
def jade10000() :
    with open("./data/jade10000.json",'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'jade')
# jade10000()

# print(path_to_downloaded_file)

def test() :
    train_db = tf.keras.utils.image_dataset_from_directory(
        directory='/data/tf-python/data/datasets',
        label_mode='categorical',
        image_size=(400, 400),
        batch_size=128,
        seed=1024,
        subset='training',
        validation_split=0.2,
        follow_links=True)
    
    # test_db = tf.keras.utils.image_dataset_from_directory(
    #     directory='/data/tf-python/data/datasets',
    #     label_mode='categorical',
    #     image_size=(400, 400),
    #     batch_size=64,
    #     seed=1024,
    #     subset='training',
    #     validation_split=0.2,
    #     follow_links=True)
    
    # for (x_train, y_train) in enumerate(train_db):
    #     print(tf.is_tensor(x_train))
        # print('datasets:', x_train.shape, tf.reduce_max(x_train), tf.reduce_min(x_train))
    normalization_layer = layers.Rescaling(1./255)
    train_db = train_db.map(lambda x, y: (normalization_layer(x), y))
    sample = next(iter(train_db))
    print('batch:', sample[0].shape, sample[1].shape, tf.reduce_max(sample[0]), tf.reduce_min(sample[0]))
    # print('batch:', sample[0], sample[1])
test()