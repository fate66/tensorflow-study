import tensorflow as tf
import json
import os


def load_img(dir, outDir, name):
    image = tf.keras.preprocessing.image.load_img(dir + name)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    # input_arr = np.array([input_arr])  # Convert single image to a batch.
    out = tf.image.resize_with_pad(
        image=input_arr, target_height=400, target_width=400)
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    tf.keras.utils.save_img(outDir + name, x=out)


def down(list, dir):
    for index, url in enumerate(list):
        print(f"处理第{index}张")
        path_to_downloaded_file = tf.keras.utils.get_file(
            fname=str(index)+'.jpg',
            origin=url,
            cache_subdir=dir,
            cache_dir='/Users/ft/tensorflow-study/data/'
        )
        root = '/Users/ft/tensorflow-study/data/'
        load_img(root + dir+'/', root + 'datasets/' +
                 dir + '/', str(index)+'.jpg')


# 油画
def painting8000():
    with open("./data/painting8000.json", 'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'paint')

# 瓷器


def porcelain10000():
    with open("./data/porcelain10000.json", 'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'porcelain')


# 珠宝
def jewellery5000():
    with open("./data/jewellery5000.json", 'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'jewellery')

        # 酒


def alcohol8000():
    with open("./data/alcohol8000.json", 'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'alcohol')

# 玉


def jade10000():
    with open("./data/jade10000.json", 'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'jade')

# 茶叶


def tea70_2():
    with open("./data/tea70_2.json", 'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'tea')

# 古籍


def ancient10000_2():
    with open("./data/ancient10000_2.json", 'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'ancient')

# 钟表


def clocks1300_2():
    with open("./data/clocks1300_2.json", 'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'clocks')


# 钱币


def coin8000_2():
    with open("./data/coin8000_2.json", 'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'coin')


# 钱币


def furniture1300_2():
    with open("./data/furniture1300_2.json", 'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'furniture')

        # 钱币


def philatelic10000_2():
    with open("./data/philatelic10000_2.json", 'r') as load_f:
        load_dict = json.load(load_f)
        down(load_dict, 'philatelic')






# 瓷器2次分类

# 下载瓷器分类
def down2philatelicAll(data, base_directory):
    for index, obj in enumerate(data):
        print(f"处理lot:{index}")
        for index, url in enumerate(obj['list']):

            path_to_downloaded_file = tf.keras.utils.get_file(
                fname=str(index)+'.jpg',
                origin= url,
                cache_subdir=str(obj['lot']),
                cache_dir=base_directory
            )
            load_img(base_directory +str(obj['lot']) + '/'  , base_directory + 'datasets/' + str(obj['lot']) + '/', str(index)+'.jpg')

def philatelicAll():
    with open("./data2/philatelicAll.json", 'r') as load_f:
        data = json.load(load_f)
        down2philatelicAll(data, '/Volumes/My Passport/tensorflow-study/data2/')

philatelicAll()

# jewellery5000()
# jade10000()
# tea70_2()
# ancient10000_2()
# clocks1300_2()
# coin8000_2()
# furniture1300_2()
# philatelic10000_2()
# porcelain10000()
# painting8000()
# alcohol8000()
# print(path_to_downloaded_file)


def test():
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
    print('batch:', sample[0].shape, sample[1].shape,
          tf.reduce_max(sample[0]), tf.reduce_min(sample[0]))
    # print('batch:', sample[0], sample[1])
# test()


def check(name):
    if name == '.DS_Store':
        return False
    return True

# 判断两个tensor的值是否相等


def tensor_equal(a, b):
    # 判断形状相等
    if a.shape != b.shape:
        return False
    # 逐值对比后若有False则不相等
    if not tf.reduce_min(tf.cast(a == b, dtype=tf.int32)):
        return False
    return True


def sortEqual():
    #  tea
    root = 'data/datasets/tea/'
    files = []
    for fname in os.listdir(root):
        if check(fname):
            image = tf.keras.preprocessing.image.load_img(
                root + fname)
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            files.append({'name': fname, 'arr': input_arr})
    temp = []
    print('文件加载完成')
    for index, obj in enumerate(files):
        flag = False
        print('文件开始对比' + str(index))
        for index, tem in enumerate(temp):
            # print(obj['arr'])
            if tensor_equal(obj['arr'], tem[0]['arr']):
                tem.append(obj)
                flag = True
                break
        if flag == False:
            temp.append([obj])
    print('文件对比完成')
    for index, tem in enumerate(temp):
        if len(tem) > 1:
            s = ''
            for index, sub in enumerate(tem):
                s += (sub['name'] + '--')
            print(s)


# sortEqual()
