import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import matplotlib.pyplot as plt

def load_img(type):
    train_db = tf.keras.utils.image_dataset_from_directory(
        directory='/data/tf-python/data/datasets',
        label_mode='categorical',
        image_size=(400, 400),
        batch_size=32,
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

# plt.figure() 

# for x in range(32):
   
#     plt.subplot(x%4+1,x%8+1)
#     plt.imshow(sample[0][x])
#     # plt.colorbar()
#     plt.grid(False) 
    
# plt.show()  

baseModel = tf.keras.applications.resnet_v2.ResNet50V2(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=(400, 400, 3),
    pooling=None)

model = keras.Sequential([
        baseModel,
        layers.Flatten(),
        keras.layers.Dense(5, activation='softmax')
    ])    

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
# model.save_weights('ckpt9/weights.ckpt')
# tf.saved_model.save(model, './model')
# model.save('my_model')
# print('saved to my_model')

# reconstructed_model = keras.models.load_model("my_model")

# test_input = tf.ones([1,32,32,3])

# Let's check:
# print(model.predict(test_input), reconstructed_model.predict(test_input))


# print(history)
