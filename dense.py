import tensorflow as tf
from tensorflow import keras

  # 定义一个神经节点，输出为 [3, 3]
def dense1 () :
    # 实现公式： out = X@W1+b1
    x = tf.random.normal([3, 784])
    # 定义一个神经层，输出为 [3, 3]
    net = tf.keras.layers.Dense(3) 
    out = net(x)
    # 输出的结构 [3, 3]
    print(out, out.shape)
     # net.keras.shape 代表的是 w [784, 3] ， net.bias.shape代表的是b [3]
    print(net.kernel.shape, net.bias.shape)
    return

   # 定义多个神经节点
def dense2 () :
    # 实现公式：relu(relu(X@W1+b1)@W2+b2) 
    # 定义多个神经节点
    model = tf.keras.Sequential([
        keras.layers.Dense(2, activation='relu'),
        keras.layers.Dense(2, activation='relu'),
        keras.layers.Dense(2),
        ])
    model.build(input_shape=[None, 3])
    for p in model.trainable_variables:
        print(p.name, p.shape)
    return


dense2()