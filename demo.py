import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

from tensorflow.keras import datasets
# import pandas as pd
# import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def fun1 () :
    data = pd.read_csv('./data/income.csv')
    x = data.Education
    y = data.Income
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(x, y, epochs=50000)
    model.predict(pd.Series([20]))
    return


# 变量创建、正态分布（随机数）、创建tensor、数据打乱、根据下标获取数据
def fun1 () :
    # 变量
    a = tf.constant(1.)
    print(a, a.numpy(), a.ndim, a.dtype) # ndim 维度
    b = tf.range([2])
    print(b, b.numpy())
    print(tf.is_tensor(a))

    print('tensor 转换类型')
    aa = np.arange(4)
    aa = tf.convert_to_tensor(aa)
    # 定义一个tensor，并取值
    print('定义一个tensor', aa.dtype, aa.numpy())
    aa = tf.cast(aa, dtype=tf.float32)
    print(aa.dtype)
    w = tf.Variable(aa)
    print(w)
#    定义一个两行三列 值为1 tensor
    tf.convert_to_tensor(np.ones([2, 3]))
    tf.convert_to_tensor(np.zeros([3,3]))

    #  convert_to_tensor 将参数作为数据源
    # 直接定义一个数组 tensor  [2, 3.0]
    tf.convert_to_tensor([2, 3.]) 
    # 直接定义一个二维数组 tensor  2行1列 形状  [2, 1]
    tf.convert_to_tensor([[2], [3.]]) 

    # 以下两步是一样的，都是生成一个和a一样的tensor
    a = tf.zeros([2, 3])
    tf.zeros(a.shape)
    tf.zeros_like(a)

# 将参数作为形状
    tf.zeros([1])
    tf.ones(1)

    # 修改默认填充的值
    tf.fill([2,2], 9)

    # 随机填充数据 正态分布
    tf.random.normal([2, 3], 1, 1)
    # 随机填充，截断分布
    tf.random.truncated_normal([3, 3])

#    0到1之间均匀采样
    tf.random.uniform([2, 2], minval=0, maxval=1)
 
    
    idx = tf.range(2)
    print('idx', idx)
    # 将数据打乱
    idx = tf.random.shuffle(idx)
    print(idx)

    a = tf.random.normal([10, 784])
    # 根据下标获取数据
    print(tf.gather(a, idx))
    return

# 根据下标取值
def fun3 () :
    a = tf.random.normal([4, 28, 28, 3])
    # 根据索引 获取第一张、第三列数据 (28, 3)
    print(a[1, 2].shape)

    #  切片，从下标为2开始截取，到 下标为3结束 某一段数据 (2, 28, 28, 3)
    print(a[1:3].shape)
 #  切片，从下标为倒数第一个开始截取，一直截取到最后一个 一段数据 (1, 28, 28, 3) 冒号后面什么都不写意思就是截取到最后一位
    print(a[-1:].shape)
 #  切片，从下标为0开始截取，一直截取到下标为2之前 一段数据 (2, 28, 28, 3) 冒号前面什么都不写意思就是从0开始截取
    print(a[:2].shape)
     #  切片，从下标为0开始截取，一直截取到下标为最后一个之前 一段数据 (3, 28, 28, 3) 冒号前面什么都不写意思就是从0开始截取
    print(a[:-1].shape)

  #  切片，第一维数组取下标为0，后面几维数组 全取 (28, 28, 3) 冒号前面和后面什么都不写意思就是从0开始截取一直到最后
    print(a[0,:,:,:].shape)

      #  切片，前面3维数组 全取 ，最后一维 取下标为0 (4，28, 28) 冒号前面和后面什么都不写意思就是从0开始截取一直到最后
    print(a[:,:,:,0].shape)

      #  第一维数组全取，第二维和第三维数组从下标0开始一直到28，每隔2位取一次，最后一维数组 全取  (4，14, 14， 3) 连着两个冒号，意思就是第一个冒号前后代表着从哪开始截取，到哪结束，第二个冒号后面的数字代表着步长，隔几位取一次
    print(a[:,0:28:2,0:28:2,:].shape)

    a = tf.range(4)
 #  数组从下标0开始一直到3，倒着取样，相当于将数组倒叙  (3210) 连着两个冒号，意思就是第一个冒号前后代表着从哪开始截取，到哪结束，第二个冒号后面的数字代表着步长，隔几位取一
    print(a[::-1].shape)

    a = tf.random.normal([4, 28, 28, 3])
   # 取第一维数组下标为0的元素，接着后面全取。(28,28,3) ... 三个点意思是能取都取了
    print(a[0, ...].shape)
 # 前三维数组全取。最后一维只取下标为0的元素 (28,28,3) 
    print(a[..., 0].shape)

    # gather 根据下标进行取值，axis就是第几未数组，只能取一个维度，indices就是这个数组中哪几个元素。取a中 第三维数组下标为1,3,5的值，其他维度的全取 （4，28,3,3）
    print(tf.gather(a, axis=2, indices = [1, 3, 5]).shape)

   # gather_nd 根据下标进行取值，但是这个是取多个维度的。
   # indices中下标为0的值为a,取的就是第一维数组中下标为a的元素，indices中下标为1的值为b,取的就是第二维数组中下标为b的元素，
   # 取a中 第一维数组下标为0的元素，取第二维数组下标为2的元素,取第三维数组下标为4的元素，其他维度的全取 （3）
    print(tf.gather_nd(a, indices = [0, 2, 4]).shape)

     # gather_nd  indices 如果是嵌套两个数组 
   # 取a中 先取第一维数组中下标为0，第二维数组中下标为1，剩下的全取的元素得到形状（28， 3）
   # 再取 第一维数组中下标为1，第二维数组中下标为1，剩下的全取的元素得到形状（28， 3）
   #  再取 第一维数组中下标为3，第二维数组中下标为1，剩下的全取的元素得到形状（28， 3） 
   # 最后三个形状合并到一个数组中 得到形状 （3, 28,3）
    print(tf.gather_nd(a, indices = [[0, 1], [1, 1], [3, 1]]).shape)
    return 

#形状变化 和 增减维度
def fun4 ():
    a = tf.random.normal([4, 28, 28, 3])
    # 形状变换 
    a = tf.reshape(a, [4, -1, 3])
    #a 的shape (4, 784, 3) 
    print(a.shape, a.ndim) 
    a = tf.reshape(a, [4, 784 * 3])
      #a 的shape (4, 2352) 
    print(a.shape, a.ndim) 
    a = tf.reshape(a, [4, 784, 3])
      #a 的shape (4, 784, 3) 
    print(a.shape, a.ndim) 

    #转置
    a = tf.random.normal([4,3,2,1])
     # 转置，如果不传参数 得到的是全转置 [1,2,3,4]
    print(tf.transpose(a).shape)
 # 转置，如果传参数 perm 中填的是下标  得到的是 [4，3,1,2] 
    print(tf.transpose(a, perm=[0, 1,3, 2]).shape)


    a = tf.random.normal([28, 28, 3])
  
    # 在第一个维度前面增加一个维度 默认是 1  (1, 28, 28, 3)
    a = tf.expand_dims(a, axis=0)
    print(a.shape)

      # 将第一个维度删了   (28, 3)
    print(tf.squeeze(a, axis=0).shape)

    # 加法的本质 是对 这个形状所描述的数据源的加法。而两个矩阵想要相加，就必须形状一样，如果不一样，能补0则补0
    a = tf.random.normal([4, 28, 28, 3])
    b = tf.random.normal([1, 1, 1])
    # 4, 28, 28, 3
    print((a+b).shape)
    print(tf.broadcast_to(a, [5, 4, 28, 28, 3]).shape)

    
    a = tf.random.normal([4, 28, 28, 3])
    a = tf.expand_dims(a, axis=0)
    print(a.shape)
    # 给某一个维度进行扩充, 对a的第一个数组数量变成原来的2倍，
    #                    对a的第二个数组数量变成原来的2倍，
    #                    对a的第三个数组数量变成原来的1倍， 
    #                    对a的第四个数组数量变成原来的1倍，
    #                    对a的第五个数组数量变成原来的1倍，
    a = tf.tile(a, [2, 2, 1, 1, 1])
    print(a.shape)

    return


def fun5 () :
    # tensor 合并
    a = tf.ones([4, 35, 8])
    b = tf.ones([2, 35, 8])
    # axis 定义了在哪个维度上进行合并  ([6, 35, 8]),要求是 除了要合并的维度外，其他的维度都必须相同
    print(tf.concat([a, b], axis = 0).shape)

    a = tf.ones([4, 35, 8])
    b = tf.ones([4, 35, 8])
   # axis 新增一个维度 axis 定义了这个维度在哪  (2, 4, 35, 8) 要求是 当前所有的维度都必须相同
    c = tf.stack([a, b], axis=0)
    print(c.shape)

    # 切割维度,axis 指定要打散的维度，注意，在要打散的维度上 有几个数据源 就打散成几个Tensor
    aa, bb = tf.unstack(c, axis=1)
    print(aa.shape, bb.shape)
    # print(tf.unstack(c, axis=1))

     # 切割维度,axis 指定要打散的维度，num_or_size_splits 标识了将要打散的维度拆成几个Tensor，具体怎么拆
     # 将 第二个数组 拆成 第一个数据源作为一个tensor 第2到3作为一个tensor
    aa, bb = tf.split(c, axis = 1,  num_or_size_splits = [1, 3])
    print(aa.shape, bb.shape)
    return 

def fun6 () :
#  向量 范数
  # 张量的二范数，一个矩阵[2, 2]，他的二范数就是 各个元素平方之和相加，最后再开平方
  a = tf.ones([2, 2])
  # 求二范数
  print(tf.norm(a))
 #整个计算过程,square 各个元素的平方、reduce_sum：各个元素的和，sqrt：开平方
  print(tf.sqrt(tf.reduce_sum(tf.square(a))))
    # 求二范数，将第二维元素 平方 相加 最后再开平方
  print(tf.norm(a, axis=1))

  # 求一范数 
  print(tf.norm(a, ord=1, axis=1))
  a = tf.random.normal([4, 19])

  #最大值、最小值、平均值，主要是两个维度，如果指定axis，则在指定的维度进行求值，如果是不指定，则在所有的维度进行求值
  # 这个值是 [2]
  print(tf.reduce_max(a).shape)
  # 最大值索引  [19]  一共有19列
  print(tf.argmax(a).shape)
   # 最小值索引  [19]  一共有19列
  print(tf.argmin(a).shape)

  a = tf.random.uniform([3, 3], maxval=10, dtype=tf.int32)
  # 排序
  print(tf.sort(a))
  # 排序后的 每一行的索引
  print(tf.argsort(a))

 #  返回每一行的前几个的最大值，返回每一行前2个最大值
  res = tf.math.top_k(a, 2)
  # 返回最大值的索引
  print(res.indices)
  # 返回最大值
  print(res.values)
  return


def fun2 () :
    (x, y), _ = datasets.mnist.load_data()
    z = []
    for num in y :
        num = num.astype(np.int32)
        z.append(num) 
    x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
    z = tf.convert_to_tensor(z, dtype=tf.int32)
    print('----------------------------------')

    print(x.shape, z)
    print(tf.reduce_min(x), tf.reduce_max(x))
    print(tf.reduce_min(z), tf.reduce_max(z))
    return 
print(tf.__version__)
fun6()


