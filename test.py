# // 反向溯源
import tensorflow as tf
import json
import os
import heapq

def imgToTensor(path,  width, height):
    image = tf.keras.utils.load_img(path)
    input_arr = tf.keras.utils.img_to_array(image)
    if width > 0 and height > 0:
        # input_arr = tf.image.resize_with_pad(
        # image=input_arr, target_height=width, target_width=height,method='bicubic')  
        input_arr = tf.image.resize_with_crop_or_pad(
            image=input_arr, target_height=width, target_width=height
        )
    # # tf.keras.utils.save_img(path2, x=input_arr)
    return  tf.convert_to_tensor(input_arr)
    # return input_arr
    

base = "/Volumes/My Passport/tensorflow-study/data3/"
base2 = "/Volumes/My Passport/tensorflow-study/data2/"
list = []
def priect(img): 
    tensor2 = imgToTensor(img,  400, 400) 
    for dirs in os.listdir(base2):
    # files = os.listdir(base2 + str(dirs))   # 读入文件夹
    #  files = os.listdir(base2  + str(dirs) )
        print(dirs)
        tensor = imgToTensor(base2 + str(dirs) + '/' + str(0) + ".jpg", 400, 400)
        correct = tf.equal(tensor, tensor2)
        correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
        tensor3 = tf.fill(tensor.shape, 1)
        total = tf.reduce_sum(tf.cast(tensor3, dtype=tf.int32)).numpy()
    # return correct / total * 100
        list.append(correct / total * 100)
        print(correct / total * 100)

priect('/Volumes/My Passport/tensorflow-study/data2/2010/2.jpg')
top_ten_max = heapq.nlargest(10, list)

print(top_ten_max)



# print('--------------------') 


    #  res = priect(base2 + str(dirs) + '/' + str(0) + ".jpg")
    #  print(res)
    #  list.append(res)

       
       
       
  
# total = 0
# for dir in list:
#     if dir > 0.21:
#         total += 1
# print(total)
    

