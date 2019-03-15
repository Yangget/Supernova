# -*- coding: utf-8 -*-
"""
 @Time    : 19-3-15 下午5:53
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : test.py
"""
from dataset import find_img
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.preprocessing import image
import pandas as pd
import numpy as np
import cv2 as cv
# img = cv.imread( 'dataset/data_new/train/87d7d5011101f4ff56244013bffde4e0_c.jpg')
# # print(img)
# cv.imshow( 'image' , img )
# cv.waitKey( 0 )
# cv.destroyAllWindows( )
list_train = pd.read_csv( 'dataset/af2019-cv-training-20190312/list_train.csv' )
list_test = pd.read_csv( 'dataset/af2019-cv-training-20190312/list_test.csv' )

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
x_train = find_img.label_find( list_train['id'].tolist() , 'dataset/data_new/train/' , '_c.jpg' )
print(x_train)
y_train = list_train['judge']

x_test = find_img.label_find( list_test['id'].tolist() , 'dataset/data_new/test/' , '_c.jpg' )
y_test = list_test['id']

X_train = []
for i in range(len(x_train)):
    img = image.load_img(x_train[i],target_size=(300,300))
    x = image.img_to_array( img )
    x = np.expand_dims( x , axis=0 )
    X_train.append( x )
    print( 'loading no.%s image' % i )

# 把图片数组联合在一起
x = np.concatenate([x for x in X_train])
print(x.shape)
