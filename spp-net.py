import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array
from keras import backend as K
from SpatialPyramidPooling import SpatialPyramidPooling
from dataset import find_img
import pandas as pd
import cv2 as cv
import os
from PIL import Image
import numpy as np

batch_size = 128
num_classes = 10
epochs = 12

list_train = pd.read_csv( 'dataset/af2019-cv-training-20190312/list_train.csv' )
list_test = pd.read_csv( 'dataset/af2019-cv-training-20190312/list_test.csv' )

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
x_train = find_img.label_find( list_train['id'].tolist() , 'dataset/data_new/train/' , '_c.jpg' )
y_train = list_train['judge']

x_test = find_img.label_find( list_test['id'].tolist() , 'dataset/data_new/test/' , '_c.jpg' )
y_test = list_test['id']


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    # input_shape = (1, img_rows, img_cols)
    input_shape = (1, None, None)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    # input_shape = (img_rows, img_cols, 1)
    input_shape = (None, None, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# model.add(Flatten())
model.add(SpatialPyramidPooling([1,2,4]))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("spp-net.h5")

if __name__ == "__main__":
    pass
