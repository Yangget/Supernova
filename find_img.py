# -*- coding: utf-8 -*-
"""
 @Time    : 19-3-13 下午7:21
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : find_img.py
"""

import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import shutil
import os
import glob
import time

"""
1. 切割数据集
2. 展示一组图
"""
# 切割数据集
def useList(df, new_df):
    """
    切割数据集
    :param df: 传入要切割的list
    :param new_df: 新名称
    :return:
    """
    # print(df)
    for i in range(len(df)):
        fl = df['id'][i]
        # 获取文件夹的名称
        folder = fl[:2]
        # 完整的文件路径
        folder_ = '../dataset/af2019-cv-training-20190312/' + folder + '/' + fl
        img_a = folder_ + '_a.jpg'
        img_b = folder_ + '_b.jpg'
        img_c = folder_ + '_c.jpg'
        src = '../dataset/data_new/' + new_df + '/'
        shutil.copy(img_a,src)
        shutil.copy(img_b,src)
        shutil.copy(img_c,src)



def drawGroup(img_a,img_b,img_c):
    plt.figure(1) # 创建第一个画板（figure）
    plt.subplot(211) # 第一个画板的第一个子图
    img_c = plt.imread(img_c) # 显示图片
    plt.imshow(img_c)
    plt.subplot(212) # 第二个画板的第二个子图
    img_b = plt.imread(img_b) # 显示图片
    plt.imshow(img_b)
    plt.figure(2) #创建第二个画板
    img_a = plt.imread(img_a) # 显示图片
    plt.imshow(img_a)
    plt.figure(1) # 调取画板1; subplot(212)仍然被调用中
    plt.subplot(211) #调用subplot(211)
    plt.axis('off') # 不显示坐标轴
    plt.show()


def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
        # os.path.getmtime() 函数是获取文件最后修改时间
        # os.path.getctime() 函数是获取文件最后创建时间
        dir_list.sort(key=lambda x: os.path.getmtime(os.path.join(file_path, x)))

if __name__ == '__main__':
    list_train = pd.read_csv( '../dataset/af2019-cv-training-20190312/list_train.csv')
    list_test = pd.read_csv( '../dataset/af2019-cv-training-20190312/list_test.csv')

    useList(list_train,'train')
    print('train ok')
    # get_file_list('../dataset/data_new/train')
    # print('sorted ok')
    useList(list_test,'test')
    print('test ok')
    # get_file_list('../dataset/data_new/test')
    # print('sorted ok')
