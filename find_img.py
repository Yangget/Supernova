# -*- coding: utf-8 -*-
"""
 @Time    : 19-3-13 下午7:21
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : find_img.py
"""
"""
寻址 和 做图函数
"""
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import shutil

def allList(list,i):
    """
    返回某一集合特定的三个子图地址
    :param list: 指定集合
    :param i: 指定号码
    :return: 三个子图的路径
    """
    df = pd.read_csv(list)


    fl = df['id'][i]
    # 获取文件夹的名称
    folder = fl[:2]
    # 完整的文件路径
    folder_ = './dataset/af2019-cv-training-20190312/' + folder + '/' + fl
    img_a = folder_ + '_a.jpg'
    img_b = folder_ + '_b.jpg'
    img_c = folder_ + '_c.jpg'
    return img_a,img_b,img_c


def drawGroup(img_a,img_b,img_c):
    """
    画出三个子图
    :param img_a: 路径
    :param img_b: 路径
    :param img_c: 路径
    :return: 图
    """
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


