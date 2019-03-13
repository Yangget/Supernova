# -*- coding: utf-8 -*-
"""
 @Time    : 19-3-13 下午7:18
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : base_pro.py
"""
import pandas as pd
import shutil

"""
离散特征值编码
乱序数据集
切割数据集存档
"""

def cut_data(df, train_size):
     """
     对数据集切割，分为训练集，测试集，有需要请自己切割验证集
     :param df: 源文件路径
     :param train_size: 训练集所占的比例
     :return: 测试集，训练集
     """
     train_size = int(len(df) * train_size)
     list_train = df[:train_size]
     list_test = df[train_size:]
     list_train.to_csv( '../dataset/af2019-cv-training-20190312/list_train.csv' , index=False )
     list_test.to_csv( '../dataset/af2019-cv-training-20190312/list_test.csv' , index=False )


if __name__ == '__main__':
     # 读取文件
     df = pd.read_csv( "../dataset/af2019-cv-training-20190312/list.csv" )

     # 离散特征编码
     df[ 'judge' ] = df[ 'judge' ].map( {
          'asteroid' : 1 ,
          'ghost' : 2 ,
          'isnova' : 3 ,
          'isstar' : 4 ,
          'known' : 5 ,
          'newtarget' : 6 ,
          'noise' : 7 ,
          'pity' : 8 } )

     # 乱序操作
     df = df.sample( frac=1 )

     # # 保存乱序的文件
     df.to_csv( '../dataset/af2019-cv-training-20190312/list_.csv' , index=False )

     # 调用切割函数
     cut_data(df, 0.7)
