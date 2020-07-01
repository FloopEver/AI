# -*- coding: utf-8 -*-
import numpy as np

import csv


#读取图片的灰度值矩阵
def read_photo():
    with open('train_image.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        data_lst = []
        for line in reader:
            data_lst.append([int(x) for x in line])
    data=np.array(data_lst)#rows是数据类型是‘list',转化为数组类型好处理
    print (data)
    return data


#读取excel文件内容（path为文件路径）
def read_excel():
    with open('train_label.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        data_lst = []
        for line in reader:
            d = [int(x) for x in line]
            newd = np.zeros(10)
            newd[d[0] - 1] = 1
            data_lst.append(newd)
    data=np.array(data_lst)#rows是数据类型是‘list',转化为数组类型好处理
    print (data)
    return data

#layerout函数
def layerout(w,b,x):
    y = np.dot(w,x) + b
    t = -1.0*y
    # n = len(y)
    # for i in range(n):
        # y[i]=1.0/(1+exp(-y[i]))
    y = 1.0/(1 + np.exp(t))
    return y


#训练函数
def mytrain(x_train,y_train):
    '''
    设置一个隐藏层，784-->隐藏层神经元个数-->10
    '''

    step=int(input('mytrain迭代步数：')) 
    a=float(input('学习因子：')) 
    inn = 784  #输入神经元个数
    hid = int(input('隐藏层神经元个数：'))#隐藏层神经元个数
    out = 10  #输出层神经元个数

    w = np.random.randn(out,hid)
    w = np.mat(w)
    b = np.mat(np.random.randn(out,1)) 
    w_h = np.random.randn(hid,inn)
    w_h = np.mat(w_h)
    b_h = np.mat(np.random.randn(hid,1)) 

    for i in range(step):
        #mini_batch
        for j in range(10000):
            #取batch为10  更新取10次的平均值
            x = np.mat(x_train[j]) 
            x = x.reshape((784,1))
            y = np.mat(y_train[j]) 
            y = y.reshape((10,1))
            hid_put = layerout(w_h,b_h,x) 
            out_put = layerout(w,b,hid_put) 

            #更新公式的实现
            o_update = np.multiply(np.multiply((y-out_put),out_put),(1-out_put)) 
            h_update = np.multiply(np.multiply(np.dot((w.T),np.mat(o_update)),hid_put),(1-hid_put)) 

            outw_update = a*np.dot(o_update,(hid_put.T)) 
            outb_update = a*o_update 
            hidw_update = a*np.dot(h_update,(x.T)) 
            hidb_update = a*h_update 

            w = w + outw_update 
            b = b+ outb_update 
            w_h = w_h +hidw_update 
            b_h =b_h +hidb_update 

    return w,b,w_h,b_h

#test函数
def mytest(x_test,y_test,w,b,w_h,b_h):
    '''
    统计1000个测试样本中有多少个预测正确了
    预测结果表示：10*1的列向量中最大的那个数的索引+1就是预测结果了
    '''
    sum = 0
    for k in range(10000):
        x = np.mat(x_test[k])
        x = x.reshape((784,1))

        y = np.mat(y_test[k])
        y = y.reshape((10,1))

        yn = np.where(y ==(np.max(y)))
        # print(yn)
        # print(y)
        hid = layerout(w_h,b_h,x);
        pre = layerout(w,b,hid);
        #print(pre)
        pre = np.mat(pre)
        pre = pre.reshape((10,1))
        pren = np.where(pre ==(np.max(pre)))
        # print(pren)
        # print(pre)
        if yn == pren:
            sum += 1

    print('1000个样本，正确的有:',sum)

def main():
    #获取图片信息
    im = read_photo()
    # immin = im.min()
    # immax = im.max()

    # im = (im-immin)/(immax-immin)

    #前4000张图片作为训练样本
    x_train = im[0:10000]
    #后1000张图片作为测试样本
    x_test = im[40000:50000]

    #获取label信息
    xl = read_excel()

    y_train = xl[0:10000]
    y_test = xl[40000:50000]

    print("---------------------------------------------------------------")
    w,b,w_h,b_h = mytrain(x_train,y_train)
    mytest(x_test,y_test,w,b,w_h,b_h)
    print("---------------------------------------------------------------")



if __name__ == '__main__':
    main()