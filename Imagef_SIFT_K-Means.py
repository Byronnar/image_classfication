#-*- encoding:utf-8 -*-
__date__ = '19/10/16'
import os, codecs
import cv2
import numpy as np
import time
from sklearn.cluster import KMeans
#导入所需要的模块，numpy科学计算库，cv2图像处理库，os，对操作系统进行操作（保存文件等）的库，codes是处理任意编码的模块，shutil模块是移动文件的

s1 = time.time()
def get_file_name(path): #获得文件名
    filenames = os.listdir(path) #  os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    path_filenames = []
    filename_list = [] #创建两个列表
    for file in filenames: #便利文件夹里面的文件
        if not file.startswith('.'):
            path_filenames.append(os.path.join(path, file))#路径拼接
            filename_list.append(file) #使用append 增加文件
    return path_filenames #返回：带文件名的路径

def kmens_detect(file_list, cluster_nums, randomState=None): # KNN 线性分类器
    features = []
    files = file_list #特征检测
    sift = cv2.xfeatures2d.SIFT_create() #调用SIFT特征提取方法
    for file in files:
        img = cv2.imread(file)#读入文件
        #方法一效率低
        #height, width = img.shape[:2]
        #res = cv2.resize(img, (2 * width, 2 * height),interpolation=cv2.INTER_CUBIC)
        #下面使用方法二
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
        # 重新设计图片大小，cv2.resize(img,(2*width,2*height)），这个当中是使用的宽高，这里一定要注意。
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #灰度化处理
        # print(gray.dtype) #打印类型

        kp,des = sift.detectAndCompute(gray, None) #调用SIFT算法
        # 检测并计算描述符
        # Kp,des=sift.detectAndCompute(gray,None)#检测并计算描述符
        # des =sift.detect(gray, None)# sift.detectAndCompute(gray, None)
        # 找到后可以计算关键点的描述符
        # Kp, des = sift.compute(gray, des)

        if des is None:
            file_list.remove(file)
            continue
        reshape_feature = des.reshape(-1, 1)
        features.append(reshape_feature[0].tolist())

    input_x = np.array(features) #计算关键点
    kmeans = KMeans(n_clusters=cluster_nums, random_state=randomState).fit(input_x) #关键点聚类
    return kmeans.labels_, kmeans.cluster_centers_

def res_fit(filenames, labels):
    files = [file.split('/')[-1] for file in filenames]
    return dict(zip(files, labels)) #打上标签

def save(path, filename, data):
    file = os.path.join(path, filename) #路径拼接
    with codecs.open(file, 'wb', encoding='utf-8') as fw: #编码
        # print('data',data)
        for f, l in data.items():
            fw.write("{}\t{}\r\n".format(f,l)) # 控制换行
def main():
    path_filenames = sorted(get_file_name("picture")) # 从picture 文件夹里面获取图片名字
    labels, cluster_centers = kmens_detect(path_filenames, 2) # 识别两类
    imgs = os.listdir('./picture')
    imgnum = len(imgs)  # 文件夹中图片的数量
    print ('文件夹里面图片数量:',imgnum)

    res_dict = res_fit(path_filenames, labels) #带上标签
    print('分类结果为:',res_dict)

    save('./', 'results.txt', res_dict) #文件保存为txt格式
    s2 = time.time()
    print('无监督分类器总运行时间为:',s2 - s1)

if __name__ == "__main__": #主函数启动
    main()
