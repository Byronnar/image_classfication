#-*- encoding:utf-8 -*-
__date__ = '19/03/21'
import os
import numpy as np
import shutil

#导入所需要的模块，numpy科学计算库，cv2图像处理库，os，对操作系统进行操作（保存文件等）的库，codes是处理任意编码的模块，shutil模块是移动文件的
isExists = os.path.exists(r'./0')
# 判断结果
if not isExists:
    os.makedirs((r'./0'))
    os.makedirs(r'./1')

txt_file = open(r"./results.txt") #打开文件
line = txt_file.readline() #读取每一行
data_list = []

while line:
    num = list(map(float, line.split( )[1])) #找到txt文件中的0和1
    data_list.append(num)
    line = txt_file.readline()
txt_file.close() #关闭文件
data_array = np.array(data_list)
print('分类结果列表:',data_array[:,0])

# 读取每张图片按照其分类复制到相应的文件夹中
imgs = os.listdir('./picture')
imgnum = len(imgs)  # 文件夹中图片的数量
j = 1
for i in range(imgnum):  # 遍历每张图片
    #print(int(data_array[i][0]))
    label=int(data_array[i][0])    #图片对应的类别
    #print(label)
    if j < 10:
        shutil.move('./picture/'+ '0' +str(j)+'.jpg', './'+str(label)+'/'+'0'+str(j)+'.jpg')
    elif j>=10:
        shutil.move('./picture/'+str(j)+'.jpg', './'+str(label)+'/'+str(j)+'.jpg')
#shutil.move()函数，将图片从一个文件夹移动到另一个文件夹。第一个参数是旧的文件路径，第二个参数为新的文件路径。
    j+=1


