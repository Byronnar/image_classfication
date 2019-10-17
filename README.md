### 处理自己的图片所需的准备工作

基于无监督学习，无需训练数据, 使用SIFT算法提取图像特征，再使用KMeans聚类算法进行图像分类。
对源代码进行了优化，实现了对应图片自动分类到各自文件夹功能，并且优化了分类准确率。
设计思路：

1）首先编写百度图片搜索网络爬虫(我github其他仓库有)，批量下载猫狗等图像数据，构建数据集。

2）利用opencv库对图像数据进行处理，进行灰度化，二值化，膨胀，高斯滤波等操作

3）学习SIFT算法跟KMeans聚类算法，取其优点

4）编写代码进行图像分类

本次使用的是传统图像分类方法，下次会实现基于深度学习卷积神经网络图像分类。

##  跑一下demo
###  环境要求:
windows/linux

python3

opencv-python==3.4.2.17

opencv-contrib-python==3.4.2.17

sklearn

1) 运行 Imagef_SIFT_K-Means.py文件

2) 运行 move_results.py文件

下图是进行聚类分类后产生的结果文件：

![Image text](https://github.com/Byronnar/image_classfication/blob/master/new_results.png)

思考能不能根据分类结果自动将图片分类到各自文件夹呢，可以遍历结果文件，取各个图片对应的标签，进行自动归类
下图是改进代码后的结果：
![Image text](https://github.com/Byronnar/image_classfication/blob/master/%E6%97%A0%E7%9B%91%E7%9D%A3.png)

