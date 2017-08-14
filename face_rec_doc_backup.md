### 推荐阅读

- http://eyalarubas.com/face-detection-and-recognition.html



## 安装

根据自己的机器硬件进行编译安装，可以开启一些特殊指令和图形运算加速支持。

### OS X

```sh
brew tap homebrew/science
brew install opencv3 --c++11 --with-contrib --with-examples --with-ffmpeg --with-gstreamer --with-python3 --with-tbb --with-qt5 --with-opengl --with-nonfree --without-python
touch /usr/local/lib/python3.6/site-packages/opencv3.pth
echo /usr/local/opt/opencv3/lib/python3.6/site-packages >> /usr/local/lib/python3.6/site-packages/opencv3.pth
```

### Arch

```sh
pacman -Syu opencv opencv-samples
```

### Ubuntu
```sh
wget -O install-opencv.sh https://raw.githubusercontent.com/milq/milq/master/scripts/bash/install-opencv.sh
chmod +x install-opencv.sh
bash ./install-opencsv.sh
```
## 人脸检测

*判断图片或视频中是否有人脸并且定位人脸的问题*

### Viola& Jones人脸检测算法

[原理](http://docs.opencv.org/3.2.0/d7/d8b/tutorial_py_face_detection.html ) 推荐阅读

#### 训练

​	训练样本（可能要至少几百张图片才有比较好的效果）分为正例样本和反例样本，其中正例样本是指待检目标样本(例如人脸或汽车等)，反例样本指其它任意图片，所有的样本图片都被归一化为同样的尺寸大小(例如，20x20)

​	分类器训练完以后，就可以应用于输入图像中的感兴趣区域(与训练样本相同的尺寸)的检测。检测到目标区域(汽车或人脸)分类器输出为1，否则输出为0。为了检测整副图像，可以在图像中移动搜索窗口，检测每一个位置来确定可能的目标。 为了搜索不同大小的目标物体，分类器被设计为可以进行尺寸改变，这样比改变待检图像的尺寸大小更为有效。所以，为了在图像中检测未知大小的目标物体，扫描程序通常需要用不同比例大小的搜索窗口对图片进行几次扫描。

1. 将图片(视频帧)缩小，可以加快检测速度，
2. 转化 BGR为灰度图像（haar特征是基于灰度图像的）
3. 使用积分通道快速计算图像的haar特征值
4. 使用AdaBoost算法分类器筛选特征
5. 将AdaBoost分类器改成级联的boosted分类器，快速丢弃非人脸特征

`弱分类器 -> 优化弱分类器 ->　强分类器 -> 级联分类器(Haar分类器)`

#### 参数

- [参数说明](http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php)

1. **scaleFactor** : 在前后两次相继的扫描中，搜索窗口的比例系数。模型在训练的时候size是固定的，所以同样size的脸可以被直接识别到，但对于一张比较大的脸（比如非常靠近摄像头），那么就需要逐步缩小它（到最小size），直至可以被模型算法检测到。 比如，设置它成 1.02，每次只会缩小 2%，就增加了匹配的可能性，因为会在更多的大小尺寸上进行检测，虽然计算量会增加，但是会提高召回率。

2. **minNeighbors** :  This parameter will affect the quality of the detected faces: higher value results in less detections but with higher quality. We're using 5 in the code.

   The detection algorithm uses a moving window to detect objects. `minNeighbors` defines how many objects are detected near the current one before it declares the face found. 

3. **size** 检测窗口的最小尺寸。缺省的情况下被设为分类器训练时采用的样本尺寸(人脸检测中缺省大小是~20×20), 小于该尺寸的脸将会被忽略。

在不同的场景和用例情况下，需要设置以上参数的组合，才能达到更好的效果。

#### 评估

多次试验调节`scaleFactor`和`minNeighbors`，发现该人脸检测算法识别正脸的效果不错，对大图片的检测的速度也相对较快，但检测非正脸图片的效果不太好。总体而言，在准确率和召回率之间难以取得一个较好的效果平衡。

但另外也发现了opencv也提供了其他的预先训练好的模型，针对人脸和部件测试的模型。

| Type of cascade classifier               | XML File                            |
| ---------------------------------------- | ----------------------------------- |
| Face detector (default)                  | haarcascade_frontalface_default.xml |
| Face detector (fast Haar)                | haarcascade_frontalface_alt2.xml    |
| Profile (side-looking) face detector     | haarcascade_profileface.xml         |
| Eye detector (separate for left and right) | haarcascade_eye.xml                 |
| Simle face detector                      | haarcascade_smile.xml               |

还有其他比如检测 upper body, full body的模型。

#### 学习

-  [Face Detection and Tracking](http://www.youtube.com/watch?v=WfdYYNamHZ8)
- [Interview regarding Face Detection by Adam Harvey](http://www.makematics.com/research/viola-jones/)

## 人脸识别

*在人脸识别的基础上，根据人脸识别这个人*

推荐阅读：[Eigenface vs Fisherface](https://cseweb.ucsd.edu/classes/wi14/cse152-a/fisherface-pami97.pdf)

#### Eigenface Recognizer

* 先把一批人脸图像转换成一个特征向量集，称为“特征脸”(*Eigenfaces*)，它们是最初训练图像集的基本组件。识别的过程是把一副新的图像投影到特征脸子空间，并通过它的投影点在子空间的位置以及投影线的长度来进行判定和识别。
* `createEigenFaceRecognizer(num_component, threshold)`
  * `num_components`: PCA主成分的维数。根据输入数据的大小而决定，通常认为80维主成分是足够的。
  * `threshold`:预测时的阈值


* 训练图像和测试图像都必须是灰度图，而且是经过归一化裁剪过的
*  eigenfaces method looks at the dataset as a whole. 
* 对光照敏感，PCA里求得的k维特征向量都是正交的
* [原理说明](http://blog.csdn.net/smartempire/article/details/21406005)

#### Fisherface Recognizer

* `createFisherFaceRecognizer(num_component, threshold)`
  * `num_component`:
  * `threshold`:
* 训练图像和测试图像都必须是灰度图，而且是经过归一化裁剪过的
* 对光照敏感，LDA求得的k维特征向量不一定是正交的
* [原理说明](http://blog.csdn.net/smartempire/article/details/23377385)

#### Local Binary Patterns Histograms Face Recognizer 

* ```
  createLBPHFaceRecognizer(radius, neighbors, grid_x, grid_y, threshold)
  ```

  * `radius`:中心像素点到周围像素点的距离，构建圆LBP特征，相邻像素距离为1，默认1
  * `neighbors`:选取的周围像素点的个数，默认8，采样点越多，计算代价越大。 
  * `grid_x`:将一张图片在x方向分的块数，默认8，区块越多，终究构建结果的特点向量的维度越高。
  * `grid_y`:将一张图片在y方向分成的块数，默认8
  * `threshold`:LBP特征向量相似度的阈值，只有两张图片的相似度小于阈值才可认为识别有效，大于阈值则返回-1

* 局部二值模式是一个简单但非常有效的纹理运算符，它的基本思想是：通过比较图片中像素和与它相邻的像素对局部进行求和。如在3x3的窗口内，以窗口中心像素为阈值，将相邻的8个像素的灰度值与其进行比较，若周围像素值大于或等于中心像素值，则该像素点的位置被标记为1，否则为0。这样，3*3邻域内的8个点经过比较可产生8位二进制数，如图1中00010011（通常转换为十进制数即LBP码，共256种），即得到该窗口中心像素点的LBP值，并用这个值来反映该区域的纹理信息。

* 基本的LBP算子只局限在3*3的邻域内，对于较大图像大尺度的结构不能很好的提取需要的纹理特征，因此研究者们对LBP算子进行了扩展。新的LBP算子LBP（P,R） 可以计算不同半径邻域大小和不同像素点数的特征值，其中P表示周围像素点个数，R表示邻域半径，同时把原来的方形邻域扩展到了圆形

* 通过对全局图像进行LBP特征提取得到LBP图，LBP特征图是不能直接来作人脸识别的，需要对LBP特征图进行分块并计算每个分块的直方图，通过直方图的统计信息进行识别，最后将各块的直方图首尾相连就得到一张图片最终的LBP特征描述向量。计算两张图片的LBP特征向量的相似度即可实现人脸识别。

* LBPH,局部二进制编码直方图, 比较简单，预测效果也好，参数包括半径radius，邻域大小即采样点个数neighbors，x和y方向的单元格数目grid_x,grid_y，还有两个参数histograms为训练数据得到的直方图，labels为直方图对应的标签。这个方法也要求训练和测试的图像是灰度图。

* 对光照变化等造成的灰度变化的有鲁棒性

  * LBP算子利用了周围点与该点的关系对该点进行量化。量化后可以更有效地消除光照对图像的影响。只要光照的变化不足以改变两个点像素值之间的大小关系，那么LBP算子的值不会发生变化，所以一定程度上，基于LBP的识别算法解决了光照变化的问题，但是当图像光照变化不均匀时，各像素间的大小关系被破坏，对应的LBP模式也就发生了变化。

* 计算简单，可以支持实时分析

* LBPH对于training set中的图片的每一张脸的处理都是独立的



### 运行demo

```shell
# collect your face feature from camera, replace `zhenglai` with your name
./003_face_rec_fisher_collect.py <your_name>
# goto att_faces/<your_name> do select 10 representive images, and deleted others

# test with att_faces database
./004_face_rec_fisher_train_test.py image

# test with camera
./004_face_rec_fisher_train_test.py camera
```



### 总结

[opencv face modules](https://github.com/opencv/opencv_contrib/tree/master/modules/face) 在github上面看起来开发并不活跃，support较少，现在基本都上了神经网络做人脸识别。在环境噪声比较弱的场景下面正脸的识别用opencv的人脸识别模块还是比较方便和快捷的。

对于 att_faces数据库测试，att_faces对于每个人有10张图片，共40x10=400张图片，最开始测试时，设置非常大的threshold(2500)有如下结果：

| size(train/test) | Eigenface(Percision/Recall) | Fisherface | LBPH      |
| ---------------- | --------------------------- | ---------- | --------- |
| 9/1              | 0.944/1.0                   | 0.944/1.0  | 1.0/1.0   |
| 8/2              | 0.945/1.0                   | 0.945/1.0  | 0.972/1.0 |
| 7/3              | 0.897/1.0                   | 0.897/1.0  | 0.953/1.0 |
| 6/4              | 0.868/1.0                   | 0.875/1.0  | 0.938/1.0 |
| 5/5              | 0.891/1.0                   | 0.879/1.0  | 0.937/1.0 |

可以看到，该测试随着train_set不断增多，测试效果越来越好，三者对比，LBPH表现最好。由于threshold很高，recall很好，降低threshold,尝试提高精度，对于fisher和eigen，都设置为1500，对于lbph设置为95，有如下结果：

| size(train/test) | Eigenface(Percision/Recall) | Fisherface   | LBPH        |
| ---------------- | --------------------------- | ------------ | ----------- |
| 9/1              | 0.944/1.0                   | 0.944/1.0    | 1.0/1.0     |
| 8/2              | 0.944/0.985                 | 0.945/0.985  | 0.972/0.985 |
| 7/3              | 0.957/0.874                 | 0.957/0.883  | 0.953/0.99  |
| 6/4              | 0.991/0.748                 | 0.991/0.755  | 0.944/0.992 |
| 5/5              | 0.991/0.705                 | 0.99.2/0.711 | 0.947/0.992 |

Eigenface和fisherface效果在percision和recall上都比较接近，lbph表现依旧良好。实际使用中，可以根据不同场景设置不同的threshold，对光照影响非常大，尝试过在稍微昏暗的地方通过camera抓取人脸识别，检测非常差，换一个视角(大于10度以上)，人脸就无法检测出来，也就无法识别。

## 人脸检索

### 相似度度量算法
- 欧式距离
- 余弦相似度
- 汉明距离

## 查询处理算法
- Naive查询处理算法
- [KD-tree](http://blog.csdn.net/silangquan/article/details/41483689)

## DNN & CNN

- [DeepFace](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf) by facebook

- [FaceNet](https://arxiv.org/abs/1503.03832) by google, open sourced implementation: [openface](https://github.com/cmusatyalab/openface)

- [Facenet](https://github.com/davidsandberg/facenet) **LFW accuracy - 0.992**

- [DeepVideoAnalytics](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics)

- [Caffe Model-Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) contains [Pose-Aware CNN Models (PAMs) for Face Recognition](https://github.com/BVLC/caffe/wiki/Model-Zoo#pose-aware-cnn-models-pams-for-face-recognition) & [ResFace101: ResNet-101 for Face Recognition](https://github.com/BVLC/caffe/wiki/Model-Zoo#resface101-resnet-101-for-face-recognition) pretrained models

- [Tensorflow Model-Zoo] contains [slim](https://github.com/tensorflow/models/tree/master/slim)

  ​

### TODO

- 寻找合适的亚洲人脸图像数据库
- haar classifier对人脸进行定位时候是采用矩形的方式，会包含脸部以外的信息，是否有其他更好的 face crop的方式？
- 在extract feature之前，可以做alignment，确保rotated faces的良好特征也可以被正确的提取出来
  - https://stackoverflow.com/questions/10143555/how-to-align-face-images-c-opencv
- 重构现有代码，抽象出各种接口，方便后续接入不同的模型框架、不同的预处理组合调整
- 构建自己的数据处理和训练 python library，交叉验证，matplotlib/ipython 作图，画出各种指标，定量比较
- 看看能不能复用现有的神经网络(facenet?)的训练好的人脸识别的模型的进行迁移学习