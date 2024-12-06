# 学习笔记

---
## 环境安装
### C/C++
使用[Msys2](https://www.msys2.org/)，一个类似于Liunx内核的包管理系统。
可以按照官方教程安装`mingw-w64-ucrt-x86_64-gcc`，个人喜欢安装`mingw-w64-x86_64-gcc`。

初始化环境
```bash
pacman -Syu
```

以下附上我所安装的所有包(**安装mingw-w64-x86_64-gcc工具链**)
```bash
pacman -S --needed base-devel mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake gcc gdb
```
添加环境变量

![环境变量](./img/environment%20variables.PNG)

#### 安装CMake
按照[CMake](https://cmake.org/download/)网站提供的msi文件安装。

![CMake官网](./img/CMake.PNG)

#### 安装OpenCV
此为C/C++语言环境安装OpenCV，首先打开CMake-gui。（[可按照此篇文档进行安装](https://blog.csdn.net/SirClown/article/details/142614854)）
需要注意的是，一定要勾选**BUILD_opencv_world**，它的作用是将所有库文件使用动态库链接的方式存储，方便后续设置环境变量只用设置一个文件夹。
编译源文件为下载的OpenCV整个源文件，而添加的*OPENCV_EXTRA_MODULES_PATH*为`opencv_contrib\modules`。
在编译完成后，需要在build文件夹中，使用cmd输入**mingw32-make**编译生成文件，最后使用**mingw32-make install**进行安装。
安装后的所有文件保存在`build\install`中，我们需要设置`\build\install\x64\mingw\bin`添加入环境变量中。

---
## 关于Yolo
### 图像标注label
使用Yolo需要对数据集进行标注，常用的标注有`labelimg`、`labelme`、`label-studio`  
推荐使用conda建立虚拟环境后，在安装对应的标注工具
```bash
conda create -n label python=3.8
```
（安装python3.8是因为labelimg不支持高版本python，安装高版本python会导致闪退）
#### labelimg
labelimg仅支持使用矩阵预测框标记图片，但可以直接生成符合YOLO格式的txt文件，以及其余的CreateML格式、PascalVOC格式。  
改变生成格式在左侧栏的第8个按钮。
##### 安装
```bash
pip install labelimg
```
##### 使用
```bash
labelimg
```
#### labelme
相比于labelimg，labelme可以支持使用多边形预测框标记图片，且支持中文，还能使用AI模型进行快速标注（只是不用手标，不代表速度标注速度快，推荐使用`SagmentAnything(accuracy)`）。此标注工具非常适合`图像分割`（SAM,SegmentAnythingModel）。
##### 安装
```bash
pip install labelme
```
##### 使用
```bash
labelme
```
#### label-studio（未使用过）
一个通过本地网页的标注工具。需要注册账号。
##### 安装
```bash
pip install label-studio
```
##### 使用
```bash
label-studio
```
### 使用Yolo进行图像训练
需要在`data`文件夹中编写数据文件，设定`path`、`train`、`val`、`test`、`number of classes`。可参考`coco128.yaml`设定，需要注意的是，在`coco128.yaml`中设定的是`path： ../datasets/coco128`，这是因为该数据文件会把图片和标签下载在`yolov5`项目外的文件夹。因此，我在`yolov5`项目里自己编写数据文件时，且将图片和标签放在`datasets`文件夹中，应该设定为`path： datasets/data_name`。 

同时，还需要修改`models/yolov5s.yaml`中的`nc`参数对应数据文件中的关键字。  

此外，非常建议，图片和标签按照7:2:1划分为训练集、验证集、测试集。

---
## 关于机器学习
### 监督学习（函数逼近：回归、分类问题）
#### 线性回归、逻辑回归
#### 决策树
#### 支持向量机
#### 神经网络
##### CV（Computer Vision）
信息理解：   
1、图像分类：将图像分为不同的类别或者标签，是CV中最基础也是最常见的任务之一。  
2、目标识别：在图像中定位并识别多个不同类别的物体。  
3、语义分割：对图像中的每个像素进行分类，将图像分割成不同的语义区域。  
4、实例分割：在语义分割的基础上，进一步区分不同物体实例。  
图像生成：  
1、图像增强：图像增强是指对图像进行处理，以改善图像的质量、对比度、亮度或锐度等方面的特征。  
2、风格迁移：将一幅图像的风格转移到另一幅图像上，从而创造出新的图像，融合了原始图像的内容和目标风格的特征。  
3、文生图：根据文字描述生成新的图像。
##### NLP（Natural Language Processing）
信息理解：  
1、文本分类：将文本按照一定的标准分类导不同的类别中，常见的应用包括情感分析、垃圾邮件过滤等。常见的模型包括朴素贝叶斯、支持向量机（SVM）、循环神经网络（RNN）、卷积神经网络（CNN）等。  
2、命名实体识别：从文本中识别和分类出命名实体，如人名、地名、组织机构名等。常用的技术包括条件随机场、双向长短期记忆网络、BERT等。
文本生成：根据给定的文本生成新的文本，常见的技术包括循环神经网络（RNN）、长短期记忆网络（LSTM）、生成对抗网络（GAN）等。
### 非监督学习
#### 聚类
#### 降维
#### 关联规则
#### 异常检测
### 强化学习

## 关于深度学习
### 误差反向传播
由输出层误差推前一层误差，~~将复杂的求导过程通过拉格朗日多项式化为简单的减法过程~~。  
在残差网络ResNet发明之前，存在梯度爆炸和梯度消失。这是因为在计算过程中，由于误差反向传播后会乘后一层误差，当神经网络层数较大时，其误差也会成指数型增大或减少。
### 残差网络ResNet
在残差网络ResNet发明之前，由于存在梯度爆炸和梯度消失，神经网络一直发展不起来。
残差网络ResNet解决的问题，降低由于神经网络层较大时，直接使用残差网络将误差系数变小。

### CNN卷积和池化
卷积层需要训练卷积核的参数，池化层无需训练。

### RNN循环神经网络
经典模型：LSTM（Super Max版RNN）

### Attention注意力模型

### GNN图神经网络 