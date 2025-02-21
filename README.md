# 学习笔记

---
## OepnCV C语言环境安装
### C/C++
* 使用[Msys2](https://www.msys2.org/),一个类似于Liunx内核的包管理系统。
可以按照官方教程安装`mingw-w64-ucrt-x86_64-gcc`,个人喜欢安装`mingw-w64-x86_64-gcc`。

* 初始化环境
```bash
pacman -Syu
```

* 以下附上我所安装的所有包(**安装mingw-w64-x86_64-gcc工具链**)
```bash
pacman -S --needed base-devel mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake gcc gdb
```
* 添加环境变量

![环境变量](./img/environment%20variables.PNG)

#### 安装CMake
* 按照[CMake](https://cmake.org/download/)网站提供的msi文件安装。

![CMake官网](./img/CMake.PNG)

#### 安装OpenCV
&emsp;&emsp;此为C/C++语言环境安装OpenCV,首先打开CMake-gui。([可按照此篇文档进行安装](https://blog.csdn.net/SirClown/article/details/142614854))
需要注意的是,一定要勾选**BUILD_opencv_world**,它的作用是将所有库文件使用动态库链接的方式存储,方便后续设置环境变量只用设置一个文件夹。
&emsp;&emsp;编译源文件为下载的OpenCV整个源文件,而添加的*OPENCV_EXTRA_MODULES_PATH*为`opencv_contrib\modules`。
&emsp;&emsp;在编译完成后,需要在build文件夹中,使用cmd输入**mingw32-make**编译生成文件,最后使用**mingw32-make install**进行安装。
&emsp;&emsp;安装后的所有文件保存在`build\install`中,我们需要设置`\build\install\x64\mingw\bin`添加入环境变量中。

---
## 关于PyCharm
&emsp;&emsp;`<2025.2.18> 遇到很傻逼的问题,在cmd中使用pip install会根据系统安装到指定文件夹,而在PyCharm的命令行中使用pip install会安装到C盘的用户站中。建议以后都用cmd去安装python包。(当你不需要虚拟环境时)

---
## 关于Yolo
### 图像标注label
* 使用Yolo需要对数据集进行标注,常用的标注有`labelimg`、`labelme`、`label-studio`  
推荐使用conda建立虚拟环境后,在安装对应的标注工具
```bash
conda create -n label python=3.8
```
(安装python3.8是因为labelimg不支持高版本python,安装高版本python会导致闪退)
#### labelimg(国内安装)
&emsp;&emsp;labelimg仅支持使用矩阵预测框标记图片,但可以直接生成符合YOLO格式的txt文件,以及其余的CreateML格式、PascalVOC格式。  
&emsp;&emsp;改变生成格式在左侧栏的第8个按钮。
##### 安装
```bash
pip install labelimg
```
##### 使用
```bash
labelimg
```
#### labelme
&emsp;&emsp;相比于labelimg,labelme可以支持使用多边形预测框标记图片,且支持中文,还能使用AI模型进行快速标注(只是不用手标,不代表速度标注速度快,推荐使用`SagmentAnything(accuracy)`)。此标注工具非常适合`图像分割`(SAM,SegmentAnythingModel)。
##### 安装
```bash
pip install labelme
```
##### 使用
```bash
labelme
```
#### label-studio(未使用过)
&emsp;&emsp;一个通过本地网页的标注工具。需要注册账号。
##### 安装
```bash
pip install label-studio
```
##### 使用
```bash
label-studio
```
### 使用Yolo进行图像训练
&emsp;&emsp;需要在`data`文件夹中编写数据文件,设定`path`、`train`、`val`、`test`、`number of classes`。可参考`coco128.yaml`设定,需要注意的是,在`coco128.yaml`中设定的是`path: ../datasets/coco128`,这是因为该数据文件会把图片和标签下载在`yolov5`项目外的文件夹。因此,我在`yolov5`项目里自己编写数据文件时,且将图片和标签放在`datasets`文件夹中,应该设定为`path: datasets/data_name`。 

&emsp;&emsp;同时,还需要修改`models/yolov5s.yaml`中的`nc`参数对应数据文件中的关键字。  

&emsp;&emsp;此外,非常建议,图片和标签按照7:2:1划分为训练集、验证集、测试集。

---
## 关于计算机视觉CV(Computer Vision)
* 信息理解:   
1. 图像分类:将图像分为不同的类别或者标签,是CV中最基础也是最常见的任务之一。  
2. 目标识别:在图像中定位并识别多个不同类别的物体。  
3. 语义分割:对图像中的每个像素进行分类,将图像分割成不同的语义区域。  
4. 实例分割:在语义分割的基础上,进一步区分不同物体实例。  

* 图像生成:  
1. 图像增强:图像增强是指对图像进行处理,以改善图像的质量、对比度、亮度或锐度等方面的特征。  
2. 风格迁移:将一幅图像的风格转移到另一幅图像上,从而创造出新的图像,融合了原始图像的内容和目标风格的特征。  
3. 文生图:根据文字描述生成新的图像。

---
## 关于自然语言
### 自然语言理解NLU(Natural Language Understanding)
* 信息理解:  
1. 文本分类:将文本按照一定的标准分类导不同的类别中,常见的应用包括情感分析、垃圾邮件过滤等。常见的模型包括朴素贝叶斯、支持向量机(SVM)、循环神经网络(RNN)、卷积神经网络(CNN)等。  
2. 命名实体识别:从文本中识别和分类出命名实体,如人名、地名、组织机构名等。常用的技术包括条件随机场、双向长短期记忆网络、BERT等。
文本生成:根据给定的文本生成新的文本,常见的技术包括循环神经网络(RNN)、长短期记忆网络(LSTM)、生成对抗网络(GAN)等。
### 自然语言处理NLP(Natural Language Processing)
### 自然语言生成NLG(Neural Language Generation)

---
## 关于机器学习
### 监督学习(函数逼近:回归、分类问题)
* 线性回归、逻辑回归
* 决策树、随机森林
* K近邻 K-NN(K-Nearest Neighbors)
* 支持向量机SVM
* 神经网络NN
  
### 非监督学习
* 聚类(K-Means算法)
* 降维
* 关联规则
* 异常检测

### 强化学习

---
## 关于深度学习
### 误差反向传播
由输出层误差推前一层误差,~~将复杂的求导过程通过拉格朗日多项式化为简单的减法过程~~。  
在残差网络ResNet发明之前,存在梯度爆炸和梯度消失。这是因为在计算过程中,由于误差反向传播后会乘后一层误差,当神经网络层数较大时,其误差也会成指数型增大或减少。
### 残差网络ResNet
在残差网络ResNet发明之前,由于存在梯度爆炸和梯度消失,神经网络一直发展不起来。
残差网络ResNet解决的问题,降低由于神经网络层较大时,直接使用残差网络将误差系数变小。

### CNN卷积和池化
1. 数据输入层:去均值,将维度都中心化为0,其目的是把样本的中心拉回到坐标系原点上。归一化处理,减少各维度数据取值范围的差异而带来的干扰。
2. 卷积层:需要训练卷积核的参数,由于图像通常是3通道图像(RGB图像),因此设定的卷积核也对各自的通道进行卷积操作(每个通道的卷积核可以不同)。每个卷积核都都是一个滤波器。在卷积层中,可以同时存在多个卷积核对其中的通道进行卷积。(例如,5\*5\*3的图像,使用两个3\*3\*3的卷积核,保证其维度相同,最终得到5\*5\*2的结果,在实际计算过程中,会用0填充边界,其次在由于使用两个卷积核则会得到两个维度)
3. 激活层:激活函数通常是非线性的,这样增加网络的深度才有意义,同时激活函数通常是可导的,这样才能进行梯度下降。常见的激活函数有,sigmoid函数、Tanh函数、**ReLU函数**、Leaky ReLU函数等。
4. 池化层无需训练,需要自己设定好参数(核大小、步长)。池化层的作用是下采样,缩小特征图尺寸,降维、去除冗余信息、对特征进行压缩、减少复杂度、减少计算量、减少内存消耗。

### RNN循环神经网络
经典模型:LSTM(Super Max版RNN)

### Attention注意力模型

### GNN图神经网络 

### Transformer


---
## 关于大模型微调技术
### 全量微调 VS LoRA微调

---
## 关于ROS
&emsp;&emsp;一个用于开发机器人软件的开源框架。尽管它的名字中包含“操作系统”，但它并不是传统意义上的操作系统（如Windows或Linux）。相反，ROS提供了一套工具、库和协议，帮助开发者更高效地构建复杂的机器人应用程序。
### 节点Node 和 包Package
* 节点Node是ROS架构中的基本计算单元。每个节点是一个独立的进程，负责执行特定的任务或功能。节点之间通过ROS的通信机制（如话题、服务、动作等）进行交互，从而共同完成复杂的机器人任务。
* 包Package是代码和资源的组织单元，用于实现机器人系统的模块化开发。每个包包含实现特定功能的代码（如节点、库、配置文件、消息定义等），并通过依赖管理和构建系统与其他包协作。

&emsp;&emsp;一个典型的ROS包包含以下文件和目录：
```bash
my_package/            # 包根目录
├── CMakeLists.txt      # 构建规则（ROS 1使用Catkin，ROS 2使用Ament/Colcon）
├── package.xml         # 包清单文件（定义元数据、依赖）
├── src/                # 源代码目录（C++、Python）
├── include/            # 头文件目录（C++）
├── msg/                # 自定义消息定义（.msg文件）
├── srv/                # 自定义服务定义（.srv文件）
├── action/             # 自定义动作定义（.action文件）
├── launch/             # 启动文件（.launch或.xml）
├── config/             # 配置文件（YAML、参数）
└── scripts/            # 可执行脚本（如Python节点）
``` 

### 常见的ROS指令
* 环境设置
```bash
# 启动ROS Master和参数服务器
roscore
# 设置ROS环境变量
source /opt/ros/melodic/setup.bash
```
* 包管理
```bash
# 查找指定包的路径
rospack find <package_name>

# 安装指定包的依赖
rosdep install <package_name>

# 创建一个新的ROS包
catkin_create_pkg <package_name> [dependencies]

# 编译当前工作空间中的所有包
catkin_make

# 编译并安装当前工作空间中的所有包
catkin_make install
```

* 节点管理
```bash
# 启动节点
rosrun <package_name> <node_name>

# 查看节点列表
rosnode list

# 查看节点信息
rosnode info <node_name>

# 停止节点
rosnode kill <node_name>

# 启动一个launch文件，通常用于启动多个节点
roslaunch <package_name> <launch_file>
```

* 话题管理
```bash
# 列出当前活跃的话题
rostopic list

# 实时显示指定话题的消息
rostopic echo <topic_name>

# 向指定话题发布消息
rostopic pub <topic_name> <message_type> <message>

# 查看指定话题的发布频率
rostopic hz <topic_name>

# 查看指定话题的带宽使用情况
rostopic bw <topic_name>
```

* 调试工具
```bash
# 启动rqt_graph工具，可视化节点和话题的连接关系
rqt_graph

# 启动rqt_console工具，查看和过滤ROS日志信息
rqt_console

# 启动rqt_plot工具，实时绘制话题数据
rqt_plot

# 启动RViz可视化工具，用于3D可视化
rviz
```