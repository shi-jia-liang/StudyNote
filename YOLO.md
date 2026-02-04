# YOLO
## 图像标注label
* 使用Yolo需要对数据集进行标注，常用的标注有`labelimg`、`labelme`、`label-studio`  
推荐使用conda建立虚拟环境后，在安装对应的标注工具
	```bash
	conda create -n label python=3.8
	```
  (安装python3.8是因为labelimg不支持高版本python，安装高版本python会导致闪退)
### labelimg(国内安装)
&emsp;&emsp;labelimg仅支持使用矩阵预测框标记图片，但可以直接生成符合YOLO格式的txt文件，以及其余的CreateML格式、PascalVOC格式。  
&emsp;&emsp;改变生成格式在左侧栏的第8个按钮。
#### 安装
```bash
pip install labelimg
```
#### 使用
```bash
labelimg
```
### labelme
&emsp;&emsp;相比于labelimg，labelme可以支持使用多边形预测框标记图片，且支持中文，还能使用AI模型进行快速标注(只是不用手标，不代表速度标注速度快，推荐使用`SagmentAnything(accuracy)`)。此标注工具非常适合`图像分割`(SAM，SegmentAnythingModel)。
#### 安装
```bash
pip install labelme
```
#### 使用
```bash
labelme
```
### label-studio(未使用过)
&emsp;&emsp;一个通过本地网页的标注工具。需要注册账号。
#### 安装
```bash
pip install label-studio
```
#### 使用
```bash
label-studio
```
## 使用Yolo进行图像训练
### 安装YOLO
```bash
pip install ultralytics
```
&emsp;&emsp;需要在`data`文件夹中编写数据文件，设定`path`、`train`、`val`、`test`、`number of classes`。可参考`coco128.yaml`设定，需要注意的是，在`coco128.yaml`中设定的是`path: ../datasets/coco128`，这是因为该数据文件会把图片和标签下载在`yolov5`项目外的文件夹。因此，我在`yolov5`项目里自己编写数据文件时，且将图片和标签放在`datasets`文件夹中，应该设定为`path: datasets/data_name`。 
&emsp;&emsp;同时，还需要修改`models/yolov5s.yaml`中的`nc`参数对应数据文件中的关键字。  
&emsp;&emsp;此外，非常建议，图片和标签按照7:2:1划分为训练集、验证集、测试集。