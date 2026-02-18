# ROS
一个用于开发机器人软件的开源框架。尽管它的名字中包含“操作系统”，但它并不是传统意义上的操作系统(如Windows或Linux)。相反，ROS提供了一套工具、库和协议，帮助开发者更高效地构建复杂的机器人应用程序。
## 学习的相关网址
[ROS Wiki](http://wiki.ros.org/)
[北京华清智能科技有限公司](http://www.autolabor.com.cn/)
## 节点Node 和 包Package
在ROS中，节点(Node) 和 包(Package) 是机器人软件开发的两个核心概念，二者协同工作实现模块化开发。
* **节点Node**是ROS架构中的基本计算单元。每个节点是一个独立的进程，负责执行特定的任务或功能。节点之间通过ROS的通信机制(如话题、服务、动作等)进行交互，从而共同完成复杂的机器人任务。
* **包Package**是代码和资源的组织单元，用于实现机器人系统的模块化开发。每个包包含实现特定功能的代码(如节点、库、配置文件、消息定义等)，并通过依赖管理和构建系统与其他包协作。

将多个节点放在一个包中，当下次需要使用时，安装一个包即可将多个节点一起安装，具有可移植性。包是节点的容器。

* 一个典型的`ROS包`包含以下文件和目录：
```bash
catkin_ws :自定义的工作空间
    |--- build :编译空间，用于存放CMake和catkin的缓存信息、配置信息和其他中间文件。
    |--- devel :开发空间，用于存放编译后生成的目标文件，包括头文件、动态&静态链接库、可执行文件等。
    |--- src :源码
        |--- package :功能包(ROS基本单元)包含多个节点、库与配置文件，包名所有字母小写，只能由字母、数字与下划线组成
            |--- CMakeLists.txt 配置编译规则，比如源文件、依赖项、目标文件
            |--- package.xml 包信息，比如:包名、版本、作者、依赖项...(以前版本是 manifest.xml)
            |--- scripts 存储python文件、sh脚本文件
            |--- src 存储C++源文件
            |--- include 头文件
            |--- msg 消息通信格式文件
            |--- srv 服务通信格式文件
            |--- action 动作格式文件
            |--- launch 可一次性运行多个节点 
            |--- config 配置信息
        |--- CMakeLists.txt	:编译的基本配置
``` 

## ROS节点通信
### 消息机制：
* 话题通信：单向异步，一般用在是实时性要求不高的场景中，比如传感器广播其采集的数据。
* 服务通信：双向同步，一般用在实时性要求比较高且使用频次底的场景下，比如获取全局静态地图。服务客户端向服务提供端发送请求，服务提供端在收到请求后立即进行处理并返回相应信息。高频次服务通信会导致代码阻塞造成严重后果。
* 动作通信：双向异步，一般用于过程性的任务执行场景下，比如导航任务。动作客户端向动作服务端发送目标，动作服务端要达到目标需要一个过程，动作服务端在执行目标的过程中实时地反馈信息，并在目标完成后返回结果。
#### 话题通信机制
在ROS中，话题(Topic) 和 消息(Message) 是节点间通信的核心机制。
* **话题Topic**是节点间*异步通信*的通道，采用发布-订阅(Publish-Subscribe)模式。有单向数据流，多对多通信。数据流驱动适合持续传输数据，松耦合发布者和订阅者无需知道对方的存在，仅需约定话题名称和消息类型。
* **消息Message**是话题中传输的*数据结构*，定义了通信内容的格式。
#### 服务通信机制
* **服务请求**
* **服务响应**
#### 动作通信机制
* **动作目标**是*动作客户端*向动作服务端发送目标
* **动作反馈**是*动作服务端*向动作客户端发送反馈
* **动作结果**是*动作服务端*向动作客户端发送结果

## launch启动多个ROS节点
XML语法:`<标记名称 属性名1="属性值1" ...> 内容 </标记名称>`
需要注意的是，如果你使用`rosrun`启动节点时，必须先使用`roscore`指令。
而当使用`roslaunch`启动节点时，会自动使用`roscore`指令。

## ROS中的相机话题
* /image_raw: 相机的原始数据
* /image_color: 相机的彩色图像数据
* /image_color_rect: 畸变校正后的彩色图像数据
* /camera_info: 相机相关参数

## ROS中的激光雷达话题

## ROS指令
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
rospack find <package_name>	# 包名package_name

# 安装指定包的依赖
rosdep install <package_name>

# 创建一个新的ROS包
catkin_create_pkg <package_name> [dependencies]	# 通用依赖项dependencies

# 在终端中进入指定软件包的文件地址
roscd <package_name>

# 编译当前工作空间中的所有包
catkin_make

# 编译并安装当前工作空间中的所有包
catkin_make install
```

* 节点管理
```bash
# 要把文件加入环境变量中才能在终端中启动节点#include <
source <workspace_name>/devel/setup.bash

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
rostopic echo <topic_name> --noarr # 显示消息不刷屏

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

# 启动rqt_tf_tree工具，用于查看tf树查看相对变换关系
rosrun rqt_tf_tree rqt_tf_tree

# 启动tf2_tools工具，生成tf树PDF文件
rosrun tf2_tools view_frames.py
```
### ROS消息包
#### 标准消息包std_msgs
|基础类型|数组类型|结构体类型|
| --- | --- | --- |
|`bool`|/|`colorRGBA`|
|`byte`|`byteMultiArray`|`duration`|
|`char`|/|`time`|
|`string`|/|`header`|
|`int8`,`int16`,`int32`,`int64`|`int8MultiArray`,`int16MultiArray`,`int32MultiArray`,`int64MultiArray`|`MultiArrayDimension`|
|`uint8`,`uint16`,`uint32`,`uint64`|`uint8MultiArray`,`uint16MultiArray`,`uint32MultiArray`,`uint64MultiArray`|`MultiArrayLayout`|
|`float32`,`float64`|`float32MultiArray`,`float64MultiArray`|/|
|`empty`|/|/|
#### 几何消息包geometry_msgs
* 加速度：`Accel`,`AccelStamped`,`AccelWithCovariance`,`AccelWithCovarianceStamped`
* 惯量：`Inertia`,`InertiaStamped`
* 空间点：`Point`,`Point32`,`PointStamped`
* 多边形：`Polygon`,`PolygonStamped`
* 空间位置：`Pose`,`Pose2D`,`PoseArray`,`PoseStamped`,`PoseWithCovariance`,`PoseWithCovarianceStamped`
* 四元数：`Quaternion`,`QuaternionStamped`
* 空间变换：`Transform`,`TransformStamped`
* 空降方向：`Twist`,`TwistStamped`,`TwistWithCovariance`,`TwistWithCovarianceStamped`
* 三维矢量：`Vector3`,`Vector3Stamped`
* 扭矩：`Wrench`,`WrenchStamped`
#### 自我诊断消息包diagnostic_msgs
#### 传感器消息包sensor_msgs
* 激光雷达：`LaserScan`,`PointCloud2`,`LaserEcho`,`MultiEchoLaserScan`
* 单点测距：`Range`
* 惯性测量：`Imu`,`MagneticField`
* 彩色相机：`CameraInfo`,`Image`,`CompressedImage`,`RegionOflnterest`
* 立体相机：`CameraInfo`,`Image`,`ChannelFloat32`,`PointCloud`,`PointCloud2`,`PointField`
* 温度测量：`Temperature`
* 湿度测量：`RelativeHumidity`
* 照度测量：`Illuminance`
* 流体压力：`FluidPressure`
* 全球定位：`NavSatFix`,`NavSatStatus`
* 运动关节：`JointState`,`MultiDOFJointState`
* 控制手柄：`Joy`,`JoyFeedback`,`JoyFeedbackArray`
* 电池状态：`BatteryState`
* 时钟源：`TimeReference`
#### 导航消息包nav_msgs
#### 形状消息包shape_msgs
#### 双目视觉消息包stereo_msgs
#### 运动轨迹消息包trajectory_msgs
#### 图形显示消息包visualization_msgs

---

## TF坐标变换
在坐标变换实现中常用的msg：`geometry_msgs/TransformStamped` 和 `geometry_msgs/PointStamped`。  
前者用于传输坐标系相关信息，后者用于传输某个坐标系内坐标点的信息。  
在坐标变换中，频繁的需要使用到坐标系的相对关系以及坐标点信息。