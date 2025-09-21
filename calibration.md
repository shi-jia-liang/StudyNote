# 标定
## 惯性里程计IMU标定
使用imu_utils进行标定
### 依赖项
* 安装依赖项
```bash
sudo apt-get install libdw-dev
sudo apt-get install libeigen3-dev
```
* 安装cere-solver
```bash
# CMake
sudo apt-get install cmake
# google-glog + gflags
sudo apt-get install libgoogle-glog-dev libgflags-dev
# Use ATLAS for BLAS & LAPACK
sudo apt-get install libatlas-base-dev
# Eigen3
sudo apt-get install libeigen3-dev
# SuiteSparse (optional)
sudo apt-get install libsuitesparse-dev

tar zxf ceres-solver-2.2.0.tar.gz
mkdir ceres-bin
cd ceres-bin
cmake ../ceres-solver-2.2.0
make -j${CPU thread}
make test
# Optionally install Ceres, it can also be exported using CMake which
# allows Ceres to be used without requiring installation, see the documentation
# for the EXPORT_BUILD_DIR option for more information.
sudo make install

# test cere-solver
bin/simple_bundle_adjuster ../ceres-solver-2.2.0/data/problem-16-22106-pre.txt
```

### 先编译code_utils
克隆文件
```bash
cd catkin_imu/src
git clone https://github.com/gaowenliang/code_utils 
```
修改文件中的错误
* mat_io_test.cpp
```bash
# 33 |
Mat img1 = imread( "/home/gao/IMG_1.PNG", IMREAD_UNCHANGED );
```
* sumpixel_test.cpp
```bash
# 2 |
include "code_utils/backward.hpp"
# 84 |
Mat img1 = imread( "/home/gao/IMG_1.PNG", IMREAD_GRAYSCALE );
# 94 |
normalize( img, img2, 0, 255, NORM_MINMAX );
# 107 |
Mat img1 = imread( "/home/gao/IMG_1.PNG", IMREAD_GRAYSCALE );
# 117 |
normalize( img, img2, 0, 255, NORM_MINMAX );
```
编译
```bash
cd ..
catkin_make
```

### 后编译imu_utils
克隆文件
```bash
cd catkin_imu/src
git clone https://github.com/gaowenliang/imu_utils
```
修改文件中的错误
* imu_an.cpp
```bash
# 添加头文件
#include <fstream>
```
编译
```bash
cd ..
catkin_make
```

## 相机标定
使用kalibr工具箱进行标定
### 安装文档
[请参考官方安装文档](https://github.com/ethz-asl/kalibr/wiki/installation)

