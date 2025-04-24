# CMake
&emsp;&emsp;在CMake指令不区分大小写
## 指定要求最小的CMake版本
cmake_minimun_required(VERSION {`XXX`})

## 设置当前项目名称
project(projectName)

## 搜索第三方库
find_package(packageName version EXACT/QUIET/REQUIRED)

## 指定头文件的搜索路径，方便编译器查找相应头文件
include_directories(${packageName_INCLUDE_DIRS})

## 打印输出信息
message(mode "It's message")
mode:{FATAL_ERROR、WARNING、STATUS、DEBUG}