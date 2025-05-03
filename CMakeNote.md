# CMake
&emsp;&emsp;在CMake指令不区分大小写
## 指定 CMake 的最低版本要求
cmake_minimum_required(VERSION <version>)

## 定义项目的名称和使用的编程语言
project(<project_name> [<language>...])

## 设置变量的值
set(<variable> <value>...)
### 设置 C++ 标准为 C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## 搜索第三方库
find_package(packageName version EXACT/QUIET/REQUIRED)

## 链接目标文件与其他库
target_link_libraries(<target> <libraries>...)

## 指定头文件的搜索路径，方便编译器查找相应头文件
include_directories(<dirs>...)
include_directories(${packageName_INCLUDE_DIRS})

## 创建一个库（静态库或动态库）及其源文件
add_library(<target> <source_files>...)

## 指定要生成的可执行文件和其源文件
add_executable(<target> <source_files>...)
add_executable(MyExecutable main.cpp other_file.cpp)

## 设置目标属性
target_include_directories(TARGET target_name
                          [BEFORE | AFTER]
                          [SYSTEM] [PUBLIC | PRIVATE | INTERFACE]
                          [items1...])

## 安装规则
install(TARGETS target1 [target2 ...]
        [RUNTIME DESTINATION dir]
        [LIBRARY DESTINATION dir]
        [ARCHIVE DESTINATION dir]
        [INCLUDES DESTINATION [dir ...]]
        [PRIVATE_HEADER DESTINATION dir]
        [PUBLIC_HEADER DESTINATION dir])

## 条件语句
if(expression)
  # Commands
elseif(expression)
  # Commands
else()
  # Commands
endif()

## 打印输出信息
message(mode "It's message")
mode:{FATAL_ERROR、WARNING、STATUS、DEBUG}