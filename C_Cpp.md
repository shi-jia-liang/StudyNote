# C/C++ Primer Plus
## OepnCV C语言环境安装
### C/C++
* 使用[Msys2](https://www.msys2.org/)，一个类似于Liunx内核的包管理系统。

可以按照官方教程安装`mingw-w64-ucrt-x86_64-gcc`，个人喜欢安装`mingw-w64-x86_64-gcc`。

* 初始化环境

```bash
pacman -Syu
```

* 以下附上我所安装的所有包(**安装mingw-w64-x86_64-gcc工具链**)
	
```bash
pacman -S --needed base-devel mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake gcc gdb
```

* 添加环境变量  
![环境变量](./Img/C_CPP/EnvironmentVariables.png)

### 安装CMake
* 按照[CMake](https://cmake.org/download/)网站提供的msi文件安装  

![CMake官网](./Img/C_CPP/CMake.png)

### 安装OpenCV
此为C/C++语言环境安装OpenCV，首先打开CMake-gui。([可按照此篇文档进行安装](https://blog.csdn.net/SirClown/article/details/142614854))

需要注意的是，一定要勾选**BUILD_opencv_world**，它的作用是将所有库文件使用动态库链接的方式存储，方便后续设置环境变量只用设置一个文件夹。

编译源文件为下载的OpenCV整个源文件，而添加的*OPENCV_EXTRA_MODULES_PATH*为`opencv_contrib\modules`。

在编译完成后，需要在build文件夹中，使用cmd输入**mingw32-make**编译生成文件，最后使用**mingw32-make install**进行安装。

安装后的所有文件保存在`build\install`中，我们需要设置`\build\install\x64\mingw\bin`添加入环境变量中。
