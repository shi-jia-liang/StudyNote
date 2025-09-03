# pdb模块
pdb 是 Python 中用于调试代码的常用技巧，它的作用相当于在代码中设置一个断点（breakpoint）。  
* 安装：pip install pdb
* 使用：**import pdb;pdb.set_trace()**  

用途：暂停程序执行，控制权交给开发者；终端显示 **(PDB)** 提示符，表示已进入调试模式。  
在 **(PDB)** 提示符下，可以输入各种命令来检查和操作程序状态：
* l(ist)：查看当前执行位置周围的代码
* n(ext)：执行下一行代码（不进入函数）
* s(tep)：执行下一行代码（进入函数内部）
* c(ontinue)：继续执行直到下一个断点或程序结束
* p\<expression\>：打印表达式的值（如 p variable_name）
* pp\<expression\>：漂亮打印复杂数据结构
* w(here)：显示当前调用栈
* u(p)：在调用栈中上移一级
* d(own)：在调用栈中下移一级
* q(uit)：退出调试器并终止程序

# kornia模块
Kornia 是一款基于 PyTorch 的可微分的计算机视觉库。  
* 安装：pip install kornia
* 使用：**import kornia**    

它由一组用于解决通用计算机视觉问题的操作模块和可微分模块组成。其核心使用 PyTorch 作为主要后端，以提高效率并利用反向模式自动微分来定义和计算复杂函数的梯度。  
受现有开源库的启发，Kornia可以由包含各种可以嵌入神经网络的操作符组成，并可以训练模型来执行图像变换、对极几何、深度估计和低级图像处理，例如过滤和边缘检测。此外，整个库都可以直接对张量进行操作。

# tqdm模块
tqdm 是 python 中用于显示进度条的模块，它常被用于显示训练、评估数据集的进度。
* 安装：pip install tqdm
* 使用：**from tqdm import tqdm**