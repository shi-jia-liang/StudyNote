# pdb模块
**import pdb;pdb.set_trace()** 是 Python 中用于调试代码的常用技巧，它的作用相当于在代码中设置一个断点（breakpoint）。  
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