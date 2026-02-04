# def函数 和 class类的区别
* `def`定义一个函数（或者方法），主要用来实现一段可复用的逻辑，不持久保存状态（除非使用闭包或者外部变量）。
	* 目的：封装单一、独立的计算或操作（例如计算损失、数据预处理、评估指标函数）。
	* 状态：无状态（一次调用完成后不保留内部状态），但可以通过返回值、闭包、或外部变量保持状态。
	* 组织结构：简单、直接，易于测试和组合。
	* 调用约定：直接调用 f(args...)。
	* 可扩展性：难以通过继承扩展（可通过组合或高阶函数）。
* `class`定义一个类型/蓝图，可以创建带状态（属性）和行为（方法）的对象，用来组织相关数据与操作
	* 目的：把数据（属性）和行为（方法）绑在一起，适合需要状态、生命周期或多方法协作的模块。
	* 状态：对象实例拥有属性（如 self.matcher, self.total_ms），适合保存训练过程中需要持续追踪的值。
	* 组织结构：方便封装复杂行为、继承复用、实现接口。
	* 调用约定：先实例化 `obj = MyClass(...)`，然后用 `obj.method(...)`；类也可以定义 `__call__` 使其实例可直接调用。
	* 可扩展性：支持继承、重写、mixins，适合需要被扩展的复杂组件。

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

# pytorch模块
PyTorch 是一个开源的深度学习框架，广泛应用于计算机视觉、自然语言处理等领域。它由 Facebook 的人工智能研究团队开发，提供了灵活且高效的张量计算和自动微分功能。  
* 安装：[Pytorch](https://pytorch.org/) # 建议选择GPU版本，根据CUDA版本选择对应命令
* 使用：**import torch torchvision**
## from torch import nn
构建和训练神经网络的核心模块，它提供了丰富的类和函数来定义和操作神经网络。
### 神经网络模型 nn.Module
nn.Module是所有自定义神经网络模型的基类。用户通常会从这个类派生自己的模型类，并在其中定义网络层结构以及前向传播函数（forward pass）。
#### 预定义神经网络模型 nn.Modules
nn.Modules是nn.Module的子类，用于定义神经网络中的各种层和操作。常见的nn.Modules包括：
* 全连接层（`nn.Linear`）
* 卷积层（`nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`）
* 池化层（`nn.MaxPool2d`, `nn.AvgPool2d`）
* 归一化层（`nn.BatchNorm2d`, `nn.LayerNorm`）
* 激活函数（`nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`）
### 自定义神经网络模型 nn.Sequential
nn.Sequential是序列容器，用于搭建神经网络的模块按照顺序添加到容器中，即将多个模块封装成一个模块。

以下是三种常用的模块封装方式：

* nn.Sequential: 允许将多个层按顺序组合起来，形成简单的线性堆叠网络
* nn.ModuleList: 按python的list一样包装多个网络层，可以动态地存储和访问子模块
* nn.ModuleDict: 按python的dict一样包装多个网络层，可以动态地存储和访问子模块

### 损失函数（Loss Functions）
#### 分类损失函数
#### 回归损失函数
* L1范数损失 **l1_loss**
* **huber_loss**
* 平滑版L1损失 **smooth_l1_loss**
* 均方误差损失 **mse_loss**
* **margin_ranking_loss**
* 铰链嵌入损失 **hinge_embedding_loss**
* 多分类间隔损失 **multilabel_margin_loss**
* **soft_margin_loss**
* **multilabel_soft_margin_loss**
* **cosine_embedding_loss**
* **multi_margin_loss**
* 交叉熵损失 **cross_entropy** :当训练有 C 个类别的分类问题时很有效. 可选参数 weight 必须是一个1维 Tensor, 权重将被分配给各个类别. 对于不平衡的训练集非常有效。
* 二分类交叉熵损失 **binary_cross_entropy** :二分类任务时的交叉熵计算函数。用于测量重构的误差, 例如自动编码机. 注意目标的值 t[i] 的范围为0到1之间.
* 带有 Logits 的二元交叉熵 **binary_cross_entropy_with_logits** :BCEWithLogitsLoss损失函数把 Sigmoid 层集成到了 BCELoss 类中. 该版比用一个简单的 Sigmoid 层和 BCELoss 在数值上更稳定, 因为把这两个操作合并为一个层之后, 可以利用 log-sum-exp 的 技巧来实现数值稳定.
* 散度损失 **kl_div** :KL 散度可用于衡量不同的连续分布之间的距离, 在连续的输出分布的空间上(离散采样)上进行直接回归时 很有效。
* 连接时序分类损失 **ctc_loss** :可以对没有对齐的数据进行自动对齐，主要用在没有事先对齐的序列化数据训练上。比如语音识别、ocr识别等。
* 负对数似然损失 **nll_loss** :用于训练 C 个类别的分类问题。
* 目标值为泊松分布的负对数似然损失 **poisson_nll_loss**
* 目标值为高斯分布的负对数似然损失 **gaussian_nll_loss**

### 实用函数接口 nn.functional
nn.functional（通常简写为 F），包含了许多可以直接作用于张量上的函数，它们实现了与层对象相同的功能，但不具有参数保存和更新的能力。例如，可以使用 F.relu() 直接进行 ReLU 操作，或者 F.conv2d() 进行卷积操作。

# pytorch_lightning模块
Pytorch Lightning 是一个基于PyTorch的高级深度学习框架，旨在将科研代码的灵活性与工程化最佳实习结合，通过标准化训练流程大幅减少模板代码。它的核心思想是，将**学术代码**（模型定义、前向/反向传播、优化器、验证等）和**工程代码**（for-loop、保存、tensorboard日志、训练策略等）解耦开来，使得代码更加简洁。  

旧包名：pytorch_lightning  

新包名：lightning
* 安装：pip install lightning
* 使用：**import lightning as pl**
1. 定义`LightningDataModule`加载数据模块 class datasets(pl.LightningDataModule):  
	step1：初始化 def \_\_init\_\_(self):   
	step2：下载数据集(可选) def prepare_data(self):  
	step3：划分数据集 def setup(self, stage: Optional[str] = None):  
	step4：加载训练数据集 def train_dataloader(self):  
	step5：加载验证数据集 def val_dataloader(self):  
	step6：加载测试数据集 def test_dataloader(self):  
2. 定义`LightningModule`模型模块 class model(pl.LightningModule):  
	step1：初始化 def \_\_init\_\_(self):  
	step2：前向推理 def forward(self, x):  
	step3：设置优化器 def configure_optimizers(self):  
	step4：训练过程 def training_step(self, train_batch, batch_idx):  
	step5：测试过程 def test_step(self, test_batch, batch_idx):  
	step6：验证过程 def validation_step(self, val_batch, batch_idx):  
	step7：反向学习 def backward(self, x):  
3. 定义`Trainer()`类 trainer=pl.Trainer()
	* 基础参数配置
		* max_epochs：最大训练轮次
		* min_epochs：最小训练轮次
		* max_steps：最大训练步数
		* min_steps：最小训练步数
		* accelerator：硬件加速器（`CPU`, `GPU`, `TPU`, `auto`）
		* devices：训练的设备数量
	* 训练参数配置
		* gradient_clip_val：梯度裁剪阈值
		* accumulate_grad_batches：梯度累积步数（模拟更大batch）
		* limit_train_batches：限制每epoch训练batch数（如 0.1 表示10%）
		* limit_val_batches：限制验证batch数
		* val_check_interval：验证频率（1.0=每epoch，0.5=每半个epoch）
		* check_val_every_n_epoch：每N轮验证一次（默认 1）
		* fast_dev_run：快速运行少量batch（如 True 或 5）
		* overfit_batches：过拟合少量batch（测试代码）
		* precision：精度
	* 回调日志
		* logger：日志器（`TersonBoardLogger`, `WandbLogger`, `CometLogger`, `MLFlowLogger`, `NeptuneLogger`）
		* callbacks：回调列表（`ModelCheckpoint`(保存模型权重), `EarlyStopping`(早停机制)）
		* log_every_n_steps：每N步记录一次日志（默认 50）
4. 额外功能
	* 开始训练 trainer.fit(model=`model`, train_dataloaders=`train_dataloader`, val_dataloaders=`val_dataloader`)
	* 保存超参数，即可自动保存所有传递给 init 的超参数 self.save_hyperparameters()
	* 恢复训练状态 trainer.fit(model, ckpt_path="path/to/your/checkpoint.ckpt")

模型权重文件格式转换：`.pt`, `.onnx`, `ckpt` 

# loguru模块
Loguru 是一个用于 Python 的强大且易用的日志记录库，旨在简化日志记录过程，同时提供丰富的功能和灵活性。
* 安装：pip install loguru
* 使用：**from loguru import logger**

主要功能：
* 简单易用：只需几行代码即可开始记录日志，无需复杂的配置。
* 丰富的日志级别：支持多种日志级别（DEBUG、INFO、WARNING、ERROR、CRITICAL），便于分类和过滤日志信息。
* 灵活的日志输出：支持将日志输出到控制台、文件、远程服务器等多种目标。
* 自动轮转日志文件：支持基于时间或文件大小自动轮转日志文件，便于日志管理。
* 结构化日志：支持 JSON 格式的结构化日志，便于日志分析。
* 异常捕获：内置异常捕获功能，自动记录未处理的异常信息。
* 线程安全：设计为线程安全，适用于多线程应用程序。	

## 关于PyCharm
&emsp;&emsp;`<2025.2.18>` 遇到很傻逼的问题，在cmd中使用pip install会根据系统安装到指定文件夹，而在PyCharm的命令行中使用pip install会安装到C盘的用户站中。建议以后都用cmd去安装python包。(当你不需要虚拟环境时)
&emsp;&emsp;解决方案:确定了是IDE的问题，进入IDE后会直接进入conda环境，而安装文件的路径也是conda的安装路径，在cmd中使用的是python的安装路径。如果不希望conda自动激活`base`环境，可以修改conda的配置：
1. 打开终端，运行以下命令：
   ```bash
   conda config --set auto_activate_base false
   ```
2. 重启终端或 IDE，Conda 将不再自动激活 base 环境。