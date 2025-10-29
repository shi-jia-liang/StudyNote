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