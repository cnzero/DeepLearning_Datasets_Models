This repro is learned from [pytorch-template](https://github.com/victoresque/pytorch-template).

I want to use this template to re-build many classical models of deep learning.

Yes, I am studying PyTorch and how to build good pipelined projects.

Let's enjoy it!!!


# 定义需求
机器学习或深度学习的训练和测试过程，自有固定的pipeline，这是很多工程中的相似之处；

而不同之处主要在于：
1. 不同的数据集及其预处理；
2. 不同的模型；

因此，希望“相似之处”能够固化下来，而在研究探索中，将绝大多数的重点关注在“不同之处”上；
然后让这种数据和模型，能在一个自动化的pipeline上运行。

[pytorch-template](https://github.com/victoresque/pytorch-template) 就非常关注其中的“相似之处”与“不同之处”。
因此，站在“巨人的肩膀上”，我将陆续进行经典的模型复现工作，希望最终达到的目标是：

```
python train.py --config Dataset1_ModelA_config.json
# or
python test.py --resume xxxxx/best_model.pth
```

就能够看到最终的效果。

因此，将工作内容主要关注在“数据”和“模型”两个方面。根据pytorch-template的框架，主要的工作在于：
1. `data_loader.data_loaders.py` 
2. `model.Models.py` 
3. `xxx_config.json` 


# sEMG-based Recognition
另外，将在尽快复现了经典模型之后，专注研究“sEMG"方向，依赖的数据集主要为NinaPro, CSL, Capg等 -- 尚未取得可以言说的进展，暂不表述。

# 关于pytorch-template的执行
## 实现关键模块，修改配置参数
如果创建pytorch的工程项目，还是推荐[pytorch-template](https://github.com/victoresque/pytorch-template)的第一手资料。如下，我仅仅是将其中部分内容稍作记录。

以`MNIST_LeNet` 为例，
1. 数据
    1. 在`data_loader.data_loaders.py` 创建并实现`MNIST_DataLoader`;
    2. 将该名称`MNIST_DataLoader`写入到`MNIST_LeNet_config.json` 的`"data_loader":{"MNIST_DataLoader", }` 中，稍微注意其他相关参数，如数据的存储路径等；
2. 模型
    1. 在`model.Models.py` 创建并实现`class MNIST_LeNet()`;
    2. 将该名称`MNIST_LeNet`写入到`MNIST_LeNet_config.json`的`"arch":"MNIST_LeNet"`中；
3. 配置文件。关注`MNIST_LeNet_config.json` 这个配置文件中的更多相关参数

## 执行
```python train.py --config MNIST_LeNet_config.json```

对模型进行训练，训练结果或断点结果均有所保存，即使中断了也无所谓，下次可以重新持续训练。

```python train.py --resume saved/models/MNIST_LeNet/time/xxx.pth```

找到相应的断点存储结果，可以持续训练。

由于其中的参数设置，如果完整训练完成，会有一个`best_model.pth` 

```python test.py --resume saved/models/MNIST_LeNet/time/xxx.pth```

即，利用某个存储的模型进行测试。


