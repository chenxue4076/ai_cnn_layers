VGG是当前最流行的CNN模型之一，2014年由Simonyan和Zisserman提出，其命名来源于论文作者所在的实验室Visual Geometry Group。AlexNet模型通过构造多层网络，取得了较好的效果，但是并没有给出深度神经网络设计的方向。VGG通过使用一系列大小为3x3的小尺寸卷积核和pooling层构造深度卷积神经网络，并取得了较好的效果。VGG模型因为结构简单、应用性极强而广受研究者欢迎，尤其是它的网络结构设计方法，为构建深度神经网络提供了方向。

图3 是VGG-16的网络结构示意图，有13层卷积和3层全连接层。VGG网络的设计严格使用3×33\times 33×3的卷积层和池化层来提取特征，并在网络的最后面使用三层全连接层，将最后一层全连接层的输出作为分类的预测。 在VGG中每层卷积将使用ReLU作为激活函数，在全连接层之后添加dropout来抑制过拟合。使用小的卷积核能够有效地减少参数的个数，使得训练和测试变得更加有效。比如使用两层3×33\times 33×3卷积层，可以得到感受野为5的特征图，而比使用5×55 \times 55×5的卷积层需要更少的参数。由于卷积核比较小，可以堆叠更多的卷积层，加深网络的深度，这对于图像分类任务来说是有利的。VGG模型的成功证明了增加网络的深度，可以更好的学习图像中的特征模式。

https://ai-studio-static-online.cdn.bcebos.com/f22dc04130f24933865fbd82e61fb8e2e15849dc4a2e411ea785dea239de6cfe

VGG在眼疾识别数据集iChallenge-PM上的具体实现如下代码所示：

# -*- coding:utf-8 -*-

# VGG模型代码
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, FC
from paddle.fluid.dygraph.base import to_variable

# 定义vgg块，包含多层卷积和1层2x2的最大池化层
class vgg_block(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_convs, num_channels):
        """
        num_convs, 卷积层的数目
        num_channels, 卷积层的输出通道数，在同一个Incepition块内，卷积层输出通道数是一样的
        """
        super(vgg_block, self).__init__(name_scope)
        self.conv_list = []
        for i in range(num_convs):
            conv_layer = self.add_sublayer('conv_' + str(i), Conv2D(self.full_name(), 
                                        num_filters=num_channels, filter_size=3, padding=1, act='relu'))
            self.conv_list.append(conv_layer)
        self.pool = Pool2D(self.full_name(), pool_stride=2, pool_size = 2, pool_type='max')
    def forward(self, x):
        for item in self.conv_list:
            x = item(x)
        return self.pool(x)

class VGG(fluid.dygraph.Layer):
    def __init__(self, name_scope, conv_arch=((2, 64), 
                                (2, 128), (3, 256), (3, 512), (3, 512))):
        super(VGG, self).__init__(name_scope)
        self.vgg_blocks=[]
        iter_id = 0
        # 添加vgg_block
        # 这里一共5个vgg_block，每个block里面的卷积层数目和输出通道数由conv_arch指定
        for (num_convs, num_channels) in conv_arch:
            block = self.add_sublayer('block_' + str(iter_id), 
                    vgg_block(self.full_name(), num_convs, num_channels))
            self.vgg_blocks.append(block)
            iter_id += 1
        self.fc1 = FC(self.full_name(),
                      size=4096,
                      act='relu')
        self.drop1_ratio = 0.5
        self.fc2= FC(self.full_name(),
                      size=4096,
                      act='relu')
        self.drop2_ratio = 0.5
        self.fc3 = FC(self.full_name(),
                      size=1,
                      )
    def forward(self, x):
        for item in self.vgg_blocks:
            x = item(x)
        x = fluid.layers.dropout(self.fc1(x), self.drop1_ratio)
        x = fluid.layers.dropout(self.fc2(x), self.drop2_ratio)
        x = self.fc3(x)
        return x



with fluid.dygraph.guard():
    model = VGG("VGG")

train(model)

通过运行结果可以发现，在眼疾筛查数据集iChallenge-PM上使用VGG，loss能有效的下降，经过5个epoch的训练，在验证集上的准确率可以达到94%左右。
