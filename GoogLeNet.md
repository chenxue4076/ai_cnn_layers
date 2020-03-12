GoogLeNet是2014年ImageNet比赛的冠军，它的主要特点是网络不仅有深度，还在横向上具有“宽度”。由于图像信息在空间尺寸上的巨大差异，如何选择合适的卷积核大小来提取特征就显得比较困难了。空间分布范围更广的图像信息适合用较大的卷积核来提取其特征，而空间分布范围较小的图像信息则适合用较小的卷积核来提取其特征。为了解决这个问题，GoogLeNet提出了一种被称为Inception模块的方案。如 图4 所示：

说明：

Google的研究人员为了向LeNet致敬，特地将模型命名为GoogLeNet
Inception一词来源于电影《盗梦空间》（Inception）

https://ai-studio-static-online.cdn.bcebos.com/96f92806a70446e880d87bad5f7b7d8936c36abcb1ff4fcd9c3391ae5fca902d

图4(a)是Inception模块的设计思想，使用3个不同大小的卷积核对输入图片进行卷积操作，并附加最大池化，将这4个操作的输出沿着通道这一维度进行拼接，构成的输出特征图将会包含经过不同大小的卷积核提取出来的特征。Inception模块采用多通路(multi-path)的设计形式，每个支路使用不同大小的卷积核，最终输出特征图的通道数是每个支路输出通道数的总和，这将会导致输出通道数变得很大，尤其是使用多个Inception模块串联操作的时候，模型参数量会变得非常巨大。为了减小参数量，Inception模块使用了图(b)中的设计方式，在每个3x3和5x5的卷积层之前，增加1x1的卷积层来控制输出通道数；在最大池化层后面增加1x1卷积层减小输出通道数。基于这一设计思想，形成了上图(b)中所示的结构。下面这段程序是Inception块的具体实现方式，可以对照图(b)和代码一起阅读。

提示：

可能有读者会问，经过3x3的最大池化之后图像尺寸不会减小吗，为什么还能跟另外3个卷积输出的特征图进行拼接？这是因为池化操作可以指定窗口大小kh=Kw=3k_h = K_w = 3k 
h ​	 =K w ​	 =3，pool_stride=1和pool_padding=1，输出特征图尺寸可以保持不变
 
 
Inception模块的具体实现如下代码所示：

class Inception(fluid.dygraph.Layer):
    def __init__(self, name_scope, c1, c2, c3, c4, **kwargs):
        '''
        Inception模块的实现代码，
        name_scope, 模块名称，数据类型为string
        c1,  图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        c2，图(b)中第二条支路卷积的输出通道数，数据类型是tuple或list, 
               其中c2[0]是1x1卷积的输出通道数，c2[1]是3x3
        c3，图(b)中第三条支路卷积的输出通道数，数据类型是tuple或list, 
               其中c3[0]是1x1卷积的输出通道数，c3[1]是3x3
        c4,  图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        '''
        super(Inception, self).__init__(name_scope)
        # 依次创建Inception块每条支路上使用到的操作
        self.p1_1 = Conv2D(self.full_name(), num_filters=c1, 
                           filter_size=1, act='relu')
        self.p2_1 = Conv2D(self.full_name(), num_filters=c2[0], 
                           filter_size=1, act='relu')
        self.p2_2 = Conv2D(self.full_name(), num_filters=c2[1], 
                           filter_size=3, padding=1, act='relu')
        self.p3_1 = Conv2D(self.full_name(), num_filters=c3[0], 
                           filter_size=1, act='relu')
        self.p3_2 = Conv2D(self.full_name(), num_filters=c3[1], 
                           filter_size=5, padding=2, act='relu')
        self.p4_1 = Pool2D(self.full_name(), pool_size=3, 
                           pool_stride=1,  pool_padding=1, 
                           pool_type='max')
        self.p4_2 = Conv2D(self.full_name(), num_filters=c4, 
                           filter_size=1, act='relu')

    def forward(self, x):
        # 支路1只包含一个1x1卷积
        p1 = self.p1_1(x)
        # 支路2包含 1x1卷积 + 3x3卷积
        p2 = self.p2_2(self.p2_1(x))
        # 支路3包含 1x1卷积 + 5x5卷积
        p3 = self.p3_2(self.p3_1(x))
        # 支路4包含 最大池化和1x1卷积
        p4 = self.p4_2(self.p4_1(x))
        # 将每个支路的输出特征图拼接在一起作为最终的输出结果
        return fluid.layers.concat([p1, p2, p3, p4], axis=1)  


GoogLeNet的架构如 图5 所示，在主体卷积部分中使用5个模块（block），每个模块之间使用步幅为2的3 ×3最大池化层来减小输出高宽。

第一模块使用一个64通道的7 × 7卷积层。
第二模块使用2个卷积层:首先是64通道的1 × 1卷积层，然后是将通道增大3倍的3 × 3卷积层。
第三模块串联2个完整的Inception块。
第四模块串联了5个Inception块。
第五模块串联了2 个Inception块。
第五模块的后面紧跟输出层，使用全局平均池化 层来将每个通道的高和宽变成1，最后接上一个输出个数为标签类别数的全连接层。

说明： 在原作者的论文中添加了图中所示的softmax1和softmax2两个辅助分类器，如下图所示，训练时将三个分类器的损失函数进行加权求和，以缓解梯度消失现象。这里的程序作了简化，没有加入辅助分类器。
https://ai-studio-static-online.cdn.bcebos.com/fcc73767fc944464aa37c4d1112cadab90f5856a0afc49eeaa12db61c9762d0f

GoogLeNet的具体实现如下代码所示：


# -*- coding:utf-8 -*-

# GoogLeNet模型代码
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, FC
from paddle.fluid.dygraph.base import to_variable

# 定义Inception块
class Inception(fluid.dygraph.Layer):
    def __init__(self, name_scope, c1, c2, c3, c4, **kwargs):
        '''
        Inception模块的实现代码，
        name_scope, 模块名称，数据类型为string
        c1,  图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        c2，图(b)中第二条支路卷积的输出通道数，数据类型是tuple或list, 
               其中c2[0]是1x1卷积的输出通道数，c2[1]是3x3
        c3，图(b)中第三条支路卷积的输出通道数，数据类型是tuple或list, 
               其中c3[0]是1x1卷积的输出通道数，c3[1]是3x3
        c4,  图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        '''
        super(Inception, self).__init__(name_scope)
        # 依次创建Inception块每条支路上使用到的操作
        self.p1_1 = Conv2D(self.full_name(), num_filters=c1, 
                           filter_size=1, act='relu')
        self.p2_1 = Conv2D(self.full_name(), num_filters=c2[0], 
                           filter_size=1, act='relu')
        self.p2_2 = Conv2D(self.full_name(), num_filters=c2[1], 
                           filter_size=3, padding=1, act='relu')
        self.p3_1 = Conv2D(self.full_name(), num_filters=c3[0], 
                           filter_size=1, act='relu')
        self.p3_2 = Conv2D(self.full_name(), num_filters=c3[1], 
                           filter_size=5, padding=2, act='relu')
        self.p4_1 = Pool2D(self.full_name(), pool_size=3, 
                           pool_stride=1,  pool_padding=1, 
                           pool_type='max')
        self.p4_2 = Conv2D(self.full_name(), num_filters=c4, 
                           filter_size=1, act='relu')

    def forward(self, x):
        # 支路1只包含一个1x1卷积
        p1 = self.p1_1(x)
        # 支路2包含 1x1卷积 + 3x3卷积
        p2 = self.p2_2(self.p2_1(x))
        # 支路3包含 1x1卷积 + 5x5卷积
        p3 = self.p3_2(self.p3_1(x))
        # 支路4包含 最大池化和1x1卷积
        p4 = self.p4_2(self.p4_1(x))
        # 将每个支路的输出特征图拼接在一起作为最终的输出结果
        return fluid.layers.concat([p1, p2, p3, p4], axis=1)  
    
class GoogLeNet(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(GoogLeNet, self).__init__(name_scope)
        # GoogLeNet包含五个模块，每个模块后面紧跟一个池化层
        # 第一个模块包含1个卷积层
        self.conv1 = Conv2D(self.full_name(), num_filters=64, filter_size=7, 
                            padding=3, act='relu')
        # 3x3最大池化
        self.pool1 = Pool2D(self.full_name(), pool_size=3, pool_stride=2,  
                            pool_padding=1, pool_type='max')
        # 第二个模块包含2个卷积层
        self.conv2_1 = Conv2D(self.full_name(), num_filters=64, 
                              filter_size=1, act='relu')
        self.conv2_2 = Conv2D(self.full_name(), num_filters=192, 
                              filter_size=3, padding=1, act='relu')
        # 3x3最大池化
        self.pool2 = Pool2D(self.full_name(), pool_size=3, pool_stride=2,  
                            pool_padding=1, pool_type='max')
        # 第三个模块包含2个Inception块
        self.block3_1 = Inception(self.full_name(), 64, (96, 128), (16, 32), 32)
        self.block3_2 = Inception(self.full_name(), 128, (128, 192), (32, 96), 64)
        # 3x3最大池化
        self.pool3 = Pool2D(self.full_name(), pool_size=3, pool_stride=2,  
                               pool_padding=1, pool_type='max')
        # 第四个模块包含5个Inception块
        self.block4_1 = Inception(self.full_name(), 192, (96, 208), (16, 48), 64)
        self.block4_2 = Inception(self.full_name(), 160, (112, 224), (24, 64), 64)
        self.block4_3 = Inception(self.full_name(), 128, (128, 256), (24, 64), 64)
        self.block4_4 = Inception(self.full_name(), 112, (144, 288), (32, 64), 64)
        self.block4_5 = Inception(self.full_name(), 256, (160, 320), (32, 128), 128)
        # 3x3最大池化
        self.pool4 = Pool2D(self.full_name(), pool_size=3, pool_stride=2,  
                               pool_padding=1, pool_type='max')
        # 第五个模块包含2个Inception块
        self.block5_1 = Inception(self.full_name(), 256, (160, 320), (32, 128), 128)
        self.block5_2 = Inception(self.full_name(), 384, (192, 384), (48, 128), 128)
        # 全局池化，尺寸用的是global_pooling，pool_stride不起作用
        self.pool5 = Pool2D(self.full_name(), pool_stride=1, 
                               global_pooling=True, pool_type='avg')
        self.fc = FC(self.full_name(),  size=1)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        x = self.pool3(self.block3_2(self.block3_1(x)))
        x = self.block4_3(self.block4_2(self.block4_1(x)))
        x = self.pool4(self.block4_5(self.block4_4(x)))
        x = self.pool5(self.block5_2(self.block5_1(x)))
        x = self.fc(x)
        return x

with fluid.dygraph.guard():
    model = GoogLeNet("GoogLeNet")

train(model)

