AlexNet与LeNet相比，具有更深的网络结构，包含5层卷积和3层全连接，同时使用了如下三种方法改进模型的训练过程：

数据增多：深度学习中常用的一种处理方式，通过对训练随机加一些变化，比如平移、缩放、裁剪、旋转、翻转或者增减亮度等，产生一系列跟原始图片相似但又不完全相同的样本，从而扩大训练数据集。通过这种方式，可以随机改变训练样本，避免模型过度依赖于某些属性，能从一定程度上抑制过拟合。

使用Dropout抑制过拟合

使用ReLU激活函数少梯度消失现象

https://ai-studio-static-online.cdn.bcebos.com/d28eb5b9ad2c458a9922e528312486eb994529fdfabd4a9488c33746a3e1dac7

# -*- coding:utf-8 -*-

# 导入需要的包
import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC


# 定义 AlexNet 网络结构
class AlexNet(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_classes=1):
        super(AlexNet, self).__init__(name_scope)
        name_scope = self.full_name()
        # AlexNet与LeNet一样也会同时使用卷积和池化层提取图像特征
        # 与LeNet不同的是激活函数换成了‘relu’
        self.conv1 = Conv2D(name_scope, num_filters=96, filter_size=11, stride=4, padding=5, act='relu')
        self.pool1 = Pool2D(name_scope, pool_size=2, pool_stride=2, pool_type='max')
        self.conv2 = Conv2D(name_scope, num_filters=256, filter_size=5, stride=1, padding=2, act='relu')
        self.pool2 = Pool2D(name_scope, pool_size=2, pool_stride=2, pool_type='max')
        self.conv3 = Conv2D(name_scope, num_filters=384, filter_size=3, stride=1, padding=1, act='relu')
        self.conv4 = Conv2D(name_scope, num_filters=384, filter_size=3, stride=1, padding=1, act='relu')
        self.conv5 = Conv2D(name_scope, num_filters=256, filter_size=3, stride=1, padding=1, act='relu')
        self.pool5 = Pool2D(name_scope, pool_size=2, pool_stride=2, pool_type='max')

        self.fc1 = FC(name_scope, size=4096, act='relu')
        self.drop_ratio1 = 0.5
        self.fc2 = FC(name_scope, size=4096, act='relu')
        self.drop_ratio2 = 0.5
        self.fc3 = FC(name_scope, size=num_classes)

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)

        x = self.fc1(x)
        # 在全连接之后使用dropout抑制过拟合
        x= fluid.layers.dropout(x, self.drop_ratio1)
        x = self.fc2(x)
        # 在全连接之后使用dropout抑制过拟合
        x = fluid.layers.dropout(x, self.drop_ratio2)
        x = self.fc3(x)
        return x


with fluid.dygraph.guard():
    model = AlexNet("AlexNet")

train(model)


