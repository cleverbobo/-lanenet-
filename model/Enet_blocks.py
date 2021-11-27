import torch.nn as nn
import torch

class InitialBlock(nn.Module):
    """ Enet的初始化block，主要包含两个分支
    第一个为主要分支：经典的卷积核，步长为2，默认大小为3*3*13
    第二个为辅助分支：最大池化，步长为2，大小为2*2
    最后将将两个分支连接到一起，总共会有16个特征地图
    关键参数：
    - 输入维数（整型）
    - 输出维数（整型）
    - 卷积核尺寸（整型，可选项）：默认为3
    - 边缘填充（整型，可选项）：默认填充0
    - 偏置（布尔型，可选项）：默认不使用
    - relu（布尔型，可选项）；默认为true，当为true时，使用relu激活
    否则使用Prelu激活
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=0,
                 bias=False,
                 relu=True):
        super(InitialBlock,self).__init__()
        # 根据输入选择合适的激活函数
        if relu :
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        # 设置主要的的卷积分支的部分
        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 5,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=bias)

        # 设置的辅助最大池化分支
        self.ext_branch = nn.MaxPool2d(kernel_size, stride=2, padding=padding)

        # 在两个分支的结果联结后使用BN算法+relu激活
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_prelu = activation

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # 将两个分支的结果进行联结
        out = torch.cat((main, ext), 1)

        # 应用BN算法
        out = self.batch_norm(out)

        return self.out_prelu(out)


class RegularBottleneck(nn.Module):
    """最普通的卷积 bottleneck结构
    一、主要分支
    1. 直连部分

    二、其他部分
    1. 1*1的卷积核，用于降低通道维数
    2. 经典/空洞/不对称卷积
    3. 1*1的卷积核，用于提高通道维数
    4. 使用dropout作为正则化的手段

    关键参数:
    - 输入和输出的通道数（整型）
    - 压缩比（整型），默认是4
    - 卷积核尺寸（整型），默认是3
    - 边缘填充（整型），默认0填充
    - 空洞卷积（整型）、默认空格为0
    - 不对称卷积（布尔型）、默认不使用
    - dropout比率（浮点型），默认为0.5
    - 偏置（布尔型）、默认不使用
    - relu激活（布尔型），否则使用Prelu
    """

    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0.5,
                 bias=False,
                 relu=True):
        super(RegularBottleneck,self).__init__()

        # 检测internal_ratio是否在预期的范围内
        # 预期的范围，整型，取值区间[1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))
        # 压缩后的通道数
        internal_channels = channels // internal_ratio

        # 激活函数设置
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # 第一个部分：主要分支（直连），所以为空

        # 第二个部分：额外的分支
        # 1x1 的压缩卷积 + bn + relu激活
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 5*5的非对称卷积，刚开始是3*1，之后是1*3
        # 卷积 + BN + relu激活
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation,
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation)
        # （空洞）卷积 + BN + relu激活
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 扩张卷积
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(channels), activation)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # 激活函数
        self.out_prelu = activation

    def forward(self, x):
        # 主要分支
        main = x

        # 副主分支
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # 将两个结果直接相加
        out = main + ext

        return self.out_prelu(out)


class DownsamplingBottleneck(nn.Module):
    """具有下采样的bottleneck
    主要分支：
    1. 最大池化，步长为2; 索引会被保存，以供后面上采样时使用

    辅助分支:
    1. 2*2的卷积核用于减少维数
    2. 经典的卷积操作，默认3*3
    3. 1x1的卷积核，用于上采样
    4. 使用dropout作为正则化.
    关键参数:
    - 输入维数（整型）
    - 输出维度（整型）
    - internal_ratio (int, optional)
    - kernel_size (int, optional):默认为3
    - padding (int, optional): 默认为0
    - dilation (int, optional): 默认为0
    - asymmetric (bool, optional): 默认不使用
    - return_indices (bool, optional):  返回最大池化时的位置信息，在上采样时有用
    - dropout_prob (float）
    - bias (bool）
    - relu (bool）
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 return_indices=True,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super(DownsamplingBottleneck,self).__init__()

        # 存储位置信息，后面会调用
        self.return_indices = return_indices

        # 检查internal_ratio的范围是否有效
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        # 激活函数设置
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # 主要分支，最大池化分支
        self.main_max1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=padding-1,
            return_indices=return_indices)

        # 辅助分支

        # 2x2 的卷积 步长为2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                dilation=dilation,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 卷积操作
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 卷积核，用于上采样
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                dilation=dilation,
                bias=bias), nn.BatchNorm2d(out_channels), activation)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # 激活函数设置
        self.out_prelu = activation

    def forward(self, x):
        # 主要分支
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # 辅助分支
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # M主要分支的paddng
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # 转换为GPU类型
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # 将两个分支的结果合并
        out = main + ext

        return self.out_prelu(out), max_indices



def AddCoords(input_tensor, with_r=False):
    # self.with_r = with_r
    # 各个维度尺寸
    batch_size, _, x_dim, y_dim = input_tensor.size()

    
    xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
    yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

    xx_channel = xx_channel.float() / (x_dim - 1)
    yy_channel = yy_channel.float() / (y_dim - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
    yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

    ret = torch.cat([
        input_tensor,
        xx_channel.type_as(input_tensor),
        yy_channel.type_as(input_tensor)], dim=1)

    # if self.with_r:
    if with_r:
        rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
        ret = torch.cat([ret, rr], dim=1)

    return ret


class UpsamplingBottleneck(nn.Module):
    """进行上采样恢复原来的信息
    主分支:
    1. 1x1 的卷积核，用于下采样
    2. 使用之前最大池化得到的序列信息来进行反池化.

    辅助分支:
    1. 1x1 的卷积核，用于降维
    2. 反卷积 (默认大小, 3x3);
    3. 1x1 的卷积核，用于升维；
    4. 使用dropout作为正则化手段

    关键参数:
    - in_channels (int): 输入
    - out_channels (int): 输出
    - internal_ratio (int, optional): 1*1的卷积核压缩比
    - kernel_size (int, optional): 卷积核大小
    - padding (int, optional): 填充方式
    - dropout_prob (float, optional): dropout概率
    - bias (bool, optional): 偏置，true时激活
    - relu (bool, optional): true：rule；false：prelu
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super(UpsamplingBottleneck,self).__init__()

        # 检测压缩比取值是否在合理范围
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))
        # 计算压缩比
        internal_channels = in_channels // internal_ratio

        # 设置激活函数
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # 主要分支
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        # 上采样
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # 辅助分支
        # 1x1 卷积，用于降维
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation)

        # 反卷积
        self.ext_conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 卷积、用于升维
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels), activation)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # 联结之后利用Prelu激活
        self.out_prelu = activation

    def forward(self, x, max_indices):
        # 主要分支
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices)
        
        
        # 扩展分支
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # 将两个分支的结果进行联结
        out = main + ext

        # 返回激活函数后得到的结果
        return self.out_prelu(out)







    