from torch.nn.modules.loss import _Loss
import torch.nn as nn
import torch

# Hnet模块，用于估计矩阵系数H

class Hnet(nn.Module):
    """ Hnet网络，用于计算矫正系数H，共有6个自由度
    H=[a,b,c]
      [0,d,e]
      [0,f,1]
    网络架构：
    1.2个conv+bn+relu 16*3*3 + maxpool
    2.2个conv+bn+relu 32*3*3 + maxpool
    3.2个conv+bn+relu 64*3*3 + maxpool 
    4.Linear+BN+Relu 1*1(压平，输出为1024)
    5.linear（最终输出为6个数）
    关键参数：
    - 输入维度（整型）
    - 输出维度（整型）
    - 卷积核尺寸（整型，可选项）：默认为3
    - 边缘填充（整型，可选项）：默认填充0
    - 偏置（布尔型，可选项）：默认不使用
    - relu（布尔型，可选项）；默认为true，当为true时，使用relu激活
    否则使用Prelu激活
    """ 
    def __init__(self,
                in_channels,
                outchannels,
                kernel_size=3,
                padding=0,
                bias=False,
                relu=True
                ):
        super(Hnet,self).__init__()
        # 根据输入选择合适的激活函数
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        # 设置合适的maxpool
        self.main_max1 = nn.MaxPool2d(
            kernel_size,
            stride=2,
            padding=padding)
        # Hnet第一部分架构 16*3*3
        self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    16,
                    kernel_size,
                    stride=1,
                    padding=(padding, 0),
                    bias=bias), nn.BatchNorm2d(16), activation,
                nn.Conv2d(
                    in_channels,
                    16,
                    kernel_size,
                    stride=1,
                    padding=(padding, 0),
                    bias=bias), nn.BatchNorm2d(16), activation,
                nn.MaxPool2d(kernel_size,stride=2,padding=padding))
        # Hnet第二部分架构 32*3*3    
        self.conv2 = nn.Sequential(
                nn.Conv2d(
                    16,
                    32,
                    kernel_size,
                    stride=1,
                    padding=padding,
                    bias=bias), nn.BatchNorm2d(32), activation,
                nn.Conv2d(
                    16,
                    32,
                    kernel_size,
                    stride=1,
                    padding=padding,
                    bias=bias), nn.BatchNorm2d(32), activation,
                nn.MaxPool2d(kernel_size,stride=2,padding=padding))  
        # Hnet第三部分架构 64*3*3
        self.conv3 = nn.Sequential(
                nn.Conv2d(
                    32,
                    64,
                    kernel_size,
                    stride=1,
                    padding=padding,
                    bias=bias), nn.BatchNorm2d(64), activation,
                nn.Conv2d(
                    32,
                    64,
                    kernel_size,
                    stride=1,
                    padding=padding,
                    bias=bias), nn.BatchNorm2d(64), activation,
                nn.MaxPool2d(kernel_size,stride=2,padding=padding))   
        # Hnet第四部分架构 用一维卷积核压平数据
        self.conv4 = nn.Sequential(
                 nn.Conv2d(
                      64,
                      1024,
                      kernel_size=1,
                      padding=padding,
                      bias=bias), nn.BatchNorm2d(1024), activation) 
        
        # Hnet第五部分架构，线性层
        self.linear = nn.Linear(in_features=1024,out_features=6,bias=bias)

    # 前向计算函数
    def forward(self,x):
        main = self.conv1(x)
        main = self.conv2(main)
        main = self.conv3(main)
        main = self.conv4(main)
        main = self.linear(main)

        return main
    
class HNetLoss(_Loss):
    """
    HNet 损失函数设计
    """

    def __init__(self, gt_pts, transformation_coefficient, name, usegpu=True):
        """

        :param gt_pts: [x, y, 1]
        :param transformation_coeffcient: [[a, b, c], [0, d, e], [0, f, 1]]
        :param name:
        :return: 
        """
        super(HNetLoss, self).__init__()
        # 输出每个点的坐标，1代表该点是车道线坐标
        self.gt_pts = gt_pts
        
        # 转置矩阵H的超参数
        self.transformation_coefficient = transformation_coefficient
        self.name = name
        # 是否使用GPU
        self.usegpu = usegpu

    def _hnet(self):
        """
        计算返回得到的由矩阵H变形得到的预测结果x,y
        :return:
        """
        # 生成矩阵H
        self.transformation_coefficient = torch.cat((self.transformation_coefficient, torch.tensor([1.0])),dim=0)
        H_indices = torch.tensor([0, 1, 2, 4, 5, 7, 8])
        H_shape = 9
        H = torch.zeros(H_shape)
        H.scatter_(dim=0, index=H_indices, src=self.transformation_coefficient)
        H = H.view((3, 3))
        
        # 矩阵乘法 H * (gt_pts^T) 即 P'= H*P
        pts_projects = torch.matmul(H, self.gt_pts.t())
        # 纠正后的曲线的X，Y值，拟合的曲线x =k*y
        Y = pts_projects[1, :]
        X = pts_projects[0, :]
        # 独热编码
        Y_One = torch.ones(Y.size())
        # 最小二乘的矩阵实现，Y最高次数为3
        Y_stack = torch.stack((torch.pow(Y, 3), torch.pow(Y, 2), Y, Y_One), dim=1).squeeze()
        # 得到权重系数w，X=w3*y^3+w2*y^2+w1*y+w0
        w = torch.matmul(torch.matmul(torch.inverse(torch.matmul(Y_stack.t(), Y_stack)),
                                      Y_stack.t()),
                         X.view(-1, 1))
        # 通过拟合的曲线计算x坐标的预测值
        x_preds = torch.matmul(Y_stack, w)
        # 将预测结果合并（x^*,y,1）
        preds = torch.stack((x_preds.squeeze(), Y, Y_One), dim=1).t()
        # 返回预测的结果
        return (H, preds)
        
    def _hnet_loss(self):
        """
        计算损失，并返回计算结果
        :return:
        """
        H, preds = self._hnet()
        # 计算真实X的预测值
        x_transformation_back = torch.matmul(torch.inverse(H), preds)
        # 均方误差
        loss = torch.mean(torch.pow(self.gt_pts.t()[0, :] - x_transformation_back[0, :], 2))
        # 返回均方差
        return loss



    def _hnet_transformation(self):
        """
        计算预测的真实值
        """
        H, preds = self._hnet()
        x_transformation_back = torch.matmul(torch.inverse(H), preds)

        return x_transformation_back

    def forward(self, input, target, n_clusters):
        # 计算并返回
        return self._hnet_loss(input, target)
