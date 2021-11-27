import torch
import torch.nn as nn
import torch.nn.functional as F


from model.shared_encoder import ENetEncoder
from model.decoder import ENetDecoder
from model.loss import DiscriminativeLoss

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LaneNet(nn.Module):
    def __init__(self):
        super(LaneNet,self).__init__()
        self.sigmoid = nn.Sigmoid()
        self._encoder = ENetEncoder(input,encoder_relu=False)
        self._encoder.to(DEVICE)

        self._sgmentation = ENetDecoder(encoder_relu=False, decoder_relu=False, segm=True)
        self._sgmentation.to(DEVICE)

        self._embedding = ENetDecoder(encoder_relu=False, decoder_relu=False, segm=False)
        self._embedding.to(DEVICE)
        

    def forward(self,input):
        # 共享的编码层
        encoder_ret,max_indices1,max_indices2 = self._encoder(input)
        # segmentation层分支
        segm_ret = self._sgmentation(encoder_ret,max_indices1,max_indices2)
        segm_ret = self.sigmoid(segm_ret)
        binary_seg_ret = torch.where(segm_ret>0.5,1,0)
        embed_ret = self._embedding(encoder_ret,max_indices1,max_indices2)
        return{
            'instance_seg_logits': embed_ret,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': segm_ret
        }
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
    # 计算损失函数
def compute_loss(net_output, binary_label, instance_label,n_objects,max_n_objects,class_weights):
    '''
        net_output:网络的输出
        binary_label:二维语义分割的标签
        instance_label：embedding分类的标签
        n_objects：tensor(n,)类型，每一个图片中含有车道线的个数
        max_n_objects：int 车道线最多的数量
    '''
    # 计算二维语义分割的损失，二位交叉熵
    ce_loss_fn = nn.BCELoss(class_weights)
    # ce_loss_fn = nn.BCELoss()
    binary_seg_logits = net_output["binary_seg_logits"].squeeze(dim=1)
    binary_loss = ce_loss_fn(binary_seg_logits, binary_label.float())
    # 计算embedding分支的损失函数，DiscriminativeLoss
    pix_embedding = net_output["instance_seg_logits"]
    ds_loss_fn = DiscriminativeLoss(delta_var=0.5,delta_dist=3,norm=2)
    embedding_loss = ds_loss_fn(pix_embedding, instance_label,n_objects,max_n_objects)
    total_loss = binary_loss + embedding_loss
    out = net_output["binary_seg_pred"]
    
    iou = 0
    p = 0
    r = 0
    acc = 0
    batch_size = out.size()[0]
    for i in range(batch_size):
        FN = out[i].numel()-((-binary_label[i]+out[i].squeeze(0))+1).nonzero().size()[0]
        PR = out[i].squeeze(0).nonzero().size()[0]
        GT = binary_label[i].nonzero().size()[0]
        TP = (out[i].squeeze(0) * binary_label[i]).nonzero().size()[0]
        union = PR + GT - TP
        iou += TP / union
        r += TP/(TP+FN)
        acc += 1-(binary_label[i]-out[i].squeeze(0)).nonzero().size()[0]/out[i].numel()
        if PR == 0:
            p += 0
        else:
            p += TP/PR
    bin_iou = iou / batch_size
    bin_p = p / batch_size
    bin_r = r / batch_size
    bin_acc = acc/batch_size
    return total_loss, binary_loss, embedding_loss, bin_iou,bin_p,bin_r,bin_acc