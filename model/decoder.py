# coding: utf-8
"""
Shared encoders (E-net).
"""
import torch.nn as nn
from model.Enet_blocks import RegularBottleneck,UpsamplingBottleneck
class ENetDecoder(nn.Module):
    """
    ENET Encoder第三层+ENET Decoder
    """

    def __init__(self,encoder_relu=False,decoder_relu=False,segm = False):
        super(ENetDecoder,self).__init__()
        self.segm_num = 1
        self.embed_num = 4
        # self.max_indices1 = max_indices1
        # self.max_indices2 = max_indices2
        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)
        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(128, 64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(64, 18, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(18, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 6 - fullconv
        if segm == True:
            self.fullconv = nn.ConvTranspose2d(18,self.segm_num,kernel_size=3,stride=2,padding=1,output_padding=1,bias=False)

        else:
            self.fullconv = nn.ConvTranspose2d(18,self.embed_num,kernel_size=3,stride=2,padding=1,output_padding=1,bias=False)
    
    def forward(self, input,max_indices1,max_indices2):
        input = self.regular3_0(input)
        input = self.dilated3_1(input)
        input = self.asymmetric3_2(input)
        input = self.dilated3_3(input)
        input = self.regular3_4(input)
        input = self.dilated3_5(input)
        input = self.asymmetric3_6(input)
        input = self.dilated3_7(input)
        input = self.upsample4_0(input,max_indices2)
        input = self.regular4_1(input)
        input = self.regular4_2(input)
        input = self.upsample5_0(input,max_indices1)
        input = self.regular5_1(input)
        output = self.fullconv(input)

        return output
