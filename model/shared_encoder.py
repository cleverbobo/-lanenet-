# coding: utf-8
"""
Shared encoders (E-net).
"""
import torch.nn as nn
from model.Enet_blocks import RegularBottleneck, DownsamplingBottleneck, InitialBlock, AddCoords
class ENetEncoder(nn.Module):
    """
    ENET Encoder
    """
    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super(ENetEncoder,self).__init__()
        self.initial_block = InitialBlock(5, 18, padding=1, relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(18, 64, padding=1, return_indices=True, dropout_prob=0.01,
                                                    relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(64, 128, padding=1, return_indices=True, dropout_prob=0.1,
                                                    relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

    def forward(self, input):
        input = AddCoords(input) 
        input = self.initial_block(input)
        input,max_indices1 = self.downsample1_0(input)
        input = self.regular1_1(input)
        input = self.regular1_2(input)
        input = self.regular1_3(input)
        input = self.regular1_4(input)
        input,max_indices2 = self.downsample2_0(input)
        input = self.regular2_1(input)
        input = self.dilated2_2(input)
        input = self.asymmetric2_3(input)
        input = self.dilated2_4(input)
        input = self.regular2_5(input)
        input = self.dilated2_6(input)
        input = self.asymmetric2_7(input)
        shared_encoder_output = self.dilated2_8(input)

        return shared_encoder_output,max_indices1,max_indices2

        
        