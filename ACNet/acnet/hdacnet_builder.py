from builder import ConvBuilder
import torch.nn as nn

class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


class HDACBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(HDACBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*3, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*3,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn1 = nn.BatchNorm2d(num_features=out_channels)
            self.square_bn2 = nn.BatchNorm2d(num_features=out_channels)
            self.square_bn3 = nn.BatchNorm2d(num_features=out_channels)
            self.out_channels = out_channels

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs1 = self.square_bn1(square_outputs[:,0*self.out_channels:1*self.out_channels,:,:])
            square_outputs2 = self.square_bn2(square_outputs[:,1*self.out_channels:2*self.out_channels,:,:])
            square_outputs3 = self.square_bn3(square_outputs[:,2*self.out_channels:3*self.out_channels,:,:])
            
            return square_outputs1 + square_outputs2 + square_outputs3



class HDACNetBuilder(ConvBuilder):

    def __init__(self, base_config, deploy):
        super(HDACNetBuilder, self).__init__(base_config=base_config)
        self.deploy = deploy

    def switch_to_deploy(self):
        self.deploy = True


    def Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', use_original_conv=False):
        if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
            return super(HDACNetBuilder, self).Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, use_original_conv=True)
        else:
            return HDACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy)


    def Conv2dBN(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
            return super(HDACNetBuilder, self).Conv2dBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=True)
        else:
            return HDACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy)


    def Conv2dBNReLU(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
            return super(HDACNetBuilder, self).Conv2dBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=True)
        else:
            se = nn.Sequential()
            se.add_module('acb', HDACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy))
            se.add_module('relu', self.ReLU())
            return se


    def BNReLUConv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
            return super(HDACNetBuilder, self).BNReLUConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=True)
        bn_layer = self.BatchNorm2d(num_features=in_channels)
        conv_layer = HDACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy)
        se = self.Sequential()
        se.add_module('bn', bn_layer)
        se.add_module('relu', self.ReLU())
        se.add_module('acb', conv_layer)
        return se
