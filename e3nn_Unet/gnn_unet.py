import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from nnunet.network_architecture.neural_network import SegmentationNetwork
import gconv.gnn as gnn

class GMaxSpatialPool3d(nn.MaxPool3d):
    """
    Performs spatial max pooling on 3d spatial inputs.
    """

    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        x1 = super().forward(x.flatten(1, 2))
        return x1.view(*x.shape[:3],*x1.shape[-3:]), H

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

#class GNN_UNet(nn.Module):
class GNN_UNet(SegmentationNetwork):
    '''Hardcoded GNN-UNet with 3 downsamples,  variable input features'''
    def __init__(self, n_channels, n_classes,group_size = 4, top_level_features = 30):
        super().__init__() 
        self.conv_op = nn.Conv3d #Needed in order to use nnUnet predict_3D
        self.n_channels = n_channels
        self.num_classes = n_classes
        self.tlf = top_level_features

        self.nonlinearity = gnn.GLeakyReLU()
        self.pool = GMaxSpatialPool3d((2,2,2))
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) 

        self.lifting_layer = gnn.GLiftingConvSE3(in_channels=n_channels, out_channels=self.tlf, kernel_size=5,padding=2,group_kernel_size=group_size)  
        self.gconv_layer1a = gnn.GSeparableConvSE3(in_channels=self.tlf, out_channels=self.tlf, kernel_size=5,group_kernel_size=group_size,padding=1) 
        self.norm1a = gnn.normalization.GInstanceNorm3d(self.tlf)
        self.gconv_layer1b = gnn.GSeparableConvSE3(in_channels=self.tlf, out_channels=self.tlf, kernel_size=5,group_kernel_size=group_size,padding=1) 
        self.norm1b = gnn.normalization.GInstanceNorm3d(self.tlf)

        self.gconv_layer2a = gnn.GSeparableConvSE3(in_channels=self.tlf, out_channels=2*self.tlf, kernel_size=5,group_kernel_size=group_size,padding=1) 
        self.norm2a = gnn.normalization.GInstanceNorm3d(2*self.tlf)
        self.gconv_layer2b = gnn.GSeparableConvSE3(in_channels=2*self.tlf, out_channels=2*self.tlf, kernel_size=5,group_kernel_size=group_size,padding=1) 
        self.norm2b = gnn.normalization.GInstanceNorm3d(2*self.tlf)

        self.gconv_layer3a = gnn.GSeparableConvSE3(in_channels=2*self.tlf, out_channels=4*self.tlf, kernel_size=5,group_kernel_size=group_size,padding=1) 
        self.norm3a = gnn.normalization.GInstanceNorm3d(4*self.tlf)
        self.gconv_layer3b = gnn.GSeparableConvSE3(in_channels=4*self.tlf, out_channels=4*self.tlf, kernel_size=5,group_kernel_size=group_size,padding=1) 
        self.norm3b = gnn.normalization.GInstanceNorm3d(4*self.tlf)

        self.gconv_layer4a = gnn.GSeparableConvSE3(in_channels=4*self.tlf, out_channels=8*self.tlf, kernel_size=5,group_kernel_size=group_size,padding=1) 
        self.norm4a = gnn.normalization.GInstanceNorm3d(8*self.tlf)
        self.gconv_layer4b = gnn.GSeparableConvSE3(in_channels=8*self.tlf, out_channels=8*self.tlf, kernel_size=5,group_kernel_size=group_size,padding=1) 
        self.norm4b = gnn.normalization.GInstanceNorm3d(8*self.tlf)

        self.gconv_layer5a = gnn.GSeparableConvSE3(in_channels=12*self.tlf, out_channels=4*self.tlf, kernel_size=5,group_kernel_size=group_size,padding=1) 
        self.norm5a = gnn.normalization.GInstanceNorm3d(4*self.tlf)
        self.gconv_layer5b = gnn.GSeparableConvSE3(in_channels=4*self.tlf, out_channels=4*self.tlf, kernel_size=5,group_kernel_size=group_size,padding=1) 
        self.norm5b = gnn.normalization.GInstanceNorm3d(4*self.tlf)

        self.gconv_layer6a = gnn.GSeparableConvSE3(in_channels=6*self.tlf, out_channels=2*self.tlf, kernel_size=5,group_kernel_size=group_size,padding=1) 
        self.norm6a = gnn.normalization.GInstanceNorm3d(2*self.tlf)
        self.gconv_layer6b = gnn.GSeparableConvSE3(in_channels=2*self.tlf, out_channels=2*self.tlf, kernel_size=5,group_kernel_size=group_size,padding=1) 
        self.norm6b = gnn.normalization.GInstanceNorm3d(2*self.tlf)

        self.gconv_layer7a = gnn.GSeparableConvSE3(in_channels=3*self.tlf, out_channels=self.tlf, kernel_size=5,group_kernel_size=group_size,padding=1) 
        self.norm7a = gnn.normalization.GInstanceNorm3d(self.tlf)
        self.gconv_layer7b = gnn.GSeparableConvSE3(in_channels=self.tlf, out_channels=self.tlf, kernel_size=5,group_kernel_size=group_size,padding=1) 
        self.norm7b = gnn.normalization.GInstanceNorm3d(self.tlf)
        self.gconv_layer7c = gnn.GSeparableConvSE3(in_channels=self.tlf, out_channels=self.num_classes, kernel_size=5,group_kernel_size=group_size,padding=1) 
        self.gconv_layer_out = gnn.GMaxGroupPool()

        
    def forward(self, x):
        x, H = self.lifting_layer(x)                                                          
        x, H = self.gconv_layer1a(x, H)                                                        
        x, H = self.norm1a(x,H)
        x, H = self.nonlinearity(x,H)
        x, H = self.gconv_layer1b(x, H)                                                        
        x, H = self.norm1b(x,H)
        x1, H = self.nonlinearity(x,H)
        x, H = self.pool(x1,H)

        x, H = self.gconv_layer2a(x, H)                                                        
        x, H = self.norm2a(x,H)
        x, H = self.nonlinearity(x,H)
        x, H = self.gconv_layer2b(x, H)                                                        
        x, H = self.norm2b(x,H)
        x2, H = self.nonlinearity(x,H)
        x, H = self.pool(x2,H)

        x, H = self.gconv_layer3a(x, H)                                                        
        x, H = self.norm3a(x,H)
        x, H = self.nonlinearity(x,H)
        x, H = self.gconv_layer3b(x, H)                                                        
        x, H = self.norm3b(x,H)
        x3, H = self.nonlinearity(x,H)
        x, H = self.pool(x3,H)

        x, H = self.gconv_layer4a(x, H)                                                        
        x, H = self.norm4a(x,H)
        x, H = self.nonlinearity(x,H)
        x, H = self.gconv_layer4b(x, H)                                                        
        x, H = self.norm4b(x,H)
        x, H = self.nonlinearity(x,H)
        x_shape = x.shape
        x = self.upsample(x.flatten(1,2))
        x =  x.view(*x_shape[:3],*x.shape[-3:])

        x = torch.cat([x,x3], dim=1)
        x, H = self.gconv_layer5a(x, H)                                                        
        x, H = self.norm5a(x,H)
        x, H = self.nonlinearity(x,H)
        x, H = self.gconv_layer5b(x, H)                                                        
        x, H = self.norm5b(x,H)
        x, H = self.nonlinearity(x,H)
        x_shape = x.shape
        x = self.upsample(x.flatten(1,2))
        x =  x.view(*x_shape[:3],*x.shape[-3:])

        x = torch.cat([x,x2], dim=1)
        x, H = self.gconv_layer6a(x, H)                                                        
        x, H = self.norm6a(x,H)
        x, H = self.nonlinearity(x,H)
        x, H = self.gconv_layer6b(x, H)                                                        
        x, H = self.norm6b(x,H)
        x, H = self.nonlinearity(x,H)
        x_shape = x.shape
        x = self.upsample(x.flatten(1,2))
        x =  x.view(*x_shape[:3],*x.shape[-3:])

        x = torch.cat([x,x1], dim=1)
        x, H = self.gconv_layer7a(x, H)                                                        
        x, H = self.norm7a(x,H)
        x, H = self.nonlinearity(x,H)
        x, H = self.gconv_layer7b(x, H)                                                        
        x, H = self.norm7b(x,H)
        x, H = self.nonlinearity(x,H)
        x, H = self.gconv_layer7c(x, H)                                                        
        x, H = self.gconv_layer_out(x,H)

        return x
