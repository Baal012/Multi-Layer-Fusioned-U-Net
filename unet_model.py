""" Full assembly of the parts to form the complete network """

#import torch.nn.functional as F

from unet_parts import *

import torch
import torch.nn.functional as F
from torch import nn
from network import SEresnext
from network import Resnet
from network.wider_resnet import wider_resnet38_a2
#from config import cfg
from network.mynn import initialize_weights, Norm2d
from torch.autograd import Variable

#from my_functionals import GatedSpatialConv as gsc

#import cv2
import numpy as np

class Crop(nn.Module):
    def __init__(self, axis, offset):
        super(Crop, self).__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, ref):
        """

        :param x: input layer
        :param ref: reference usually data in
        :return:
        """
        for axis in range(self.axis, x.dim()):
            ref_size = ref.size(axis)
            indices = torch.arange(self.offset, self.offset + ref_size).long()
            indices = x.data.new().resize_(indices.size()).copy_(indices).long()
            x = x.index_select(axis, Variable(indices))
        return x


class MyIdentity(nn.Module):
    def __init__(self, axis, offset):
        super(MyIdentity, self).__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, ref):
        """

        :param x: input layer
        :param ref: reference usually data in
        :return:
        """
        return x

class SideOutputCrop(nn.Module):
    """
    This is the original implementation ConvTranspose2d (fixed) and crops
    """

    def __init__(self, num_output, kernel_sz=None, stride=None, upconv_pad=0, do_crops=True):
        super(SideOutputCrop, self).__init__()
        self._do_crops = do_crops
        self.conv = nn.Conv2d(num_output, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        if kernel_sz is not None:
            self.upsample = True
            self.upsampled = nn.ConvTranspose2d(1, out_channels=1, kernel_size=kernel_sz, stride=stride,
                                                padding=upconv_pad,
                                                bias=False)
            ##doing crops
            if self._do_crops:
                self.crops = Crop(2, offset=kernel_sz // 4)
            else:
                self.crops = MyIdentity(None, None)
        else:
            self.upsample = False

    def forward(self, res, reference=None):
        side_output = self.conv(res)
        if self.upsample:
            side_output = self.upsampled(side_output)
            side_output = self.crops(side_output, reference)

        return side_output


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))
         

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear',align_corners=True)
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:],
                                      mode='bilinear',align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class UNet(nn.Module):
    #def __init__(self, n_channels, n_classes, bilinear=True):
    def __init__(self, n_channels, num_classes, bilinear=True, trunk=None, criterion=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = num_classes
        self.bilinear = bilinear
        self.criterion = criterion
        
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
        self.outc = OutConv(64, num_classes)

    
        
        
        
        

        self.interpolate = F.interpolate


        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn3 = nn.Conv2d(128, 1, 1)
        self.dsn4 = nn.Conv2d(256, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)

        self.fit_dim1 = nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=False)
        self.fit_dim2 = nn.Conv2d(320, 128, kernel_size=1, padding=0, bias=False)
        self.fit_dim3 = nn.Conv2d(704, 256, kernel_size=1, padding=0, bias=False)
        self.fit_dim4 = nn.Conv2d(1472, 512, kernel_size=1, padding=0, bias=False)

        self.res1 = Resnet.BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1)
        self.res2 = Resnet.BasicBlock(192, 192, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.res3 = Resnet.BasicBlock(448, 448, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.res4 = Resnet.BasicBlock(960, 960, stride=1, downsample=None)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        #self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        #self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        #self.gate3 = gsc.GatedSpatialConv2d(8, 8)
         
        self.aspp = _AtrousSpatialPyramidPoolingModule(4096, 256,
                                                       output_stride=8)

        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280 + 256, 256, kernel_size=1, bias=False)

        self.final_seg = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        self.sigmoid = nn.Sigmoid()
        initialize_weights(self.final_seg)


    def forward(self, x, gts=None):
        m1 = self.inc(x)
        m2 = self.down1(m1)
        m3 = self.down2(m2)
        m4 = self.down3(m3)
        m5 = self.down4(m4)
        

        x_size = x.size() 
        m1_size = m1.size()
        m2_size = m2.size()
        m3_size = m3.size()
        m4_size = m4.size()

        #print(m2.size())
        #print(m1_size)
        #s2 = F.interpolate(self.dsn3(m2), m1_size[2:],
        #                    mode='bilinear', align_corners=True)  #有dsn的
        #print(self.dsn3(m2).size())

        #s3 = F.interpolate(self.dsn4(m3), m2_size[2:],
        #                    mode='bilinear', align_corners=True)
        #s4 = F.interpolate(self.dsn5(m4), m3_size[2:],
        #                    mode='bilinear', align_corners=True)


        #没有dsn的
        s2 = F.interpolate(m2, m1_size[2:],
                            mode='bilinear', align_corners=True)  #有dsn的
        #print(self.dsn3(m2).size())

        s3 = F.interpolate(m3, m2_size[2:],
                            mode='bilinear', align_corners=True)
        s4 = F.interpolate(m4, m3_size[2:],
                            mode='bilinear', align_corners=True)
        m1f = F.interpolate(m1, x_size[2:], mode='bilinear', align_corners=True)

        im_arr = x.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8)
        #canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        #for i in range(x_size[0]):
        #    canny[i] = cv2.Canny(im_arr[i],10,100)
        #canny = torch.from_numpy(canny).cuda().float()

        cs = self.res1(m1f)
        #print(m1_size)
        cs1 = F.interpolate(cs, m1_size[2:],
                           mode='bilinear', align_corners=True)
        #print(cs1.size())
        cs1_m = torch.cat([m1, cs1], dim=1)
        #print(cs1_m.size())
        cs1_m = self.fit_dim1(cs1_m)   #可以和m1拼接然后和m拼接了
        #print(cs1_m.size())
        


        #cs = self.d1(cs)
        #cs = self.gate1(cs, s2)
        #print(s2.size())
        #print(cs1.size())
        cs2 = torch.cat([s2,cs1],dim=1)
        cs2 = self.res2(cs2)
        cs2 = F.interpolate(cs2, m2_size[2:],
                           mode='bilinear', align_corners=True)
        cs2_m = torch.cat([m2, cs2], dim=1)
        cs2_m = self.fit_dim2(cs2_m)   #可以和m2拼接然后和m拼接了
        #print(cs2_m.size())
        #print(m2.size())
        #cs = self.d2(cs)
        #cs = self.gate2(cs, s3)
        cs3 = torch.cat([s3,cs2],dim=1)
        cs3 = self.res3(cs3)
        cs3 = F.interpolate(cs3, m3_size[2:],
                           mode='bilinear', align_corners=True)
        cs3_m = torch.cat([m3, cs3], dim=1)
        cs3_m = self.fit_dim3(cs3_m)   #可以和m3拼接然后和m拼接了
        #print("\n")
        #print(cs3_m.size())
        #print(m3.size())


        #cs = self.d3(cs)
        #cs = self.gate3(cs, s4)
        cs4 = torch.cat([s4,cs3],dim=1)
        cs4 = self.res4(cs4)
        cs4 = F.interpolate(cs4, m4_size[2:],
                           mode='bilinear', align_corners=True)
        cs4_m = torch.cat([m4, cs4], dim=1)
        cs4_m = self.fit_dim4(cs4_m)  #可以和m4拼接然后和m拼接了
        
        #print(cs4_m.size())
        #print(m4.size())




        m = self.up1(m5, cs4_m)
        m = self.up2(m, cs3_m)
        m = self.up3(m, cs2_m)
        m = self.up4(m, cs1_m)
        logits = self.outc(m)
        '''
        edge_out = self.sigmoid(cs)
        cat = torch.cat((edge_out, canny), dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)
        '''
        '''
        # aspp
        x = self.aspp(m7, acts)
        dec0_up = self.bot_aspp(x)

        dec0_fine = self.bot_fine(m2)
        dec0_up = self.interpolate(dec0_up, m2.size()[2:], mode='bilinear',align_corners=True)
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)

        dec1 = self.final_seg(dec0)  
        seg_out = self.interpolate(dec1, x_size[2:], mode='bilinear')            
       '''
        #if self.training:
        #    return self.criterion((seg_out, edge_out), gts)     #那个里面是bce2d         
        #else:
        return logits







        #return logits


def main():
    img=np.zeros((1,3,256,256))
    net = UNet(n_channels=3, num_classes=1, bilinear=True)
    img = torch.tensor(img, dtype=torch.float32)
    img=net(img)

#main()
