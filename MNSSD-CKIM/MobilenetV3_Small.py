# for torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import torchvision

# for general lib
import math
from math import sqrt
from itertools import product as product
import logging
from collections import OrderedDict

# for my lib
from coco_utils import *

# backend
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# 一个常规卷积模块     
def conv_bn(inp, oup, stride, groups=1, activation=nn.ReLU6):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        activation(inplace=True)
    )

# 1x1卷积 depth-wise (groups)
def conv_1x1_bn(inp, oup, groups=1, activation=nn.ReLU6):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        activation(inplace=True)
    )

# 激活函数    
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + float(3.0), inplace=True) / float(6.0)
        return out
class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + float(3.0), inplace=True) / float(6.0)
        return out

# SE模块 in_size是channel数量,给feature加权
class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #每个通道特征合并为一个向量  1xin_size
        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),#缩减通道数
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),#放大通道数
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )
    def forward(self, x):
        return x * self.se(x)# 这里加权，因为se(x)的每个通道只有1x1,是一个数

# 一个完整的block        
class Block(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        self.output_status = False
        if kernel_size == 5 and in_size == 48 and expand_size == 288: # 这其实已经锁定了那一层了 conv4_3_feats size: 300 -> 150 -> 75 -> 38
            self.output_status = True
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )
    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x))) # conv -> bn -> activate input_size -> expand_size
        if self.output_status:
            expand = out # 扩张？
        out = self.nolinear2(self.bn2(self.conv2(out))) # conv -> bn -> activate expand_size -> expand_size
        out = self.bn3(self.conv3(out)) # conv -> bn -> activate expand_size -> output_size
        if self.se != None: # 如果有注意力模块就加上
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out #是否需要shotcut
	
        if self.output_status:
            return (expand, out)# 这里会把expand也返回？为什么这一层的expand要返回呢？
        return out

# 这里要拿出哪两层？
# 看了这么多(tflite版本也是)，基本上是拿出来 19x19 和 10x10 这两层，原作里说的是拿 38x38 和 19x19,不知道为啥不一样了
# 后边的extra_conv里要拿出来 5x5 3x3 1x1
class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Small, self).__init__()
        # 第一个卷积
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)# 150x150x16
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        # 连接许多模块
        self.bneck = nn.Sequential(
            #     k  in  exp out        act             se    s                 输出
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),#1   75x75x16
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),#2           38x38x24
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),#4           38x38x24
            Block(5, 24, 96, 40, nn.ReLU(inplace=True), SeModule(40), 2),#4   19x19x40
            Block(5, 40, 240, 40, nn.ReLU(inplace=True), SeModule(40), 1),#5  19x19x40
            Block(5, 40, 240, 40, nn.ReLU(inplace=True), SeModule(40), 1),#6  19x19x40
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),#7               19x19x48   

            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),#8               19x19x48
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),#9               10x10x96 这个exp是最后一个19x19放进去
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),#10              10x10x96
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),#11              10x10x93
        )
        # 后接卷积？？这里为啥还要接 原版moblienet里确实接了一个pw-conv 把通道从96扩展到了576
        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False) 
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()	#conv7_feats size: 300 -> 150 -> 75 -> 38 -> 19 -> 10 # 10x10x576  这个放进去
        self.init_weights() # 这个之前忘记了

    # 他这个加载参数其实没实现
    def init_weights(self, pretrained=None):#"./mbv3_large.old.pth.tar"
        if isinstance(pretrained, str):
            logger = logging.getLogger()

            checkpoint = torch.load(pretrained,map_location='cpu') ["state_dict"]      
            self.load_state_dict(checkpoint,strict=False)
            for param in self.parameters():
                param.requires_grad = True # to be or not to be

            #  also load module  
            # if isinstance(checkpoint, OrderedDict):
            #     state_dict = checkpoint
            # elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            #     state_dict = checkpoint['state_dict']
            # else:
            #     print("No state_dict found in checkpoint file")

            # if list(state_dict.keys())[0].startswith('module.'):
            #     state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
            # # load state_dict
            # if hasattr(self, 'module'):
            #     self.module.load_state_dict( state_dict,strict=False)
            # else:
            #     self.load_state_dict(state_dict,strict=False)    




            print("\nLoaded base model.\n")
        # 这个用不上了，因为他后来保存了整个模型，不需要这个了
        elif pretrained is None:
            print("\nNo loaded base model.\n")
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
    # 前向传播路径  
    def forward(self, x):
        # 第一个卷积
        out = self.hs1(self.bn1(self.conv1(x)))
        # blocks
        for i, block in enumerate(self.bneck):
            out = block(out)
            if isinstance(out, tuple):# 特定层才是touple(expand, out) conv4_3_feats size: 300 -> 150 -> 75 -> 38
                conv4_3_feats =out[0]                
                out = out[1]
        # 最后一个卷积
        out = self.hs2(self.bn2(self.conv2(out)))# conv7_feats 19x19
        
        conv7_feats=out		
        # 返回两个特定层的特征
        return conv4_3_feats,conv7_feats

# 辅助卷积，就是ssd在mobilenet后边加的那几层 input_size 19x19
class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """
    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()
        self.extra_convs = []

        # 这里conv是采用 pw -> dw -> pw 的流程
        # pw1 是缩减一下通道
        # dw  是缩减size
        # pw2 是混合通道，并减少通道

        # 576 -> 256 -> 512    10x10 -> 5x5
        self.extra_convs.append(conv_1x1_bn(576, 256))# pw扩张通道
        self.extra_convs.append(conv_bn(256, 256, 2, groups=256))#dw 降低size
        self.extra_convs.append(conv_1x1_bn(256, 512, groups=1)) # pw 混合通道  5x5x512

        # 512 -> 128 -> 256    5x5 -> 3x3
        self.extra_convs.append(conv_1x1_bn(512, 128))
        self.extra_convs.append(conv_bn(128, 128, 2, groups=128))
        self.extra_convs.append(conv_1x1_bn(128, 256)) # 3x3x256
    
        # 256 -> 128 -> 256    3x3 -> 2x2
        self.extra_convs.append(conv_1x1_bn(256, 128))
        self.extra_convs.append(conv_bn(128, 128, 2, groups=128))
        self.extra_convs.append(conv_1x1_bn(128, 256)) # 2x2x256
            
        # 256 -> 64 -> 128     2x2 -> 1x1
        self.extra_convs.append(conv_1x1_bn(256, 64))
        self.extra_convs.append(conv_bn(64, 64, 2, groups=64))
        self.extra_convs.append(conv_1x1_bn(64, 128)) # 1x1x128
        self.extra_convs = nn.Sequential(*self.extra_convs) 
        
        self.init_conv2d()
        
    def init_conv2d(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
    def forward(self, conv7_feats):
        """
        Forward propagation.
        :param conv7_feats: lower-level conv7 feature map
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
 
        outs = []
        out=conv7_feats
        for i, conv in enumerate(self.extra_convs):
            
            out = conv(out)
            if i % 3 == 2:
                outs.append(out)
                
        conv8_2_feats=outs[0]
        conv9_2_feats=outs[1]
        conv10_2_feats=outs[2]
        conv11_2_feats=outs[3]
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats

class PredictionConvolutions(nn.Module):
    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()
        self.n_classes = n_classes
        
        n_boxes = {'conv4_3': 4,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 6,
                   'conv11_2': 6}
        
        
        
        input_channels=[288, 576, 512, 256, 256, 128]
        
        # 第一组Conv2D回归box
        #                               19x19x288                19x19x16(4个box)
        self.loc_conv4_3 = nn.Conv2d(input_channels[0], n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        #                               10x10x576                10x10x24(6个box)
        self.loc_conv7 = nn.Conv2d(input_channels[1], n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        #                               5x5x512                  5x5x24(6个box)
        self.loc_conv8_2 = nn.Conv2d(input_channels[2], n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        #                               3x3x256                  3x3x24(6个box)
        self.loc_conv9_2 = nn.Conv2d(input_channels[3], n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        #                               2x2x256                  2x2x24(6个box)
        self.loc_conv10_2 = nn.Conv2d(input_channels[4], n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        #                               1x1x128                  1x1x24(6个box)
        self.loc_conv11_2 = nn.Conv2d(input_channels[5], n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)

        # 第二组Conv2D回归class
        #                               19x19x288                19x19x84(4x21)
        self.cl_conv4_3 = nn.Conv2d(input_channels[0], n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        #                               10x10x576                10x10x126(6x21)
        self.cl_conv7 = nn.Conv2d(input_channels[1], n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        #                               5x5x512                  5x5x126(6x21)
        self.cl_conv8_2 = nn.Conv2d(input_channels[2], n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        #                               3x3x256                  3x3x126(6个box)
        self.cl_conv9_2 = nn.Conv2d(input_channels[3], n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        #                               2x2x256                  2x2x126(6个box)
        self.cl_conv10_2 = nn.Conv2d(input_channels[4], n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        #                               1x1x128                  1x1x126(6个box)
        self.cl_conv11_2 = nn.Conv2d(input_channels[5], n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)
        
        self.init_conv2d()
        
    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        batch_size = conv4_3_feats.size(0)
        # 传播，调换维度，展开成一维
        # conv4_3_feats batchx288x19x19 -> l_conv4_3 batchx16x19x19
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  
        l_conv4_3 = l_conv4_3.permute(0, 2, 3,1).contiguous() #batchx19x19x16 这里是把bchw -> bhwc 因为torch读数据是按照维度顺序读的，这样改以后可以保证
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)#19x19行，4列？

        l_conv7 = self.loc_conv7(conv7_feats)  
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  
        l_conv7 = l_conv7.view(batch_size, -1, 4)  

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4) 

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4) 

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  

        # 这里算one hot向量表示类别
        # conv4_3_feats batchx288x19x19 -> cl_conv4_3 batchx84x19x19
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()  # batchx19x19x84
        c_conv4_3 = c_conv4_3.view(batch_size, -1,self.n_classes)  # batchx(19x19x4)x21

        c_conv7 = self.cl_conv7(conv7_feats)  
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  
        c_conv7 = c_conv7.view(batch_size, -1,self.n_classes)  # batchx(10x10x6)x21

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # batchx(5x5x6)x21

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # batchx(3x3x6)x21

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes) # batchx(2x2x6)x21

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # batchx(1x1x6)x21
        
        
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1) # 按照dim1拼接(列) 这个东西是一个4列的
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2],dim=1)# 这个应该是 2278x21
        return locs, classes_scores

class SSD300(nn.Module):
    """
    The SSD300 network - encapsulates the base MobileNet network, auxiliary, and prediction convolutions.
    """
    def __init__(self, n_classes):
        super(SSD300, self).__init__()
        self.n_classes = n_classes
        self.base = MobileNetV3_Small(num_classes=self.n_classes)
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)
      
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 288, 1, 1))  
        nn.init.constant_(self.rescale_factors, 20)
              
        # 这个是prior_box
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
      
        conv4_3_feats, conv7_feats = self.base(image)  
          
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()+1e-10  
        conv4_3_feats = conv4_3_feats / norm  
        conv4_3_feats = conv4_3_feats * self.rescale_factors  
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)  
        
        feats_list = []
        feats_list.append(conv4_3_feats)
        feats_list.append(conv7_feats)
        feats_list.append(conv8_2_feats)
        feats_list.append(conv9_2_feats)
        feats_list.append(conv10_2_feats)
        feats_list.append(conv11_2_feats)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,conv11_2_feats)  
        return locs, classes_scores, feats_list

    def create_prior_boxes(self):
  
        fmap_dims = {'conv4_3': 19,
                     'conv7': 10,
                     'conv8_2': 5,
                     'conv9_2': 3,
                     'conv10_2': 2,
                     'conv11_2': 1}             
        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 3., 0.5, .333],
                         'conv11_2': [1., 2., 3., 0.5, .333]}

        fmaps = list(fmap_dims.keys())
        prior_boxes = []
        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]
                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])
                        
                        
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])
        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  
        prior_boxes.clamp_(0, 1)  
        return prior_boxes
    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  
        
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
        for i in range(batch_size):
            
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy.to(device)))  
            
            image_boxes = list()
            image_labels = list()
            image_scores = list()
            max_scores, best_label = predicted_scores[i].max(dim=1)  
            
            for c in range(1, self.n_classes):
                
                class_scores = predicted_scores[i][:, c]  
                score_above_min_score = class_scores > min_score  
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  
                class_decoded_locs = decoded_locs[score_above_min_score]  
                
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  
                class_decoded_locs = class_decoded_locs[sort_ind]  
                
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  
                
                
                
                
                suppress = torch.zeros((n_above_min_score), dtype=torch.bool).to(device)  
                
                for box in range(class_decoded_locs.size(0)):
                    
                    if suppress[box] == 1:
                        continue
                    
                    
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    
                    
                    suppress[box] = 0
                               
                image_boxes.append(class_decoded_locs[~suppress])
                image_labels.append(torch.LongTensor((~ suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[~suppress])
            
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))
            
            image_boxes = torch.cat(image_boxes, dim=0)  
            image_labels = torch.cat(image_labels, dim=0)  
            image_scores = torch.cat(image_scores, dim=0)  
            n_objects = image_scores.size(0)
            
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  
                image_boxes = image_boxes[sort_ind][:top_k]  
                image_labels = image_labels[sort_ind][:top_k]  
            
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)
        return all_images_boxes, all_images_labels, all_images_scores  

class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """ 
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy # 这个就是model的prior_box
        self.priors_xy = cxcy_to_xy(priors_cxcy)# 把prior_box变成xy形式
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
    #                                                          标注label  # batch_size x tensor([60, 57, 29, 66]) 
    def forward(self, predicted_locs, predicted_scores, boxes, labels):


        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  
        # print("predicted_locs size: ", predicted_locs.shape)
        # print("true_locs size: ", true_locs.shape)
        # print("predicted_scores size: ", predicted_scores.shape)
        # print("true_classes size: ", true_classes.shape)
        # print("MultiBoxLoss forwarding   >>>  n_classes: %d" % n_classes)#81
        # print("     boxes: ")
        # print(boxes)
        # print("     labels: ")
        # print(labels)
        # 根据传进来bbox算出来对应的那个最符合的prior_box,并且记录这个prior的label
        for i in range(batch_size):
            # print("------------------------------- box[%d] and label -----------------------------"%i)
            # print(boxes[i])
            # print(labels[i])
            n_objects = boxes[i].size(0)# 这张图片里有几个物体
            overlap = find_jaccard_overlap(boxes[i],self.priors_xy) # 返回IoU 矩阵 (n1,n2) n1是标注文件
            # overlap里应该有0,这时找max怎么算? ------> 经过测试发现overlap为0的都算成0号box了

            # print("         obj num: %d"%n_objects)
            # IoU值                  对应的label_box索引(n1中的第几个，也就是第几个物体,这里都在n1范围内，所以后边那一步不会越界)
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0) # 这里的dim是dim to reduce要删除的维度，把n1删除了就剩n2了
            # overlap_for_each_prior
            # tensor([0.0000, 0.0000, 0.0000,  ..., 0.3058, 0.1777, 0.1704], device='cuda:0')
            # object_for_each_prior  对应的是和第几个label_box的IoU
            # tensor([0, 0, 0,  ..., 1, 1, 1], device='cuda:0')
            # print("         IoU for each prior:")
            # print(overlap_for_each_prior)
            # print("         obj for each prior:")
            # print(object_for_each_prior)
            
            # 这里和上边相反，得到的是每个label_box对应的prior_box id
            _, prior_for_each_object = overlap.max(dim=1) #长度是obj num,他默认一个图片里的Obj不能比prior多
            # prior_for_each_object
            # tensor([1726, 1702,  629,  721], device='cuda:0')
            # print("         Prior for each obj")
            # print(prior_for_each_object)
            # 把prior_box里和label_box最匹配的那几个label_box id拿出来
            # print("         obj for each prior(selected ones)")
            # print(object_for_each_prior[prior_for_each_object])
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)# 这个是一个tensor(1,2,3,4)
            # print(object_for_each_prior[prior_for_each_object])
            # object_for_each_prior[prior_for_each_object] 这个是从label_box找到对应的Priorbox,在看这个prior_box对应哪个Label_box 
            # print("         Obj for each prior(should changed)")
            # print(object_for_each_prior.shape)
            # print(object_for_each_prior[prior_for_each_object])

            # 这里把IoU矩阵里找到了对应label_box的prior_box设置成了1
            overlap_for_each_prior[prior_for_each_object] = 1.

            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@object_for_each_prior:")
            # print(object_for_each_prior.shape)
            # print(object_for_each_prior)
            label_for_each_prior = labels[i][object_for_each_prior]   # 新生成这个tensor长2278，每个位置是该prior对应的label(不是在labels里的索引，而是值)

            # 如果IoU太小，那么就把label设置成背景
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0 
            
            # 每个batch都有一个这个
            true_classes[i] = label_for_each_prior
            # 这里有问题吧，如果某个Prior没有对应的Box怎么办
            # 1. 把minx,miny转换成cx,cy (没问题)
            # 2. 计算出box和priorbox的偏差 (他有一个特定的方法)
            # ps: 这个函数有点问题，因为torch.log的输入如果是0，那么就会返回无穷大，这里
            # print("going to convert this")# 这里转出来的h应该是0了
            # print(boxes[i][object_for_each_prior])
            # print("going to encode true_locs")
            # print(boxes[i].shape)
            # print((boxes[i][object_for_each_prior]).shape)#
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)# 这里不太对啊 
        
        positive_priors = true_classes != 0  # [ True,  True,  True, False, False]
        # print("positive_priors shape:")
        # print(positive_priors.shape)
        # print(positive_priors)
        # print("predicted_locs shape:")
        # print(predicted_locs.shape)
        # print(predicted_locs[positive_priors])        
        # print("true_locs shape:")
        # print(true_locs.shape)
        # print(true_locs[positive_priors])

        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])
        
  
        n_positives = positive_priors.sum(dim=1)  
        n_hard_negatives = self.neg_pos_ratio * n_positives  
        
        
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  
        
        conf_loss_pos = conf_loss_all[positive_priors]  
        
        
        conf_loss_neg = conf_loss_all.clone()  
        conf_loss_neg[positive_priors] = 0.  
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  
        
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  
        # print("conf_loss : ", conf_loss.item())
        # print("loc_loss : ", loc_loss.item())# 这个会inf

        return conf_loss+self.alpha * loc_loss, conf_loss, loc_loss
