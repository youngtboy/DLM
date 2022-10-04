import random

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transform
import torchvision.transforms.functional
from torchvision.transforms.functional import hflip
import numpy as np
from PIL import Image
from utils.misc_utils import to_2tuple
import einops

class DataTransforms(nn.Module):
    def __init__(self, train_size):
        super(DataTransforms, self).__init__()
        self.trainsize = train_size
        self.rgb_transform = transform.Compose([
            transform.Resize(to_2tuple(self.trainsize)),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.depth_transform = transform.Compose([
            transform.Resize(to_2tuple(self.trainsize)),
            transform.ToTensor(),
        ])

        self.ann_transform = transform.Compose([
            transform.Resize(to_2tuple(self.trainsize)),
            transform.ToTensor(),
        ])

    def coloreddepthTransform(self, depth):
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        depth = cv2.resize(depth, to_2tuple(self.trainsize), interpolation=cv2.INTER_LINEAR)
        depth = ((depth/255.0)-0.5)*2.0
        depth = np.clip(depth, -1., 1.)
        return einops.rearrange(depth, 'h w c -> c h w')
    
    def forward(self, rgb, depth, cdcp_ann, pre_ann):
        rgb = self.rgb_transform(rgb)
        cdcp_ann = self.ann_transform(cdcp_ann)

        _depth = self.depth_transform(depth)
        
        depth = np.array(depth)
        depth = 255 - depth
        depth_1 = cv2.applyColorMap(depth, cv2.COLORMAP_AUTUMN)
        depth_2 = cv2.applyColorMap(depth, cv2.COLORMAP_BONE)
        depth_1 = self.coloreddepthTransform(depth_1)
        depth_2 = self.coloreddepthTransform(depth_2)
        if pre_ann is None:
            return rgb, _depth, np.stack((depth_1, depth_2), 0), cdcp_ann
        pre_ann = self.ann_transform(pre_ann)
        return rgb, _depth, np.stack((depth_1, depth_2), 0), cdcp_ann, pre_ann
