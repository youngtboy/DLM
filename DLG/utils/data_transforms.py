import cv2
import torch.nn as nn
import numpy as np
from utils.misc_utils import to_2tuple
import einops


class DataTransforms(nn.Module):
    def __init__(self, train_size):
        super(DataTransforms, self).__init__()
        self.trainsize = train_size

    def depthTransform(self, depth):
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        depth = cv2.resize(depth, to_2tuple(self.trainsize), interpolation=cv2.INTER_LINEAR)
        depth = ((depth/255.0)-0.5)*2.0
        depth = np.clip(depth, -1., 1.)
        return einops.rearrange(depth, 'h w c -> c h w')

    def forward(self, depth):
        _depth = cv2.resize(depth, to_2tuple(self.trainsize))

        depth = 255 - depth
        depth_1 = cv2.applyColorMap(depth, cv2.COLORMAP_AUTUMN)
        depth_2 = cv2.applyColorMap(depth, cv2.COLORMAP_BONE)

        depth_1 = self.depthTransform(depth_1)
        depth_2 = self.depthTransform(depth_2)

        return np.stack((_depth, _depth), 0), np.stack((depth_1, depth_2), 0)