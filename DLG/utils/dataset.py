import torch
from torch.utils.data import Dataset
import os.path as osp
import os
from utils.data_transforms import DataTransforms
import cv2
import numpy as np
from utils.misc_utils import to_2tuple


class RGBD_Dataset(Dataset):
    def __init__(self,
                 root_dir,
                 depth_dir,
                 train_size=256,  
                ):

        super(RGBD_Dataset, self).__init__()

        self.root_dir = root_dir
        self.depth_dir = osp.join(self.root_dir, depth_dir)
        self.depths_list = [depth_name for depth_name in os.listdir(self.depth_dir)]
        self.data_transform = DataTransforms(train_size)

    def __len__(self):
        return len(self.depths_list)

    def __getitem__(self, index):      
        depth = cv2.imread(osp.join(self.depth_dir, self.depths_list[index]), 0)
        depth, corlored_depth = self.data_transform(depth)
        return depth, corlored_depth


class Test_Dataset(Dataset):
    def __init__(self,
                 root_dir,
                 depth_dir,
                 testsize=256,
                 ):
        self.testsize = testsize

        self.root_dir = root_dir
        self.depth_dir = osp.join(self.root_dir, depth_dir)
        self.depths_list = [depth_name for depth_name in os.listdir(self.depth_dir)]
        self.color_list = [cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE]
        self.index = 0

    def load_data(self):
        depth = cv2.imread(osp.join(self.depth_dir, self.depths_list[self.index]), 0)
        shapes = depth.shape
        depth = 255 - depth
        depths = []
        for color in self.color_list:
            _depth = cv2.applyColorMap(depth, color)
            _depth = cv2.cvtColor(_depth, cv2.COLOR_BGR2RGB)
            _depth = cv2.resize(_depth, to_2tuple(self.testsize), interpolation=cv2.INTER_LINEAR)
            _depth = np.clip(((_depth / 255.0) - 0.5) * 2, -1., 1.)
            _depth = _depth.transpose(2, 0, 1)
            depths.append(_depth)
        depths = np.stack(depths, 0)
        depths = torch.from_numpy(depths)[None, :]

        name = self.depths_list[self.index].split(".")[0]

        self.index += 1
        return depths, name, shapes

    def __len__(self):
        return len(self.depths_list)