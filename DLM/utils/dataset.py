import torch
from torch.utils.data import Dataset
import os.path as osp
import os
from PIL import Image
from utils.data_transforms import DataTransforms
import torchvision.transforms as transforms
from utils.misc_utils import to_2tuple
import numpy as np
import cv2


class RGBD_Dataset(Dataset):
    def __init__(self,
                 root_dir,
                 depth_dir="Depth",
                 image_dir="Image",
                 cdcp_dir="CDCP_GT",
                 img_suffix=".jpg",
                 ann_suffix=".png",
                 train_size=256,
                ):
        super(RGBD_Dataset, self).__init__()
        self.root_dir = root_dir
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        self.depth_dir = osp.join(self.root_dir, depth_dir)
        self.rgb_dir = osp.join(self.root_dir, image_dir)
        self.cdcp_gt_dir = osp.join(self.root_dir, cdcp_dir)
        self.pre_gt_dir = osp.join(self.root_dir, "Pre_GT")
        if not os.path.exists(self.pre_gt_dir):
            os.makedirs(self.pre_gt_dir, exist_ok=True)
        self.depths_list = [depth_name for depth_name in os.listdir(self.depth_dir)]
        self.imgs_list = [depth_name.split(".")[0]+self.img_suffix for depth_name in self.depths_list]
        self.anns_list = [depth_name.split(".")[0]+self.ann_suffix for depth_name in self.depths_list]
        self.data_transform = DataTransforms(train_size)

    def __len__(self):
        return len(self.depths_list)

    def __getitem__(self, index):
        rgb = Image.open(osp.join(self.rgb_dir, self.imgs_list[index])).convert("RGB")
        depth = Image.open(osp.join(self.depth_dir, self.depths_list[index])).convert("L") 
        cdcp_ann = Image.open(osp.join(self.cdcp_gt_dir, self.anns_list[index])).convert("RGB")
        if os.path.exists(osp.join(self.pre_gt_dir, self.anns_list[index])):
            pre_ann = Image.open(osp.join(self.pre_gt_dir, self.anns_list[index])).convert("RGB")
            rgb, depth, colored_depth, cdcp_ann, pre_ann = self.data_transform(rgb, depth, cdcp_ann, pre_ann)
            return rgb, depth, colored_depth, cdcp_ann, pre_ann
        else:
            rgb, depth, colored_depth, cdcp_ann = self.data_transform(rgb, depth, cdcp_ann, None)
            return rgb, depth, colored_depth, cdcp_ann


class Test_Dataset(Dataset):
    def __init__(self,
                 root_dir,
                 img_dir,
                 ann_dir,
                 depth_dir=None,
                 img_suffix=".jpg",
                 ann_suffix=".png",
                 depth_suffix=".png",
                 testsize=256,
                 ):
        self.testsize = testsize
        self.root_dir = root_dir
        self.img_dir = osp.join(self.root_dir, img_dir)
        self.ann_dir = osp.join(self.root_dir, ann_dir)
        self.depth_dir = osp.join(self.root_dir, depth_dir)

        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        self.depth_suffix = depth_suffix

        self.depths_list = [depth_name for depth_name in os.listdir(self.depth_dir)]
        self.imgs_list = [depth_name.split(".")[0]+self.img_suffix for depth_name in self.depths_list]
        self.anns_list = [depth_name.split(".")[0]+self.ann_suffix for depth_name in self.depths_list]

        self.rgb_transform = transforms.Compose([
            transforms.Resize(to_2tuple(testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depth_transform = transforms.Compose([
            transforms.Resize(to_2tuple(testsize)),
            transforms.ToTensor()])

        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
        
        self.color_list = [cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE]

        self.index = 0

    def load_data(self):
        
        image = Image.open(osp.join(self.img_dir, self.imgs_list[self.index])).convert("RGB")
        image = self.rgb_transform(image).unsqueeze(0)

        ori_depth = Image.open(osp.join(self.depth_dir, self.depths_list[self.index])).convert("L")
        ori_depth = self.depth_transform(ori_depth).unsqueeze(0)

        gt = Image.open(osp.join(self.ann_dir, self.anns_list[self.index])).convert("L")
        gt = self.gt_transform(gt).unsqueeze(0)
        
        depth = cv2.imread(osp.join(self.depth_dir, self.depths_list[self.index]), 0)
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
        colored_depth = torch.from_numpy(depths)[None, :]
        
        name = self.anns_list[self.index]

        self.index += 1

        return image, ori_depth, colored_depth, gt, name

    def __len__(self):
        return len(self.imgs_list)