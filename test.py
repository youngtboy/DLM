import torch
import os
from utils.dataset import Test_Dataset
from utils.misc_utils import to_2tuple
from src.dlg import DLG
from argparse import ArgumentParser
import einops
from tqdm import tqdm
import torch.nn.functional as F
import cv2


def main(args):
    assert torch.cuda.is_available(), 'Error: \'gpu\' or \'cuda\' is not available'
    torch.cuda.set_device(int(args.device.split(":")[1]))
    device = torch.device(args.device)

    val_train_dataset = Test_Dataset(root_dir=args.data_root,
                                     depth_dir="Depth",
                                     testsize=args.test_size)
    print("Loading {} images for testing".format(len(val_train_dataset)))
    
    val_loader = val_train_dataset
    
    # Model
    test_size = to_2tuple(args.test_size)
    model = DLG(test_size, 2)
    model = model.to(device)

    # Load weights
    assert os.path.exists(args.pretrained_weight), "Error: The file does not exist"
    print("Using weights, checkpoint is {}".format(args.pretrained_weight))
    loaded_dict = torch.load(args.pretrained_weight, map_location=device)
    model.load_state_dict(loaded_dict["state_dict"], strict=True)

    model.eval()
    data_name = val_loader.depth_dir.split("Depth")[0].split("/")[-2]
    with torch.no_grad():
        for _ in tqdm(range(len(val_loader)), desc=data_name+" is validating"):
            depths, name, shape = val_loader.load_data()
            depths = depths.float().cuda()
            depths = einops.rearrange(depths, 'b t c h w -> (b t) c h w')
            recon_depth, recons, masks, _ = model(depths)
            masks = einops.rearrange(masks, '(b t) s c h w -> b t s c h w', t=2)
                
            os.makedirs(os.path.join(args.out_dir, data_name, "s_0_0"), exist_ok=True)
            s_0_0 = F.interpolate(masks[0][0], size=shape, mode="bilinear", align_corners=True)
            cv2.imwrite(os.path.join(args.out_dir, data_name, "s_0_0", name+".png"), (s_0_0[0][0].cpu().numpy()*255.))
            os.makedirs(os.path.join(args.out_dir, data_name, "s_1_0"), exist_ok=True)
            s_1_0 = F.interpolate(masks[0][1], size=shape, mode="bilinear", align_corners=True)
            cv2.imwrite(os.path.join(args.out_dir, data_name, "s_1_0", name+".png"), (s_1_0[0][0].cpu().numpy()*255.))
            os.makedirs(os.path.join(args.out_dir, data_name, "s_mean"), exist_ok=True)
            cv2.imwrite(os.path.join(args.out_dir, data_name, "s_mean", name+".png"),
                        (((s_1_0[0][0]+s_0_0[0][0])/2).cpu().numpy()*255.))
        val_loader.index = 0
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")

    # path config
    parser.add_argument('--data_root', type=str, default="./data/NJUD_NLPR_DUT")
    parser.add_argument('--pretrained_weight', type=str, default="./checkpoint/DLG.pth")
    parser.add_argument('--out_dir', type=str, default="./out/test")

    parser.add_argument('--test_size', type=int, default=256)
    args = parser.parse_args()
    main(args)