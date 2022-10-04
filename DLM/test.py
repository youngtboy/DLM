import torch
import os
import argparse
from utils.dataset import Test_Dataset
from tqdm import tqdm
from utils.misc_utils import save_result
import torch.nn.functional as F
from src.dlm import DLM
from utils.train_val_utils import getMetrics
from prettytable import PrettyTable


def main(args):
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    assert torch.cuda.is_available(), 'Error: \'gpu\' or \'cuda\' is not available'

    val_dataset_NJUD = Test_Dataset(root_dir=args.data_root,
                                    img_dir="Test_Data/NJUD/Image",
                                    ann_dir="Test_Data/NJUD/GT",
                                    depth_dir="Test_Data/NJUD/Depth",
                                    img_suffix=".jpg",
                                    ann_suffix=".png",
                                    testsize=args.test_size)
    val_dataset_NLPR = Test_Dataset(root_dir=args.data_root,
                                    img_dir="Test_Data/NLPR/Image",
                                    ann_dir="Test_Data/NLPR/GT",
                                    depth_dir="Test_Data/NLPR/Depth",
                                    img_suffix=".jpg",
                                    ann_suffix=".png",
                                    testsize=args.test_size)
    val_dataset_SSD = Test_Dataset(root_dir=args.data_root,
                                   img_dir="Test_Data/SSD/Image",
                                   ann_dir="Test_Data/SSD/GT",
                                   depth_dir="Test_Data/SSD/Depth",
                                   img_suffix=".jpg",
                                   ann_suffix=".png",
                                   testsize=args.test_size)
    val_dataset_SIP = Test_Dataset(root_dir=args.data_root,
                                   img_dir="Test_Data/SIP/Image",
                                   ann_dir="Test_Data/SIP/GT",
                                   depth_dir="Test_Data/SIP/Depth",
                                   img_suffix=".jpg",
                                   ann_suffix=".png",
                                   testsize=args.test_size)
    val_dataset_STERE = Test_Dataset(root_dir=args.data_root,
                                     img_dir="Test_Data/STERE/Image",
                                     ann_dir="Test_Data/STERE/GT",
                                     depth_dir="Test_Data/STERE/Depth",
                                     img_suffix=".jpg",
                                     ann_suffix=".png",
                                     testsize=args.test_size)
    val_dataset_DES = Test_Dataset(root_dir=args.data_root,
                                   img_dir="Test_Data/DES/Image",
                                   ann_dir="Test_Data/DES/GT",
                                   depth_dir="Test_Data/DES/Depth",
                                   img_suffix=".jpg",
                                   ann_suffix=".png",
                                   testsize=args.test_size)
    val_dataset_DUTD = Test_Dataset(root_dir=args.data_root,
                                    img_dir="Test_Data/DUTD/Image",
                                    ann_dir="Test_Data/DUTD/GT",
                                    depth_dir="Test_Data/DUTD/Depth",
                                    img_suffix=".jpg",
                                    ann_suffix=".png",
                                    testsize=args.test_size)
    val_dataset_LFSD = Test_Dataset(root_dir=args.data_root,
                                    img_dir="Test_Data/LFSD/Image",
                                    ann_dir="Test_Data/LFSD/GT",
                                    depth_dir="Test_Data/LFSD/Depth",
                                    img_suffix=".jpg",
                                    ann_suffix=".png",
                                    testsize=args.test_size)

    val_dataset = [val_dataset_NJUD, val_dataset_NLPR, val_dataset_STERE, val_dataset_DUTD,
                   val_dataset_DES, val_dataset_SIP, val_dataset_LFSD, val_dataset_SSD]

    # Model
    s_model = DLM()
    s_model = s_model.to(device)

    # Load weights
    assert os.path.exists(args.checkpoint), "Error: The checkpoint does not exist"
    print("Using model, checkpoint is {}".format(args.checkpoint))
    loaded_dict = torch.load(args.checkpoint, map_location=device)
    s_model.load_state_dict(loaded_dict["state_dict_sod"], strict=False)

    # test
    s_model.eval()
    with torch.no_grad():
        for val_loader in val_dataset:
            data_name = val_loader.img_dir.split("Image")[0].split("/")[-2]
            for _ in tqdm(range(len(val_loader)), desc=data_name + " is validating"):
                image, _, _, label, name = val_loader.load_data()
                size = label.size()
                image = image.cuda()
                pred = s_model(image, None, None, None)
                pred = F.interpolate(
                    input=pred, size=size[-2:],
                    mode='bilinear', align_corners=False
                )
                save_result(os.path.join(args.result_dir, data_name), pred, name)
            val_loader.index = 0

            _result = getMetrics(os.path.join(args.data_root, "Test_Data/{}/GT".format(data_name)),  # ground truth
                                 os.path.join(args.result_dir, data_name))  # prediction
            _table = PrettyTable(["Sm", "MaxFm", "Em", "MAE"])
            _table.title = data_name
            _table.add_row([round(_result["Smeasure"], 3),
                            round(_result["maxFm"], 3),
                            round(_result["maxEm"], 3),
                            round(_result["MAE"], 3)])
            print("\n")
            print(_table)
            print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_size', type=int, default=256)
    parser.add_argument('--data_root', type=str, default="./data/")
    parser.add_argument('--checkpoint', type=str, help='initial weights path', default="./checkpoint/best.pth")
    parser.add_argument('--result_dir', type=str, default="./out/test")

    args = parser.parse_args()
    main(args)