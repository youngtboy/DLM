import torch
import torch.nn.functional as F
from utils.misc_utils import save_result
import os
import cv2
from tqdm import tqdm
from prettytable import PrettyTable


def validate(save_dir, val_loader, sod_model, gpl_model):
    sod_model.eval()
    gpl_model.eval()
    data_name = val_loader.img_dir.split("Image")[0].split("/")[-2]
    with torch.no_grad():
        for _ in tqdm(range(len(val_loader)), desc=data_name+" is validating"):
            image, depth, colored_depth, label, name = val_loader.load_data()
            size = label.size()
            image = image.cuda()

            pred = sod_model(image, None, None, None)
            pred = F.interpolate(
                input=pred, size=size[-2:],
                mode='bilinear', align_corners=False
            )
            save_result(os.path.join(save_dir, data_name), pred, name)
        val_loader.index = 0

        _result = getMetrics("./data/Test_Data/{}/GT".format(data_name),  # ground truth
                             os.path.join(save_dir, data_name))  # prediction
        _table = PrettyTable(["Smeasure", "maxFm", "meanFm",  "adpFm", "wFmeasure", "maxEm", "meanEm", "adpEm", "MAE"])
        _table.title = data_name
        _table.add_row([round(_result["Smeasure"], 3),
                        round(_result["maxFm"], 3),
                        round(_result["meanFm"], 3),
                        round(_result["adpFm"], 3),
                        round(_result["wFmeasure"], 3),
                        round(_result["maxEm"], 3),
                        round(_result["meanEm"], 3),
                        round(_result["adpEm"], 3),
                        round(_result["MAE"], 3)])
        print(_table)


def getMetrics(mask_root, pred_root):
    from utils.metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    MAE = MAE()

    mask_root = os.path.join(mask_root)
    pred_root = os.path.join(pred_root)
    pred_name_list = sorted(os.listdir(pred_root))

    for mask_name in tqdm(pred_name_list, total=len(pred_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        MAE.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]

    results = {
        "Smeasure": sm,
        "wFmeasure": wfm,
        "MAE": mae,
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
        "maxFm": fm["curve"].max(),
    }
    return results