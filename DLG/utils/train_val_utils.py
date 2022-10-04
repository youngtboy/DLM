import torch
import torch.nn.functional as F
import os
import cv2
from tqdm import tqdm
import einops


def validate(dir, val_loader, model):
    model.eval()
    data_name = val_loader.depth_dir.split("Depth")[0].split("/")[-2]
    with torch.no_grad():
        for _ in tqdm(range(len(val_loader)), desc=data_name+" is validating"):
            depths, name, shapes = val_loader.load_data()
            depths = depths.float().cuda()
            depths = einops.rearrange(depths, 'b t c h w -> (b t) c h w')
            recon_depth, recons, masks, _ = model(depths)
            masks = einops.rearrange(masks, '(b t) s c h w -> b t s c h w', t=2)  # [1, 2, 2, 1, h, w]
                
            os.makedirs(os.path.join(dir, data_name, "s_0_0"), exist_ok=True)
            s_0_0 = F.interpolate(masks[0][0], size=shapes, mode="bilinear", align_corners=True)
            cv2.imwrite(os.path.join(dir, data_name, "s_0_0", name+".png"), (s_0_0[0][0].cpu().numpy()*255.))
            os.makedirs(os.path.join(dir, data_name, "s_1_0"), exist_ok=True)
            s_1_0 = F.interpolate(masks[0][1], size=shapes, mode="bilinear", align_corners=True)
            cv2.imwrite(os.path.join(dir, data_name, "s_1_0", name+".png"), (s_1_0[0][0].cpu().numpy()*255.))
            os.makedirs(os.path.join(dir, data_name, "s_mean"), exist_ok=True)
            cv2.imwrite(os.path.join(dir, data_name, "s_mean", name+".png"),
                        (((s_1_0[0][0]+s_0_0[0][0])/2).cpu().numpy()*255.))

        val_loader.index = 0


def getMetrics(mask_root, pred_root):
    from utils.metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    MAE = MAE()

    mask_root = os.path.join(mask_root)
    pred_root = os.path.join(pred_root)
    name_list = sorted(os.listdir(mask_root))

    for mask_name in tqdm(name_list, total=len(name_list)):
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