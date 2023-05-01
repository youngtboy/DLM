import torch
import os
import torch.optim as optim
from utils.dataset import RGBD_Dataset, Test_Dataset
from utils.misc_utils import to_2tuple, crf_refine
from src.dlm import DLM
from src.dlg import DLG
from argparse import ArgumentParser
import torch.nn.functional as F
from utils.lr_config import adjust_lr_poly
from utils.train_val_utils import validate
from tqdm import tqdm
from loss.criterion import _dlm_loss, dlm_loss, _dlg_loss
import cv2
import einops


def main(args):
    assert torch.cuda.is_available(), 'Error: \'gpu\' or \'cuda\' is not available'
    torch.cuda.set_device(int(args.device.split(":")[1]))
    device = torch.device(args.device)

    train_dataset = RGBD_Dataset(root_dir=args.data_root,
                                 depth_dir="Depth",
                                 image_dir="Image",
                                 cdcp_dir="CDCP_GT",
                                 train_size=args.train_size)
    print("Loading {} images for training".format(len(train_dataset)))


    val_dataset_NJUD = Test_Dataset(root_dir="./data",
                                    img_dir="Test_Data/NJUD/Image",
                                    ann_dir="Test_Data/NJUD/GT",
                                    depth_dir="Test_Data/NJUD/Depth",
                                    img_suffix=".jpg",
                                    ann_suffix=".png",
                                    testsize=args.test_size)
    val_dataset_NLPR = Test_Dataset(root_dir="./data",
                                    img_dir="Test_Data/NLPR/Image",
                                    ann_dir="Test_Data/NLPR/GT",
                                    depth_dir="Test_Data/NLPR/Depth",
                                    img_suffix=".jpg",
                                    ann_suffix=".png",
                                    testsize=args.test_size)
    val_dataset_SSD = Test_Dataset(root_dir="./data",
                                   img_dir="Test_Data/SSD/Image",
                                   ann_dir="Test_Data/SSD/GT",
                                   depth_dir="Test_Data/SSD/Depth",
                                   img_suffix=".jpg",
                                   ann_suffix=".png",
                                   testsize=args.test_size)
    val_dataset_SIP = Test_Dataset(root_dir="./data",
                                   img_dir="Test_Data/SIP/Image",
                                   ann_dir="Test_Data/SIP/GT",
                                   depth_dir="Test_Data/SIP/Depth",
                                   img_suffix=".jpg",
                                   ann_suffix=".png",
                                   testsize=args.test_size)
    val_dataset_STERE = Test_Dataset(root_dir="./data",
                                     img_dir="Test_Data/STERE/Image",
                                     ann_dir="Test_Data/STERE/GT",
                                     depth_dir="Test_Data/STERE/Depth",
                                     img_suffix=".jpg",
                                     ann_suffix=".png",
                                     testsize=args.test_size)
    val_dataset_DES = Test_Dataset(root_dir="./data",
                                   img_dir="Test_Data/DES/Image",
                                   ann_dir="Test_Data/DES/GT",
                                   depth_dir="Test_Data/DES/Depth",
                                   img_suffix=".jpg",
                                   ann_suffix=".png",
                                   testsize=args.test_size)
    val_dataset_DUTD = Test_Dataset(root_dir="./data",
                                    img_dir="Test_Data/DUTD/Image",
                                    ann_dir="Test_Data/DUTD/GT",
                                    depth_dir="Test_Data/DUTD/Depth",
                                    img_suffix=".jpg",
                                    ann_suffix=".png",
                                    testsize=args.test_size)
    val_dataset_LFSD = Test_Dataset(root_dir="./data",
                                    img_dir="Test_Data/LFSD/Image",
                                    ann_dir="Test_Data/LFSD/GT",
                                    depth_dir="Test_Data/LFSD/Depth",
                                    img_suffix=".jpg",
                                    ann_suffix=".png",
                                    testsize=args.test_size)
    val_trainset = Test_Dataset(root_dir="./data",
                                img_dir="Test_Data/NJUD_NLPR_DUT/Image",
                                ann_dir="Test_Data/NJUD_NLPR_DUT/GT",
                                depth_dir="Test_Data/NJUD_NLPR_DUT/Depth",
                                img_suffix=".jpg",
                                ann_suffix=".png",
                                testsize=args.test_size)

    val_dataset = [val_dataset_NLPR, val_dataset_NJUD, val_dataset_SSD, val_dataset_SIP, val_dataset_STERE,
                   val_dataset_DES, val_dataset_DUTD, val_dataset_LFSD]


    # Train dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=True,
                                               batch_size=args.batch_size,
                                               pin_memory=True,
                                               drop_last=True,
                                               num_workers=args.num_workers)

    # Model
    dlg_model = DLG(to_2tuple(args.train_size), 2)
    dlm_model = DLM()
    dlg_model = dlg_model.to(device)
    dlm_model = dlm_model.to(device)

    # Load weights
    if args.pretrained_weight is not None:
        assert os.path.exists(args.pretrained_weight), "Error: The checkpoint does not exist"
        print("Using weights, checkpoint is {}".format(args.pretrained_weight))
        loaded_dict = torch.load(args.pretrained_weight, map_location=device)
        dlg_model.load_state_dict(torch.load(args.pretrained_weight_gpl, map_location=device)["state_dict"])
        dlm_model.rgb_encoder.load_state_dict(loaded_dict, strict=False)
    else:
        print.info("not using pretrained weights.")

    # Optimizer
    dlg_optimizer = optim.Adam(dlg_model.parameters(), lr=args.lr)
    dlm_optimizer = optim.Adam(dlm_model.parameters(), lr=args.lr)

    iter_per_epoch = len(train_dataset)//args.batch_size
    total_iters = args.epochs * iter_per_epoch

    ###################################################### training ##############################################################

    for epoch in range(0, args.update_freq):
        dlg_model.train()
        dlm_model.train()
        mean_epoch_loss = torch.zeros(1).to(device)
        train_loader = tqdm(train_loader)
        recon_criterion = torch.nn.MSELoss()
        rank_criterion = torch.nn.MarginRankingLoss(0.)
        for step, data in enumerate(train_loader, start=0):
            global_step = step + epoch * iter_per_epoch
            images, depths, colored_depths, cdcp_labels = data
            images, depths, colored_depths, cdcp_labels = images.cuda(), depths.cuda(), colored_depths.cuda(), cdcp_labels.cuda()
            colored_depths = colored_depths.float().cuda()
            colored_depths = einops.rearrange(colored_depths, 'b t c h w -> (b t) c h w')
    
            adjust_lr_poly(dlm_optimizer, args.lr, global_step, total_iters)
            adjust_lr_poly(dlg_optimizer, args.lr, global_step, total_iters)
            dlm_optimizer.zero_grad()
            dlg_optimizer.zero_grad()
    
            recon_depth, _, masks, _ = dlg_model(colored_depths)
    
            _depths = depths[:, None].repeat(1, 2, 1, 1, 1)
            _depths = einops.rearrange(_depths, 'b t c h w -> (b t) c h w')[:, None].repeat(1, 2, 1, 1, 1)
            r = (_depths * masks).mean([-1, -2])
            r1 = r[:, 0].squeeze()
            r2 = r[:, 1].squeeze()
            y = torch.full((args.batch_size*2,), -1).cuda()
            rank_loss = 5 * rank_criterion(r1, r2, y)
    
            recon_loss = 100 * recon_criterion(colored_depths, recon_depth)
            entropy_loss = 5 * -(masks * torch.log(masks + 1e-5)).sum(dim=1).mean()
            masks = einops.rearrange(masks, '(b t) s c h w -> b t s c h w', t=2)
    
            dlg_labels = masks[:, :, 0, 0]
            dlg_labels = dlg_labels.mean(dim=1)[:, None].repeat(1, 3, 1, 1)
            masks1 = masks[:, 0]
            masks2 = masks[:, 1]
            masks2 = einops.rearrange(masks2, 'b s c h w -> b c s h w')
            temporal_diff = torch.pow((masks1 - masks2), 2).mean([-1, -2])
            consistency_loss = 5 * temporal_diff.view(-1, 2*2).min(1)[0].mean()
    
            dlg_loss = recon_loss + entropy_loss + consistency_loss + rank_loss
    
            init_pre_label = (dlg_labels+cdcp_labels)/2
            _depth_labels = dlg_labels.clone()
            _cdcp_labels = cdcp_labels.clone()
            _init_pre_label = init_pre_label.clone()
    
            mask_pred, dlg_um, cdcp_um, pre_um = dlm_model(images, _depth_labels, _cdcp_labels, _init_pre_label)
            dlg_labels[dlg_labels >= 0.5] = 1
            dlg_labels[dlg_labels < 0.5] = 0
            cdcp_labels[cdcp_labels >= 0.5] = 1
            cdcp_labels[cdcp_labels < 0.5] = 0
            init_pre_label[init_pre_label >= 0.5] = 1
            init_pre_label[init_pre_label < 0.5] = 0
            mask_pred= F.interpolate(mask_pred, size=images.shape[2:], mode="bilinear", align_corners=False)
            dlg_um = F.interpolate(dlg_um, size=images.shape[2:], mode="bilinear", align_corners=False)
            cdcp_um = F.interpolate(cdcp_um, size=images.shape[2:], mode="bilinear", align_corners=False)
            pre_um = F.interpolate(pre_um, size=images.shape[2:], mode="bilinear", align_corners=False)
    
            loss = _dlm_loss(mask_pred, dlg_labels, cdcp_labels, init_pre_label, dlg_um, cdcp_um, pre_um) + dlg_loss
            loss.backward()
            dlg_optimizer.step()
            dlm_optimizer.step()
            mean_epoch_loss = (mean_epoch_loss * step + loss.detach()) / (step + 1)
            train_loader.desc = "[epoch {}] loss: {}".format(epoch, round(mean_epoch_loss.item(), 4))
    
    ###############################################################################################################################
    
    infos = {
        "state_dict_sod": dlg_model.state_dict(),
        "state_dict_gpl": dlm_model.state_dict()
    }
    
    # save weights
    weights_dir = os.path.join(args.save_weight, args.project_name)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    torch.save(infos, os.path.join(weights_dir, "Epoch_{}".format(args.update_freq) + ".pth"))
    
    dlm_model.eval()
    with torch.no_grad():
        for _ in tqdm(range(len(val_trainset)), desc="pre_label is generating"):
            image, _, _, label, name = val_trainset.load_data()
            size = label.size()
            image = image.cuda()
            pred = dlm_model(image, None, None, None)
            pred = F.interpolate(
                input=pred, size=size[-2:],
                mode='bilinear', align_corners=False
            )
            pred = pred.sigmoid().cpu().numpy().squeeze()
            pred = 255 * (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            crf_pred = crf_refine(cv2.imread(os.path.join(args.data_root, "Image/{}".format(name.split(".")[0] + ".jpg")), 1),
                        pred.astype('uint8'))
            cv2.imwrite(os.path.join(args.data_root, "NJUD_NLPR_DUT/Pre_GT/{}".format(name)), crf_pred)
        val_trainset.index = 0

    ###############################################################################################################################

    for epoch in range(args.update_freq, args.epochs):
        dlm_model.train()
        dlg_model.train()
        mean_epoch_loss = torch.zeros(1).to(device)
        train_loader = tqdm(train_loader)
        rank_criterion = torch.nn.MarginRankingLoss(0.)
        recon_criterion = torch.nn.MSELoss()
        for step, data in enumerate(train_loader, start=0):
            global_step = step + epoch * iter_per_epoch
            images, depths, colored_depths, cdcp_labels, pre_labels = data
            images, depths, cdcp_labels, pre_labels = images.cuda(), depths.cuda(), cdcp_labels.cuda(), pre_labels.cuda()
            colored_depths = colored_depths.float().cuda()
            colored_depths = einops.rearrange(colored_depths, 'b t c h w -> (b t) c h w')

            adjust_lr_poly(dlm_optimizer, args.lr, global_step, total_iters)
            adjust_lr_poly(dlg_optimizer, args.lr, global_step, total_iters)
            dlm_optimizer.zero_grad()
            dlg_optimizer.zero_grad()

            recon_depth, _, masks, _ = dlg_model(colored_depths)

            _depths = depths[:, None].repeat(1, 2, 1, 1, 1)
            _depths = einops.rearrange(_depths, 'b t c h w -> (b t) c h w')[:, None].repeat(1, 2, 1, 1, 1)
            score = (_depths * masks).mean([-1, -2])
            score1 = score[:, 0].squeeze()
            score2 = score[:, 1].squeeze()
            y = torch.full((args.batch_size*2,), -1).cuda()

            rank_loss = 5 * rank_criterion(score1, score2, y)
            recon_loss = 100 * recon_criterion(colored_depths, recon_depth)
            entropy_loss = 5 * -(masks * torch.log(masks + 1e-5)).sum(dim=1).mean()
            masks = einops.rearrange(masks, '(b t) s c h w -> b t s c h w', t=2)
            depth_labels = masks[:, :, 0, 0]  # [b, t, h, w]

            depth_labels = depth_labels.mean(dim=1)[:, None].repeat(1, 3, 1, 1)

            masks1 = masks[:, 0]
            masks2 = masks[:, 1]

            masks2 = einops.rearrange(masks2, 'b s c h w -> b c s h w')
            temporal_diff = torch.pow((masks1 - masks2), 2).mean([-1, -2])
            consistency_loss = 5 * temporal_diff.view(-1, 2*2).min(1)[0].mean()

            dlg_loss = recon_loss + entropy_loss + consistency_loss + rank_loss

            _depth_labels = depth_labels.clone()
            _cdcp_labels = cdcp_labels.clone()
            _pre_labels = pre_labels.clone()
            mask_pred, dlg_um, cdcp_um, pre_um = dlm_model(images, _depth_labels, _cdcp_labels, _pre_labels)
            depth_labels[depth_labels >= 0.5] = 1
            depth_labels[depth_labels < 0.5] = 0
            cdcp_labels[cdcp_labels >= 0.5] = 1
            cdcp_labels[cdcp_labels < 0.5] = 0
            pre_labels[pre_labels >= 0.5] = 1
            pre_labels[pre_labels < 0.5] = 0
            mask_pred = F.interpolate(mask_pred, size=images.shape[2:], mode="bilinear", align_corners=False)
            dlg_um = F.interpolate(dlg_um, size=images.shape[2:], mode="bilinear", align_corners=False)
            cdcp_um = F.interpolate(cdcp_um, size=images.shape[2:], mode="bilinear", align_corners=False)
            pre_um = F.interpolate(pre_um, size=images.shape[2:], mode="bilinear", align_corners=False)

            dlg_loss = dlg_loss + _dlg_loss(depth_labels[:, 0][:, None], cdcp_labels, pre_labels, cdcp_um, pre_um, epoch, args.update_freq, args.epochs)
            loss = dlm_loss(mask_pred, depth_labels, cdcp_labels, pre_labels, dlg_um, cdcp_um, pre_um, epoch, args.update_freq, args.epochs) + dlg_loss
            loss.backward()
            dlg_optimizer.step()
            dlm_optimizer.step()
            mean_epoch_loss = (mean_epoch_loss * step + loss.detach()) / (step + 1)
            train_loader.desc = "[epoch {}] loss: {}".format(epoch, round(mean_epoch_loss.item(), 4))

        ###############################################################################################################################

        if (epoch+1) % args.update_freq == 0:
            dlm_model.eval()
            data_name = val_trainset.img_dir.split("Image")[0].split("/")[-2]
            with torch.no_grad():
                for _ in tqdm(range(len(val_trainset)), desc=data_name + " is validating"):
                    image, _, _, label, name = val_trainset.load_data()
                    size = label.size()
                    image = image.cuda()
                    pred = dlm_model(image, None, None, None)
                    pred = F.interpolate(
                        input=pred, size=size[-2:],
                        mode='bilinear', align_corners=False
                    )
                    pred = pred.sigmoid().cpu().numpy().squeeze()
                    pred = 255 * (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                    crf_pred = crf_refine(cv2.imread(os.path.join(args.data_root, "Image/{}".format(name.split(".")[0] + ".jpg")), 1),
                               pred.astype('uint8'))
                    pre_gt = cv2.imread(os.path.join(args.data_root, "Pre_GT/{}".format(name)), 0)
                    updated_pre_gt = pre_gt * args.alpha + crf_pred * (1 - args.alpha)
                    updated_pre_gt = 255 * (updated_pre_gt - updated_pre_gt.min()) / (updated_pre_gt.max() - updated_pre_gt.min() + 1e-8)
                    cv2.imwrite(os.path.join(args.data_root, "Pre_GT/{}".format(name)), updated_pre_gt)
                val_trainset.index = 0

        ####################################################### validation #####################################################################

        if ((epoch + 1) % args.eval_freq) == 0:
            infos = {
                # "optimizer": optimizer.state_dict(),
                "state_dict_sod": dlm_model.state_dict(),
                "state_dict_gpl": dlg_model.state_dict()
            }

            # save weights
            weights_dir = os.path.join(args.save_weight, args.project_name,)
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)
            torch.save(infos, os.path.join(weights_dir, "Epoch_{}".format(epoch + 1) + ".pth"))

            for val_data in val_dataset:
                validate(os.path.join(args.result_dir, args.project_name), val_data, dlm_model, dlg_model)

        ###############################################################################################################################


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")

    # path config
    parser.add_argument('--data_root', type=str, default="./data/NJUD_NLPR_DUT")
    parser.add_argument('--project_name', type=str, default="DLM")
    parser.add_argument('--pretrained_weight', type=str, default="./checkpoint/pvt.pth")
    parser.add_argument('--pretrained_weight_gpl', type=str, default="./checkpoint/dlg.pth")
    parser.add_argument('--save_weight', type=str, default="./checkpoint")
    parser.add_argument('--result_dir', type=str, default="./out")

    # train&val config
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--train_size', type=int, default=256)
    parser.add_argument('--test_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--update_freq', type=int, default=5)

    # LR config
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--alpha', type=float, default=0.7)

    args = parser.parse_args()
    main(args)
