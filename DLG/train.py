import torch
import os
import torch.optim as optim
from utils.dataset import RGBD_Dataset, Test_Dataset
from torch.utils.tensorboard import SummaryWriter
from utils.misc_utils import to_2tuple
from src.dlg import DLG
from argparse import ArgumentParser
import torch.nn as nn
from utils.lr_config import set_learning_rate
from utils.train_val_utils import validate
import einops


def main(args):
    assert torch.cuda.is_available(), 'Error: \'gpu\' or \'cuda\' is not available'
    torch.cuda.set_device(int(args.device.split(":")[1]))
    device = torch.device(args.device)

    train_dataset = RGBD_Dataset(root_dir=args.data_root,
                                 depth_dir="Depth",
                                 train_size=args.train_size)

    print("Loading {} images for training".format(len(train_dataset)))
    
    val_train_dataset = Test_Dataset(root_dir=args.data_root,
                                     depth_dir="Depth",
                                     testsize=args.test_size)

    val_dataset = [val_train_dataset]

    # Train dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=True,
                                               batch_size=args.batch_size,
                                               pin_memory=True,
                                               drop_last=True,
                                               num_workers=args.num_workers)
    # Model
    train_size = to_2tuple(args.train_size)
    model = DLG(train_size, 2)
    model = model.to(device)

    # Load weights
    if args.pretrained_weight is not None:
        assert os.path.exists(args.pretrained_weight), "Error: The file does not exist"
        print("Using weights, checkpoint is {}".format(args.pretrained_weight))
        loaded_dict = torch.load(args.pretrained_weight, map_location=device)
        model.encoder_cnn.load_state_dict(loaded_dict, strict=False)
    else:
        print("not using pretrained weights.")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ############################### train ###############################
    it = 0
    ent_scale = 1e-2
    loss_scale = 1e2
    cons_scale = 1e-2
    while it < args.iters:
        model.train()
        recon_criterion = nn.MSELoss()
        rank_criterion = nn.MarginRankingLoss(0.)
        for step, data in enumerate(train_loader, start=0):
            depths, colored_depths = data
            depths = depths.float()[:, :, None].cuda()  # [b, 2, 1, h, w]
            colored_depths = colored_depths.float().cuda()  # [b, 2, 3, h, w]
            depths = einops.rearrange(depths, 'b t c h w -> (b t) c h w')[:, None].repeat(1, 2, 1, 1, 1)
            colored_depths = einops.rearrange(colored_depths, 'b t c h w -> (b t) c h w')  # [b*2, 3, h, w]
            optimizer.zero_grad()
            recon_depth, _, masks, _ = model(colored_depths)
            recon_loss = loss_scale * recon_criterion(colored_depths, recon_depth)
            entropy_loss = ent_scale * -(masks * torch.log(masks + 1e-5)).sum(dim=1).mean()
            r = (depths * masks).mean([-1, -2])
            r1 = r[:, 0].squeeze()
            r2 = r[:, 1].squeeze()
            y = torch.full((args.batch_size*2,), -1).cuda()
            rank_loss = ent_scale * rank_criterion(r1, r2, y)
            tmasks = einops.rearrange(masks, '(b t) s c h w -> b t s c h w', b=args.batch_size)
            mask_t_1 = tmasks[:, 0]
            mask_t = tmasks[:, 1]
            mask_t = einops.rearrange(mask_t, 'b s c h w -> b c s h w')
            temporal_diff = torch.pow((mask_t_1 - mask_t), 2).mean([-1, -2])
            consistency_loss = cons_scale * temporal_diff.view(-1, 2*2).min(1)[0].mean()
            loss = recon_loss + entropy_loss + consistency_loss + rank_loss
            loss.backward()
            optimizer.step()

            if (it + 1) % args.log_freq == 0:
                print('iteration {}'.format(it+1), 'loss {:.08f}'.format(loss.detach().cpu().numpy()))

            # validation
            if (it + 1) % args.eval_freq == 0:
                # save infos
                infos = {
                    # "optimizer": optimizer.state_dict(),
                    "state_dict": model.state_dict(),
                }

                # save weights
                weights_dir = os.path.join(args.save_weight, args.project_name,)
                if not os.path.exists(weights_dir):
                    os.makedirs(weights_dir)
                torch.save(infos, os.path.join(weights_dir, "Iter_{}".format(it + 1) + ".pth"))

                # val
                for val_data in val_dataset:
                    validate(os.path.join(args.result_dir, args.project_name), val_data, model)

            # LR warmup
            if it < args.warmup_it:
                set_learning_rate(optimizer, args.lr * it / args.warmup_it)

            # LR decay
            if it % args.decay_it == 0 and it > 0:
                set_learning_rate(optimizer, args.lr * (0.5 ** (it // args.decay_it)))
                ent_scale = ent_scale * 5.0
                cons_scale = cons_scale * 5.0

            it += 1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")

    # path config
    parser.add_argument('--data_root', type=str, default="./data/NJUD_NLPR_DUT")
    parser.add_argument('--project_name', type=str, default="DLG")
    parser.add_argument('--pretrained_weight', type=str, default=None)
    parser.add_argument('--save_weight', type=str, default="./checkpoint")
    parser.add_argument('--result_dir', type=str, default="./out")

    # train&val config
    parser.add_argument('--iters', type=int, default=300000)
    parser.add_argument('--train_size', type=int, default=256)
    parser.add_argument('--test_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--eval_freq', type=int, default=1000)

    # LR config
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--warmup_it', type=int, default=200)
    parser.add_argument('--decay_it', type=int, default=8e4)

    args = parser.parse_args()
    main(args)