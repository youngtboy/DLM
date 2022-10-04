import torch
import torch.nn.functional as F


def _dlm_loss(pred, dlg_mask, cdcp_mask, pre_mask, dlg_um, cdcp_um, pre_um):
    dlg_mask = dlg_mask[:, 0].unsqueeze(1)
    cdcp_mask = cdcp_mask[:, 0].unsqueeze(1)
    pre_mask = pre_mask[:, 0].unsqueeze(1)
    dlg_bce = F.binary_cross_entropy_with_logits(pred, dlg_mask, reduction='none')
    cdcp_bce = F.binary_cross_entropy_with_logits(pred, cdcp_mask, reduction='none')
    pre_bce = F.binary_cross_entropy_with_logits(pred, pre_mask, reduction='none')
    intersection = (dlg_mask.int() & cdcp_mask.int()) + ((1-dlg_mask).int() & (1-cdcp_mask).int())
    l_base1 = (intersection*cdcp_bce).mean()
    l_uc1 = ((((1/torch.pow(torch.exp(dlg_um), 2))*dlg_bce + dlg_um)).mean() +
           (((1/torch.pow(torch.exp(cdcp_um), 2))*cdcp_bce + cdcp_um)).mean() +
           (((1/torch.pow(torch.exp(pre_um), 2))*pre_bce + pre_um)).mean())
    return l_base1+l_uc1


def dlm_loss(pred, dlg_mask, cdcp_mask, pre_mask, dlg_um, cdcp_um, pre_um, epoch, update_freq, all_epochs):
    ours_mask = dlg_mask[:, 0].unsqueeze(1)
    cdcp_mask = cdcp_mask[:, 0].unsqueeze(1)
    pre_mask = pre_mask[:, 0].unsqueeze(1)
    epoch = epoch - (epoch % update_freq)
    alpha = 2*(1 - epoch / all_epochs) ** 0.9
    beta = 2-alpha
    ours_bce = F.binary_cross_entropy_with_logits(pred, ours_mask[:, 0].unsqueeze(1), reduction='none')
    cdcp_bce = F.binary_cross_entropy_with_logits(pred, cdcp_mask[:, 0].unsqueeze(1), reduction='none')
    pre_bce = F.binary_cross_entropy_with_logits(pred, pre_mask[:, 0].unsqueeze(1), reduction='none')
    intersection = (ours_mask.int() & cdcp_mask.int() & pre_mask.int()) + ((1-ours_mask).int() & (1-cdcp_mask).int() & (1-pre_mask).int())
    l_base1 = alpha * (intersection*cdcp_bce).mean()
    l_uc1 = beta*((((1/torch.pow(torch.exp(dlg_um), 2))*ours_bce + dlg_um)).mean() +
                 (((1/torch.pow(torch.exp(cdcp_um), 2))*cdcp_bce + cdcp_um)).mean() +
                 (((1/torch.pow(torch.exp(pre_um), 2))*pre_bce + pre_um)).mean())
    l_update = beta * F.binary_cross_entropy_with_logits(pred, pre_mask, reduction='none')
    return l_base1 + l_update.mean() + l_uc1


def _dlg_loss(pred, cdcp_mask, pre_mask, cdcp_um, pre_um, epoch, update_freq=5, all_epochs=200):
    cdcp_mask = cdcp_mask[:, 0].unsqueeze(1)
    pre_mask = pre_mask[:, 0].unsqueeze(1)
    epoch = epoch - (epoch % update_freq)
    alpha = 2*(1 - epoch / all_epochs) ** 0.9
    beta = 2-alpha
    cdcp_bce = F.binary_cross_entropy_with_logits(pred, cdcp_mask, reduction='none')
    pre_bce = F.binary_cross_entropy_with_logits(pred, pre_mask, reduction='none')
    intersection = (pre_mask.int() & cdcp_mask.int()) + ((1-pre_mask).int() & (1-cdcp_mask).int())
    l_base2 = alpha*(intersection*cdcp_bce).mean()
    l_um2 = beta*((((1/torch.pow(torch.exp(cdcp_um), 2))*cdcp_bce + cdcp_um)).mean() +
           (((1/torch.pow(torch.exp(pre_um), 2))*pre_bce + pre_um)).mean())
    return l_base2 + l_um2