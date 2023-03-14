import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def huber_loss(y_pred, theta=0.1):
    """
    Huber Loss from DVIS paper
    https://github.com/jia2lin3yuan1/2020-instanceSeg/blob/main/layers/modules/instance_loss.py
    """
    less_grad_factor = 1. / (2 * theta)
    less_loss_bias = less_grad_factor * theta ** 2
    less_than_theta = (y_pred < theta).float()

    loss = less_than_theta * (y_pred ** 2) * less_grad_factor + \
           (1 - less_than_theta) * (y_pred - theta + less_loss_bias)

    return loss


def pi_l1_loss(pi_pred, pi_target, margin=1.0, pos_wght=3.0):
    """
    Params: pi_pred -- float tensor in size [N, N]
            pi_target -- float tensor in size [N, N], with value {0, 1}, which
                        ** 1 for pixels belonging to different object
                        ** 0 for pixels belonging to same object
            margin/pos_wght -- float
    """
    loss_pi_1 = huber_loss(F.relu(margin - pi_pred))
    loss_pi_0 = huber_loss(pi_pred)

    loss = pos_wght*pi_target*loss_pi_1 + loss_pi_0*(1.0 - pi_target)
    return loss


class BinaryLoss(nn.Module):
    """Compute binary loss to force background pixels close to 0 and foreground pixels far from 0"""
    def __init__(self, margin=2.0):
        super(BinaryLoss, self).__init__()

        self.margin = margin

    def forward(self, y_pred, y_true):
        isFG = (y_true > 0.5).float()
        loss_0 = huber_loss(F.relu(y_pred))
        loss_1 = huber_loss(F.relu(self.margin - y_pred))

        loss = (1 - isFG)*loss_0 + isFG*loss_1

        return loss.mean().float()


class PermuInvLoss(nn.Module):
    """
    This class compute the permutation-invariant loss between the targets and the predictions.
    The pixels used to compute pi_loss is sampled on each object
    It encourages pixels in same instance close to each other,
                  pixels from different instances far away from each other.
    """
    def __init__(self, margin=1.0, pi_pairs=4096, avg_num_obj=16, pos_wght=3.0, FG_stCH=1):
        super(PermuInvLoss, self).__init__()

        self.margin = margin
        self.pi_pairs = pi_pairs
        self.avg_num_obj = avg_num_obj
        self.pos_wght = pos_wght
        self.fg_stCH = FG_stCH

    def sampling_over_objects(self, targets_onehot, BG=0):
        """
        Params:
            targets_onehot -- tensor in [N, ch] with integers values.
            target_classes -- if not None, in size [ch], ch is No. of Objs in targets.
            BG -- if True, BG is counted as one object.
        """
        eff_idx, smpl_wght = [], []
        cnts = targets_onehot.sum(axis=0)
        num_obj = (cnts>0).sum() if BG else (cnts[self.fg_stCH:] > 0).sum()

        # sample over each object
        avg = self.pi_pairs//num_obj

        for k in range(cnts.size(0)):
            if cnts[k] == 0 or (BG == 0 and k < self.fg_stCH):
                continue

            # sample index on current object
            idx = targets_onehot[:, k].nonzero()
            perm = torch.randperm(idx.size(0))
            cur_sel = idx[perm][:avg]
            smpl_size = torch.FloatTensor([cur_sel.size(0)]).cuda()
            obj_wght = torch.pow(self.pi_pairs/(smpl_size + 1.), 1./3)

            # add into the whole stack.
            eff_idx.append(cur_sel)
            smpl_wght.append(torch.ones(cur_sel.size(0), 1, dtype=torch.float)*obj_wght)

        if len(eff_idx) == 0:
            return None, None
        else:
            eff_idx = torch.cat(eff_idx, axis=0).squeeze()  # [N]
            smpl_wght = torch.cat(smpl_wght, axis=0)  # [N,1]
            return eff_idx, smpl_wght

    def forward(self, y_pred, y_true, BG=False, sigma=1e-2):
        """
        Compute the permutation invariant loss on pixel pairs if both pixels are in the instances.
        Params:
            preds -- [bs, 1, ht, wd] from Relu(). here, ch could be:
            targets -- [bs, ch', ht, wd]. onehot matrix
            weights -- [bs, 1, ht, wd]
            target_ids -- [bs, ch']
            BG -- if True, treat BG as one instance.
                | if False, don't sample point on BG pixels,
                            and compute difference only on FG channels if ch>1
        """
        bs, ch, ht, wd = y_pred.size()

        # reshape
        preds_1D = y_pred.view(bs, ch, ht*wd).permute(0, 2, 1)  # in size [bs, N, ch]
        targets_1D = y_true.view(bs, -1, ht*wd).permute(0, 2, 1)  # in size [bs, N, ch']

        # compute loss for each sample
        all_loss = []

        for b in range(bs):
            with torch.no_grad():
                smpl_idx, smpl_wght = self.sampling_over_objects(targets_1D[b], BG=BG)

            if smpl_idx is None:
                continue

            # Compute pairwise differences over pred/target/weight
            smpl_pred = preds_1D[b][smpl_idx]
            smpl_target = targets_1D[b][smpl_idx].float()

            pi_pred = torch.clamp(torch.abs(smpl_pred-smpl_pred.permute(1, 0)), max=5.)

            with torch.no_grad():
                target_numi = torch.matmul(smpl_target, smpl_target.permute(1, 0))
                target_tmp = smpl_target.pow(2).sum(axis=1).pow(0.5)
                target_demi = torch.matmul(target_tmp[:, None], target_tmp[None, :])
                pi_target = ((target_numi/target_demi) < 0.5).float()

            pi_obj_wght = (smpl_wght + smpl_wght.permute(1, 0))

            # Compute loss
            loss = pi_l1_loss(pi_pred, pi_target, self.margin, self.pos_wght)

            if pi_obj_wght is not None:
                loss = torch.mul(loss, pi_obj_wght)

            flag = (loss > sigma).float()
            loss = (loss * flag).sum() / (flag.sum() + 1.)
            all_loss.append(loss)

        return torch.stack(all_loss).mean()


class QuantizationLoss(nn.Module):
    """
    This class computes the quantization distance on predictions of the network.
    It only works for the regression model that network directly output 1 channel instance label
    """
    def __init__(self):
        super(QuantizationLoss, self).__init__()

    def forward(self, y_pred):
        """
        @Param: y_pred -- instance map after relu. size [bs, 1, ht, wd], only on foreground
        """
        assert(y_pred.size()[1] == 1)

        return torch.abs(y_pred[y_pred > 0] - torch.round(y_pred[y_pred > 0])).mean()


class RegularizationLoss(nn.Module):
    """
    This class computes the Mumford-Shah regularization loss on the prediction of the network.
    The way to approximate it is introduced in https://arxiv.org/abs/2007.11576
    """
    def __init__(self):
        super(RegularizationLoss, self).__init__()

        # create kernel_h/v in size [1, 1, 3, 3]
        kernel_h = np.zeros([1, 1, 3, 3], dtype=np.float32)
        kernel_h[:, :, 1, 0] = 0.0
        kernel_h[:, :, 1, 1] = 1.0
        kernel_h[:, :, 1, 2] = 1.0
        self.kernel_h = torch.FloatTensor(kernel_h).cuda()

        kernel_v = np.zeros([1, 1, 3, 3], dtype=np.float32)
        kernel_v[:, :, 0, 1] = 0.0
        kernel_v[:, :, 1, 1] = 1.0
        kernel_v[:, :, 2, 1] = -1.0
        self.kernel_v = torch.FloatTensor(kernel_v).cuda()

    def forward(self, y_pred):
        """
        @Param: preds -- instance map after relu. size [bs, ch, ht, wd]
        """
        ch = y_pred.size(1)
        loss = []
        for k in range(ch):
            loss_h = F.conv2d(y_pred[:, k: k+1, :, :], self.kernel_h)
            loss_v = F.conv2d(y_pred[:, k: k+1, :, :], self.kernel_v)

            tmp = torch.log(loss_h**2 + loss_v**2 + 1)
            loss.extend([tmp])

        loss = torch.stack(loss)
        if loss.max() > 0:
            return loss[loss > 0].mean()
        else:
            return None


class GlobalLoss(nn.Module):
    """Sum all previous losses to get final loss"""
    def __init__(self):
        super(GlobalLoss, self).__init__()

        self.BinaryLoss = BinaryLoss()
        self.QuantizationLoss = QuantizationLoss()
        self.RegularizationLoss = RegularizationLoss()
        self.PermuInvLoss = PermuInvLoss()

    def forward(self, y_pred, y_true):
        lb = self.BinaryLoss(y_pred, y_true)
        lq = self.QuantizationLoss(y_pred)
        lr = self.RegularizationLoss(y_pred)
        lpi = self.PermuInvLoss(y_pred, y_true)

        return lb + lq + lr + lpi

