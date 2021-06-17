from __future__ import absolute_import

import time
import collections
import unittest
import numpy as np
import cv2 as cv
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


def checkout_objective(args):
    return mahalanobis_score


def multiscale_pixel_wise_mse(y_s, a, y_t, t_mean, t_std, alpha=10):
    loss = 0
    cnt = 0
    for i, t in enumerate(y_t):
        s = [x[i] for x in y_s]  # ensemble components per scale
        loss += pixel_wise_mse(s, a, t, t_mean[..., i], t_std[..., i], alpha)
        cnt += 1
    return loss / cnt


def pixel_wise_mse(y_s, a, y_t, t_mean, t_std, alpha=10):
    loss = 0
    t_mean = unsqueeze_stat(t_mean)
    t_std = std_clamp(unsqueeze_stat(t_std))
    a = -a.float() * (alpha + 1) + 1
    for y in y_s:
        p = ((y - (y_t - t_mean) * t_std**-1)**2).sum(dim=1)
    loss = (p * a).mean()
    return loss


def multiscale_generic_modeling_loss(objective, x, y_s, a, y_t, 
                                     t_mean, t_std, args=None):
    loss = 0
    cnt = 0
    for i, t in enumerate(y_t):
        s = [x[i] for x in y_s]  # ensemble components per scale
        loss += objective(x, s, a, t,
                          t_mean[..., i], t_std[..., i], args)
        cnt += 1
    return loss / cnt


def multiscale_gaussian_modeling_loss(x, y_s, a, y_t,
                                      t_mean, t_std, args=None):
    return multiscale_generic_modeling_loss(
        gaussian_modeling_loss,
        x, y_s, a, y_t, t_mean, t_std, args)


def gaussian_modeling_loss(x, y_s, a, y_t, t_mean, t_std, args=None):
    loss = 0
    feature_size = y_t.size(1)
    t_mean = unsqueeze_stat(t_mean)
    t_std = std_clamp(unsqueeze_stat(t_std))
    m = (x[:, 1] != 0).unsqueeze(1)
    m &= (t_mean.abs().sum(1, keepdim=True) != 0)

    class CenterCrop2d(nn.Module):
        def __init__(self, crop_size):
            super(CenterCrop2d, self).__init__()
            self.crop_size = int(crop_size)

        def forward(self, x):
            row_pos = int(math.floor((x.size(2) - self.crop_size)/2))
            col_pos = int(math.floor((x.size(3) - self.crop_size)/2))
            return x[:, :,
                     row_pos: row_pos + self.crop_size,
                     col_pos: col_pos + self.crop_size
                     ]

        def extra_repr(self):
            s = 'crop_size={crop_size}'
            return s.format(**self.__dict__)

    for y in y_s:
        if args.use_random_crop:
            cropper = CenterCrop2d(y.size(2))
            m = cropper(m)
            t_mean = cropper(t_mean)
            t_std = cropper(t_std)

        # TODO: the logic of masking is in development. Be cautious to use it.

        if 'mask_tr_outlier' in args.label:
            n_t = (y_t - t_mean) * t_std**-1
            m = n_t.abs() < 3  # 99.9%
            y = y * m.long()
            n_t = n_t * m.long()

        if hasattr(args, 'uncertainty') and args.uncertainty:
            n_t = (y_t - t_mean) * t_std**-1
            # probability weighting learning
            def gaussian_p(x):
                return torch.exp(-x**2 / 2)
            p = gaussian_p(n_t)
            loss += (p * (y - n_t)**2).sum(dim=1).mean()

        elif hasattr(args, 'z_score_mask') and 0 < args.z_score_mask:
            n_t = (y_t - t_mean) * t_std**-1
            m = (y.abs() <= args.z_score_mask).detach()
            loss += ((y - n_t)**2 * m.long()).sum(dim=1).mean()

        elif 'mahalanobis' in args.label:
            loss += ((y - y_t)**2 / t_std**2).sum(dim=1).mean()

        elif 'cosine' in args.label:
            loss -= F.cosine_similarity(y, (y_t - t_mean) * t_std**-1) \
                [m.squeeze(1)].mean()

        else:
            loss += (y - (y_t - t_mean) * t_std**-1) \
                .pow(2).sum(dim=1, keepdim=True)[m].mean()

    return loss / len(y_s)


def compute_gradient_penalty(model, x, y, c=1, m=None, x_area=[1],
                             verbose=False):
    """Calculate the gradient penalty loss for WGAN GP

        coded by @eriklindernoren
        https://github.com/eriklindernoren/PyTorch-GAN
    """
    gradients, alpha = compute_gradient(model, x, y, x_area)
    gradient_penalty = 0

    for grad in gradients:
        loss = (grad.norm(2, dim=1) - c) ** 2
        if m is not None:  # m in (b, h, w)
            loss = loss[m]
        gradient_penalty += loss.mean() / len(gradients)

    if verbose:
        return gradient_penalty, alpha
    else:
        return gradient_penalty


def compute_gradient(model, x, y, x_area=[1]):
    """Calculate the gradient penalty loss for WGAN GP

        coded by @eriklindernoren
        https://github.com/eriklindernoren/PyTorch-GAN
    """
    device = x.device
    alpha = None
    if (x - y).abs().sum() > 1e-5:
        alpha = torch.Tensor(np.random.random((x.size(0), 1, 1, 1))).to(device)
        y = y.to(device)
        x = alpha * x + (1 - alpha) * y
    x = x.requires_grad_(True)

    outputs = model(x)
    if not isinstance(outputs, list):
        outputs = [outputs]

    gradients = []
    if 3 == len(outputs):  # multi-scale
        x_area = [33**2, 17**2, 9**2]  # fast dense inputs
    else:
        x_area = x_area
    for i, output in enumerate(outputs):
        dummy = torch.Tensor(*output.shape).fill_(1.0).to(device)
        # Get gradient w.r.t. input
        grad = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=dummy,
            create_graph=True,
            allow_unused=True,
            only_inputs=True,
        )[0]

        gradients.append(grad / x_area[i])

    return gradients, alpha


def multiscale_ae_mse_loss(x, y_s, a, y_t, t_mean, t_std):
    loss = 0
    cnt = 0
    for i, t in enumerate(y_t):
        s = [x[i] for x in y_s]  # ensemble components per scale
        loss += ae_mse_loss(x, s, a, t, t_mean[..., i], t_std[..., i])
        cnt += 1
    return loss / cnt


def ae_mse_loss(x, y_s, a, y_t, t_mean, t_std):
    loss = 0
    for y in y_s:
        loss += ((y[:, :3] - x)**2).sum(dim=1).mean()
    return loss


def get_multiscale_gaussian_modeling_score(x, y_s, y_t, t_mean, t_std,
                                           val_scores=None, args=None):
    scores = []
    for i, t in enumerate(y_t):
        # if val_scores is not None and i not in [2]:
        #     continue
        s = [x[i] for x in y_s]  # ensemble components per scale
        scores.append(
            get_gaussian_modeling_score(
                x, s, t, t_mean[..., i], t_std[..., i], val_scores[i]
                if val_scores is not None else None, args))
        # weighting with the inverse of RF size
        # if val_scores is not None:
        #    scores[-1] *= (i+1)**2 / 7.
    return scores


def get_gaussian_modeling_score(x, y_s, y_t, t_mean, t_std,
                                val_scores=None, args=None):
    def sq_norm(t, dim=1):
        return (t**2).mean(dim=dim)

    def score_normalize(s, mean, std):
        score = (s - mean) / std.clamp_(1e-5, 1e5)
        if std.size(1) == score.size(1) and std.size(2) == score.size(2):
            score[std <= 1e-5] = 0
        return score

    b = y_t.size(0)
    # e-score
    y_mean = torch.stack(y_s, dim=0).mean(0)
    t_mean = unsqueeze_stat(t_mean)
    t_std = std_clamp(unsqueeze_stat(t_std))
    if 'mahalanobis' in args.label:
        e_score = sq_norm((y_mean - y_t) * t_std**-1)
    elif 'cosine' in args.label:
        e_score = -F.cosine_similarity(y_mean, (y_t - t_mean) * t_std**-1)
    else:
        m = t_std <= 1e-5
        score = y_mean - (y_t - t_mean) * t_std**-1
        if 1 == m.size(2) and 1 < score.size(2):
            m = m.repeat(1, 1, score.size(2), score.size(3))
        score[m] = 0
        e_score = sq_norm(score)
    # v-score
    v_score = sq_norm(torch.stack(y_s, dim=-1)).mean(-1) - \
        sq_norm(y_mean)  # b h w

    if val_scores is None:
        return e_score, v_score

    if 1 == len(y_s):
        return score_normalize(e_score,
                               unsqueeze_score(val_scores[0]),
                               unsqueeze_score(val_scores[1]))
    else:
        return score_normalize(e_score,
                               unsqueeze_score(val_scores[0]),
                               unsqueeze_score(val_scores[1])) + \
            score_normalize(v_score,
                            unsqueeze_score(val_scores[2]),
                            unsqueeze_score(val_scores[3]))


def norm_score(x, t_model, s_model, args, val_scores=None, reduction=False):
    return mahalanobis_score(x, s_model, args, val_scores, reduction)


def mahalanobis_score(x, model, args, val_scores=None, reduction=False):
    device = x.device
    outputs = model(x)[-1:]  # assume the squared outputs

    if val_scores is None:
        means = []
        stds = []
    else:
        score = 0
        f_idx = 0

    for i, s in enumerate(outputs):
        if 3 == len(s.size()):
            s = s.unsqueeze(0)
        scores = s.sum(1).detach()  # bxhxw : assumes Mahalanobis distance > 0
        if val_scores is None:
            if reduction:
                means.append(
                    scores.view(scores.size(0), -1).mean(-1, keepdim=True))
                stds.append(
                    scores.view(scores.size(0), -1).std(-1, keepdim=True))
            else:
                means.append(scores)
                stds.append(scores)
        else:
            if 0 == val_scores[0] and 1 == val_scores[1]:
                score += scores
            else:
                mean = val_scores[0][:, f_idx].to(device)
                std = val_scores[1][:, f_idx].to(device)
                score += (scores - mean) / (std + 1e-9)
            f_idx += 1

    if val_scores is None:
        return torch.stack(means, dim=1), torch.stack(stds, dim=1)
    else:
        return score / len(outputs)


def paired_mse_loss(x, s_model, t_model, keeplist=False, options=None):
    s_outputs = s_model(x)
    t_outputs = t_model(x)

    if s_model.branches is not None:
        for i, branch in enumerate(s_model.branches):
            t_outputs[i] = branch(t_outputs[i])

    s_outputs = s_outputs[-1:]
    t_outputs = t_outputs[-1:]

    main_loss = mse_loss

    if keeplist:
        return [main_loss(s, t) for s, t in zip(s_outputs, t_outputs)]
    else:
        loss = 0
        for s, t in zip(s_outputs, t_outputs):
            loss += main_loss(s, t)
        return loss

def mse_per_stage_score(x, t_model, s_model, args, val_scores=None):
    device = x.device
    outputs = [s_model(x), t_model(x)]

    if val_scores is None:
        means = []
        stds = []
    else:
        score = 0
        f_idx = 0

    if not isinstance(outputs[0], list):
        outputs = [[outputs[0]], [outputs[1]]]

    for i, (s, t) in enumerate(zip(*outputs)):
        if hasattr(s_model, 'branches') and s_model.branches is not None:
            t = s_model.branches[i](t)

        if 3 != i:  # ad-hoc: for the last kd pair
            continue

        if 3 == len(s.size()):
            s = s.unsqueeze(0)
            t = t.unsqueeze(0)

        mse = (s - t).pow(2).mean(1)

        if val_scores is None:
            scores = mse.detach()
            means.append(scores.view(mse.size(0), -1).mean(-1, keepdim=True))
            stds.append(scores.view(mse.size(0), -1).std(-1, keepdim=True))
        else:
            C = 1
            scores = mse.detach()
            mean = val_scores[0][:, f_idx : f_idx + C].to(device)
            std = val_scores[1][:, f_idx : f_idx + C].to(device)
            score += ((scores - mean) / (std + 1e-9)).mean(1)
            f_idx += C

    if val_scores is None:
        return torch.stack(means, dim=1), torch.stack(stds, dim=1)
    else:
        return score / len(outputs)


def unsqueeze_stat(x):
    if 0 == len(x.size()):
        x = x.unsqueeze(0)
    if 1 == len(x.size()):
        return x.unsqueeze(0).unsqueeze(2).unsqueeze(2)
    elif 2 == len(x.size()):
         return x.unsqueeze(2).unsqueeze(2)
    elif 3 == len(x.size()):
        return x.unsqueeze(0)
    elif 4 == len(x.size()):
        return x
    else:
        print(len(x.size()))
        raise NotImplementedError()


def unsqueeze_score(x):
    if 2 == len(x.size()):
        return x.unsqueeze(0)
    elif 0 == len(x.size()):
        return x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


def std_clamp(x):
    return x.clamp(1e-5, 1e5)


def correlation_loss(y_s, dim=0):
    '''Descriptor Compactness. (see Vassileios et al., 2017)
    '''
    num = y_s.size(dim)
    mean = y_s.mean(dim=dim, keepdim=True)
    std = y_s.std(dim=dim, keepdim=True)
    op = 'af,bf->ab' if 1 == dim else 'ca,cb->ab'
    cov = torch.einsum(op, (y_s - mean, y_s - mean)) / num
    corr = cov / torch.einsum(op, (std, std))
    return corr[~torch.eye(corr.size(0)).bool()].pow(2).mean()  # off-diagnal elements


def mse_loss(x, y):
    return F.mse_loss(x, y)


def ae_triplet_loss(x, y, margin=1.0):
    return mse_loss(x, y) + triplet_loss(x, y, margin)


def bn_loss(x, y):
    mean = y.mean(0)
    std = y.std(0)
    mean_loss = F.mse_loss(mean, torch.zeros_like(mean))
    std_loss = F.mse_loss(std, torch.ones_like(std))
    return mean_loss, std_loss


def get_batch_fpr(pred, gt, reduction='sum'):  # false positive / negative
    fp = (pred & ~gt.bool()).float().sum(2).sum(1)  # b
    n = (~gt.bool()).float().sum(2).sum(1)  # b
    m = n != 0
    fp = fp[m]
    n = n[m]
    if 'sum' == reduction:
        return \
            (fp / n).sum(0).item(), fp.size(0)
    else:
        if 0 == fp.size(0):
            return 0
        return (fp / n).sum(0).item() / fp.size(0)


def get_batch_tpr(pred, gt, reduction='sum'):  # true positive / positive or recall
    tp = (pred & gt.bool()).float().sum(2).sum(1)  # b
    p = (gt.bool()).float().sum(2).sum(1)  # b
    m = p != 0  # only for anomaly examples
    tp = tp[m]
    p = p[m]  # otherwise, denominator is zero!
    if 'sum' == reduction:
        if 0 == p.size(0):
            print('get_batch_tpr(): tried to divide by zero!')
            return 0, 1
        return \
            (tp / p).sum(0).item(), p.size(0)
    else:
        if 0 == p.size(0):
            print('get_batch_tpr(): tried to divide by zero!')
            return 0
        return (tp / p).sum(0).item() / p.size(0)


def get_batch_prec(pred, gt, reduction='sum'):  # true positive / (tp + fp)
    tp = (pred & gt.bool()).float().sum(2).sum(1)  # b
    P = pred.float().sum(2).sum(1)
    m = P != 0  # at least one prediction is needed for measuring precision!
    tp = tp[m]
    P = P[m]
    if 'sum' == reduction:
        return \
            (tp / P).sum(0).item(), tp.size(0)
    else:
        if 0 == tp.size(0):
            return 0
        return (tp / P).sum(0).item() / tp.size(0)


def get_batch_pro(pred, gt, reduction='sum'):
    '''
    Per-Region Overlap (PRO) which weights ground-truth regions of different
        size equally. For each connected component whithin the ground truth,
        the relative overlap (TPR) with the thresholded anomaly region is
        computed (by averaging).

        Args:
            pred (BoolTensor): thresholded prediction with the size of (B, H, W)
            gt   (BoolTensor): ground-truth segmentation with the size of (B, H, W)
    '''
    debug = False
    m = gt.int().sum(2).sum(1) > 0  # only calculate anomaly samples
    pred = pred[m]
    gt = gt[m].bool()  # make sure this is of BoolTensor
    b, h, w = pred.size()

    # a permutation may be more efficient than serial markers
    marker = torch.LongTensor(range(h * w))[torch.randperm(h * w)] \
        .view(1, h, w).repeat(b, 1, 1).to(pred.device)

    # get region marker, bg = 0, random numbered marker
    marker = marker * gt.long()
    # Iteration of max-pooling & masking with `gt`
    marker = watershed(marker, gt, verbose=debug)

    pro = torch.Tensor(b).to(pred.device).zero_()
    for i in range(b):  # for each image
        per_region_tpr = []
        for m in marker[i].view(-1).unique().cpu().numpy():  # for each region
            if 0 != m:  # if not background
                tp = (pred[i] & (marker[i] == m)).float().sum()
                p = (marker[i] == m).float().sum()
                per_region_tpr.append(tp / p)
        pro[i] = torch.Tensor(per_region_tpr).mean().item()

    # for debugging with opencv
    debug_watershed = False
    if debug_watershed:  # gt-region-count test
        def unique_count(x, verbose=False):
            t = time.time()
            count = []
            for i in range(x.size(0)):
                count.append(x[i].view(-1).unique().size(0) - 1)  # exclude bg
            if verbose:
                print('{:1.1f} elapsed for unique_count.'
                      .format(time.time() - t))
            return torch.LongTensor(count).to(x.device)

        t = time.time()
        x = marker * gt.long()
        x = watershed(x, gt, verbose=debug)
        region_count = unique_count(x, verbose=debug)
        print('{} time elapsed for gt region count using maxpool.'.format(
            time.time() - t))

        # gt-region-count with opencv watershed
        # $ pip install opencv-python
        # $ apt-get install libxrender1 libsm6 libglib2.0-0 libxext6
        import numpy as np
        import cv2 as cv
        check_count = 0
        t = time.time()
        for i in range(b):
            img = np.uint8((gt[i].unsqueeze(2) * 255).cpu().numpy())
            ret, sure_fg = cv.threshold(img, 0.5*img.max(), 255, 0)

            ret, markers = cv.connectedComponents(sure_fg)
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
            markers = cv.watershed(img, markers)
            check_count += 1
            if region_count[i] != (torch.Tensor(markers).unique().size(0)-1):
                print('{} == {}'.format(
                    region_count[i], torch.Tensor(markers).unique().size(0)-1))
                torch.save(gt[i], '{}_{}.pth'.format('bad', i))
                torch.save(markers, '{}_{}_cv.pth'.format('bad', i))
        print('{} examples are checked.'.format(check_count))
        print('{} time elapsed for gt region count using opencv.'.format(
            time.time() - t))

    if 'sum' == reduction:
        return pro.sum().item(), b
    else:
        return pro.sum().item() / b


def get_thresholds(t, num_samples):
    if 1 < t.size(1) and 1 < t.size(2):
        # use the worst-case for efficient determination of thresholds
        _, max_idx = t.view(t.size(0), -1).max(1)[0].max(0)
        t = t[max_idx]
        return [t.view(-1).kthvalue(
            max(1, math.floor(t.numel() * i / num_samples)))[0]
            for i in range(num_samples, 0, -1)]
    else:  # for classifying roc
        return t.view(-1).sort(descending=True)[0].cpu().numpy()


def get_batch_auc(pred, gt, at_fpr=1.0, y_fn=get_batch_pro, num_samples=40):
    ''' Calculate batch AUPROC

        Args:
            pred (Tensor): a tensor of (N, H, W)
            gt   (Tensor): a tensor of (N, H, W)
    '''
    fpr_to_y_fn = {}
    recall_to_precision = {}

    for threshold in get_thresholds(pred, num_samples):
        pred_ = pred > threshold
        # area_filter_(pred_)
        # print(threshold)
        fpr_avg = get_batch_fpr(pred_, gt, reduction='mean')
        if y_fn == get_batch_fpr:
            fpr_to_y_fn[fpr_avg] = fpr_avg
        else:
            fpr_to_y_fn[fpr_avg] = y_fn(pred_, gt, reduction='mean')
        # print('{} -> {}'.format(fpr_avg, fpr_to_y_fn[fpr_avg]))
        recall_to_precision[fpr_to_y_fn[fpr_avg]] = \
            (get_batch_prec(pred_, gt, reduction='mean'), threshold)
        if fpr_avg > at_fpr:
            break
    fpr_to_y_fn_ordered = collections.OrderedDict(sorted(fpr_to_y_fn.items()))
    return fpr_to_y_fn_ordered, recall_to_precision


def get_batch_ap(pred, gt, at_tpr=1.0, y_fn=get_batch_prec, num_samples=40):
    ''' Calculate batch Average Precision

        Args:
            pred (Tensor): a tensor of (N, H, W)
            gt   (Tensor): a tensor of (N, H, W)
    '''
    tpr_to_y_fn = {}
    recall_to_precision = {}

    for threshold in get_thresholds(pred, num_samples):
        pred_ = pred > threshold
        tpr_avg = get_batch_tpr(pred_, gt, reduction='mean')
        tpr_to_y_fn[tpr_avg] = y_fn(pred_, gt, reduction='mean')
        # print('{} -> {}'.format(tpr_avg, tpr_to_y_fn[tpr_avg]))
        recall_to_precision[tpr_avg] = (tpr_to_y_fn[tpr_avg], threshold)
        if tpr_avg > at_tpr:
            break
    tpr_to_y_fn_ordered = collections.OrderedDict(sorted(tpr_to_y_fn.items()))
    return tpr_to_y_fn_ordered, recall_to_precision


def evaluate(pred, gt, method='auproc', at_fpr=.3, num_samples=40,
             verbose=True, approximation='slope'):

    if 'ap' == method:
        scores, recall_to_precision = get_batch_ap(pred, gt, at_tpr=at_fpr, 
                                                   y_fn=get_batch_prec,
                                                   num_samples=num_samples)
    else:
        # get batch-AUPROC scores
        if 'auproc' == method:
            y_fn = get_batch_pro
        elif 'fpr' == method:
            y_fn = get_batch_fpr
        else:  # AUROC
            y_fn = get_batch_tpr

        scores, recall_to_precision = get_batch_auc(pred, gt, at_fpr, y_fn=y_fn,
                                                    num_samples=num_samples)

    def calc_bar_area(scores, at_fpr=1.0):
        acut = 0.  # area cut
        area = 0.  # area all
        fpr = []
        pro = []
        for i, (f, p) in enumerate(scores.items()):
            fpr.append(f)
            pro.append(p)
        for i in range(len(scores)):
            # calculate bin_size
            assert 1 < len(scores)
            if len(fpr) - 1 != i:
                fpr_right = fpr[i+1]
            else:
                fpr_right = 1.0
            b_left = (fpr[i] - fpr[i-1]) / 2
            b_right = (fpr_right - fpr[i]) / 2
            if 0 == i:  # left-end
                b = fpr[i] + b_right
            elif len(fpr) - 1 == i:  # right-end
                b = b_left + 1. - fpr[i]
            else:
                b = b_left + b_right
            # calculate area
            if fpr[i] + b_right > at_fpr:
                b_cut = max(0, at_fpr - fpr[i] + b_left)  # bin cut
                acut += b_cut * pro[i]
            else:
                acut += b * pro[i]
            area += b * pro[i]
        return acut / at_fpr

    def calc_slope_area(scores, at_fpr=1.0):
        acut = 0.  # area cut
        fpr = []
        pro = []
        for i, (f, p) in enumerate(scores.items()):
            fpr.append(f)
            pro.append(p)
        for i in range(len(scores)):
            if 0 == i:  # left-end
                acut += fpr[i] * pro[i] / 2
            elif len(fpr) - 1 == i:  # right-end, TODO: more precisely?
                pro_mid = (pro[i] + pro[i-1]) / 2
                acut += (at_fpr - fpr[i-1]) * (pro_mid + pro[i-1]) / 2
            else:
                acut += (fpr[i] - fpr[i-1]) * (pro[i] + pro[i-1]) / 2
        return acut / at_fpr

    if 'bar' == approximation:
        return calc_bar_area(scores, at_fpr), recall_to_precision
    else:
        return calc_slope_area(scores, at_fpr), recall_to_precision
