from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np
import torch.nn.functional as F
import torch


def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    smooth = 1e-3
    # print('vpshape:', v_p.shape)
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    # print(v_p.shape,v_l.shape)
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        # print(tprec)
        tsens = cl_score(v_l,skeletonize(v_p))
        # print(tsens)
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return (2*tprec*tsens+smooth)/(tprec+tsens+smooth)
    # return (2*tprec*tsens)/(tprec+tsens)





def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))

    return torch.min(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def binary_soft_clDice_loss(pred, label, smooth=1e-4, iter_=1):
    assert pred.size() == label.size()
    num = pred.size()[0]
    pred_flat = pred.view(pred.size()[0], -1)
    label_flat = label.view(pred.size()[0], -1)
    pred_skel_flat = soft_skel(pred, iter_).view(pred.size()[0], -1)
    label_skel_flat = soft_skel(label, iter_).view(pred.size()[0], -1)
    tprec = ((label_flat * pred_skel_flat).sum(1) + smooth) / (pred_skel_flat.sum(1) + smooth)
    tsens = ((pred_flat * label_skel_flat).sum(1) + smooth) / (label_skel_flat.sum(1) + smooth)
    soft_cl = 2 * tprec * tsens / (tprec + tsens)
    # return 1 - soft_cl.sum() / num
    return soft_cl.sum() / num



