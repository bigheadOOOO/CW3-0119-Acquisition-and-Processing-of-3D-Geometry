import torch
import torch.nn.functional as F

import i3d.diff_operators
from ..igp_utils import sample_points


def loss_eikonal(
        net,  gtr=None, deform=None,
        npoints=1000, use_surf_points=False, invert_sampling=True,
        x=None, dim=3, reduction='mean', weights=None
):
    if x is None:
        x, weights = sample_points(
            npoints, dim=dim, sample_surf_points=use_surf_points,
            inp_nf=gtr, out_nf=net, deform=deform,
            invert_sampling=invert_sampling,
        )
        bs, npoints = x.size(0), x.size(1)
    else:
        assert weights is not None
        if len(x.size()) == 2:
            bs, npoints = 1, x.size(0)
        else:
            bs, npoints = x.size(0), x.size(1)
    x = x.view(bs, npoints, dim)

    x.requires_grad = True
    y = net(x.view(1, -1, dim))
    grad_norm = i3d.diff_operators.gradient(y, x).norm(dim=-1)

    loss_all = torch.nn.functional.mse_loss(
        grad_norm, torch.ones_like(grad_norm), reduction='none')
    loss_all = loss_all * weights

    if reduction == 'none':
        loss = loss_all
    elif reduction == 'mean':
        loss = loss_all.mean()
    elif reduction == 'sum':
        loss = loss_all.sum()
    else:
        raise NotImplementedError
    return loss

