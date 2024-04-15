# coding: utf-8

import torch
import torch.nn.functional as F
import i3d.diff_operators
# TODO: NFGP
from NFGP import igp_utils
from NFGP.losses.eikonal_loss import loss_eikonal
from i3d.diff_operators import laplace
# TODO: END
def zero_tensor(input_tensor):
    """Create a zero tensor with the same shape as input."""
    return torch.zeros(input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device)
def sdf_loss(condition_flag, select_a, select_b):
    return torch.where(
        condition_flag,
        select_a,
        select_b
    )
def normal_alignment(ground_truth_sdf, ground_truth_vectors, pred_vectors):
    """
        Compute loss based on the alignment of normals.

        Args:
        - mask (torch.Tensor): Mask indicating boundary condition.
        - gt_normals (torch.Tensor): Ground-truth normals.
        - gradient (torch.Tensor): Computed gradient.

        Returns:
        - loss (torch.Tensor): Computed loss based on normal alignment.
    """
    return sdf_loss(
        ground_truth_sdf == 0,
        1 - F.cosine_similarity(pred_vectors, ground_truth_vectors, dim=-1)[..., None],
        zero_tensor(ground_truth_sdf)
    )
def compute_eikonal_loss(gradient_norm):
    """
        Compute Eikonal constraint loss.

        Args:
        - gradient_norm (torch.Tensor): Norm of the computed gradient.

        Returns:
        - loss (torch.Tensor): Computed Eikonal constraint loss.
    """
    return (gradient_norm - 1.) ** 2 # 1.: make it float

def sdf_sitzmann(model_construct, ground_truth):
    """
        Compute the Sitzmann et al. loss function for SDF experiments [1].

        Args:
        - model_construct (dict): Model output with 'model_in' and 'model_out' keys.
        - ground_truth (dict): Ground-truth data with 'sdf' and 'normals' keys.

        Returns:
        - loss (dict): Calculated loss values for each constraint.

        References:
        [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
        & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
        Activation Functions. ArXiv. Retrieved from http://arxiv.org/abs/2006.09661
    """
    ground_truth_sdf = ground_truth["sdf"]
    ground_truth_normals = ground_truth["normals"]

    vertices_posi = model_construct["model_in"]
    pred_sdf = model_construct["model_out"]

    gradient = i3d.diff_operators.gradient(pred_sdf, vertices_posi)

    # Initial-boundary constraints
    boundary_condition = (ground_truth_sdf != -1)
    sdf_off_loss = sdf_loss(boundary_condition, pred_sdf, zero_tensor(pred_sdf))
    inter_loss = sdf_loss(boundary_condition,
                                       zero_tensor(pred_sdf),
                                       torch.exp(-1e2 * torch.abs(pred_sdf))).mean() * 1e2
    normal_loss = sdf_loss(boundary_condition,
                                        1 - F.cosine_similarity(gradient, ground_truth_normals, dim=-1)[..., None],
                                        zero_tensor(gradient[..., :1])).mean() * 1e2
    grad_loss = torch.abs(gradient.norm(dim=-1) - 1.).mean() * 5e1

    return_dict = {
        "sdf_off_loss": torch.abs(sdf_off_loss).mean() * 3e3,
        "inter_loss": inter_loss,
        "normal_loss": normal_loss,
        "grad_loss": grad_loss,
    }
    return return_dict


def loss_true_sdf(X, ground_truth):
    """
        Compute the loss based on true SDF values for off-surface points.

        Args:
        - X (dict): Model output with 'model_in' and 'model_out' keys.
        - ground_truth (dict): Ground-truth data with 'sdf' and 'normals' keys.

        Returns:
        - loss (dict): Calculated loss values for each constraint.
    """
    # Extract ground truth and predictions
    ground_truth_sdf = ground_truth['sdf'] # the value of Ground true distance field.
    ground_truth_normals = ground_truth['normals']

    coords = X['model_in']
    pred_sdf = X['model_out']

    gradient = i3d.diff_operators.gradient(pred_sdf, coords)

    # Initial-boundary constraints
    # Determine points on and off the surface
    on_sdf_mask = (ground_truth_sdf == 0)  # On the surface
    off_sdf_mask = ~on_sdf_mask  # Off the surface

    sdf_on_surf_loss = sdf_loss(on_sdf_mask, pred_sdf ** 2, zero_tensor(pred_sdf))
    sdf_off_surf_loss = sdf_loss(off_sdf_mask, (ground_truth_sdf - pred_sdf) ** 2, zero_tensor(pred_sdf))
    normal_loss_loss = normal_alignment(ground_truth_sdf, ground_truth_normals, gradient)

    # PDE constraints
    grad_loss_loss = compute_eikonal_loss(gradient.norm(dim=-1))[..., None]

    return {
       'sdf_on_surf': sdf_on_surf_loss.mean() * 3e3,
       'sdf_off_surf': sdf_off_surf_loss.mean() * 2e2,
       'normal_loss': normal_loss_loss.mean() * 1e2,  # 1e1,
       'grad_loss': grad_loss_loss.mean() * 5e1  # 1e1
    }


def sdf_align_principal_directions(model_output, ground_truth):
    """
       Loss function that aligns the principal directions on the 0 level-set.

       Args:
       - model_output (dict): Model output with 'model_in' and 'model_out' keys.
       - ground_truth (dict): Ground-truth data with 'sdf', 'normals', 'min_curv',
                              'max_curv', 'max_principal_directions' keys.

       Returns:
       - loss (dict): Calculated loss values for each constraint.
    """
    ground_truth_sdf = ground_truth['sdf']
    ground_truth_normals = ground_truth['normals']
    ground_truth_min_curvature = ground_truth["min_curv"]
    ground_truth_max_curvature = ground_truth["max_curv"]
    ground_truth_dirs = ground_truth["max_principal_directions"]

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = i3d.diff_operators.gradient(pred_sdf, coords)
    hessian = i3d.diff_operators.hessian(pred_sdf, coords)
    # principal directions
    pred_dirs = i3d.diff_operators.principal_directions(gradient, hessian)

    on_sdf_mask = (ground_truth_sdf == 0)  # On the surface
    off_sdf_mask = ~on_sdf_mask  # Off the surface

    dirs_loss = sdf_loss(on_sdf_mask,
                                      1 - (F.cosine_similarity(pred_dirs[0][..., :3], ground_truth_dirs, dim=-1)[..., None])**2,
                                      zero_tensor(ground_truth_sdf))

    aux_dirs_loss = sdf_loss(
        on_sdf_mask,
        F.cosine_similarity(pred_dirs[0][..., :3], ground_truth_normals, dim=-1)[..., None]**2,
        zero_tensor(ground_truth_sdf)
    )

    dirs_loss = dirs_loss + 0.1*aux_dirs_loss

    # Removing problematic curvatures and planar points
    planar_curvature = 0.5 * torch.abs(ground_truth_min_curvature - ground_truth_max_curvature)
    dirs_loss = sdf_loss(
        planar_curvature > 10 & planar_curvature < 5000,
        dirs_loss,
        zero_tensor(dirs_loss)
    ).mean()
    # dirs_loss = sdf_loss(planar_curvature < 5000, dirs_loss, zero_tensor(dirs_loss)).mean()

    return {
        "sdf_on_surf": sdf_loss(on_sdf_mask, pred_sdf ** 2, zero_tensor(pred_sdf)).mean() * 3e3,
        "sdf_off_surf": sdf_loss(off_sdf_mask, (ground_truth_sdf - pred_sdf) ** 2, zero_tensor(pred_sdf)).mean() * 2e2,
        "normal_loss": normal_alignment(ground_truth_sdf, ground_truth_normals, gradient).mean() * 1e2,  # 1e1,
        "grad_loss": compute_eikonal_loss(gradient.norm(dim=-1)).mean() * 5e1,
        "dirs_loss": dirs_loss
    }


def sdf_mean_curvf(model_output, ground_truth):
    """
        Loss function that fits the mean curvatures on the 0 level-set.

        Args:
        - model_output (dict): Model output with 'model_in' and 'model_out' keys.
        - ground_truth (dict): Ground-truth data with 'sdf', 'normals', and 'curvature' keys.

        Returns:
        - loss (dict): Calculated loss values for each constraint.
    """
    ground_truth_sdf = ground_truth['sdf']
    ground_truth_normals = ground_truth['normals']
    ground_truth_curvature = ground_truth["curvature"]

    on_sdf_mask = (ground_truth_sdf == 0)  # On the surface

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = i3d.diff_operators.gradient(pred_sdf, coords)

    # mean curvature
    pred_curvature = i3d.diff_operators.divergence(gradient, coords)
    curv_loss = torch.where(
        on_sdf_mask,
        (pred_curvature - ground_truth_curvature) ** 2,
        zero_tensor(pred_curvature)
    )

    return {
        'sdf_on_surf': sdf_loss(on_sdf_mask, pred_sdf ** 2, zero_tensor(pred_sdf)).mean() * 3e3,
        'sdf_off_surf': sdf_loss(~on_sdf_mask, (ground_truth_sdf - pred_sdf) ** 2, zero_tensor(pred_sdf)).mean() * 2e2,
        'normal_loss': normal_alignment(ground_truth_sdf, ground_truth_normals, gradient).mean() * 1e2,  # 1e1,
        'grad_loss': compute_eikonal_loss(gradient.norm(dim=-1))[..., None].mean() * 5e1,
        'curv_loss': curv_loss.mean() * 1e-1
    }
# TODO: NFGP LOSS SHAPE
def loss_boundary(gtr, net, npoints=1000, dim=3, x=None, use_surf_points=False):
    """
    This function tries to enforce that the field [gtr] and [net] are similar.
    Basically computing |gtr(x) - net(x)| for some [x].
    [x] will be sampled from surface of [gtr] if [use_surf_points] is True
    Otherwise, [x] is sampled from [-1, 1]^3

    :param gtr:
    :param net:
    :param npoints:
    :param dim:
    :param x:
    :param use_surf_points:
    :return:
    """
    if x is None:
        x, _ = igp_utils.sample_points(
            npoints, dim=dim, sample_surf_points=use_surf_points,
            invert_sampling=False, out_nf=gtr, deform=None
        )
        x = x.detach().cuda().float()
        bs = 1
        x = x.view(bs, npoints, dim)
    else:
        if len(x.size()) == 2:
            bs, npoints = 1, x.size(0)
        else:
            bs, npoints = x.size(0), x.size(1)
    x = x.view(bs, npoints, dim)

    if use_surf_points:
        net_y = net(x)
        loss_all = F.mse_loss(net_y, torch.zeros_like(net_y), reduction='none')
    else:
        net_y = net(x)
        gtr_y = gtr(x)
        loss_all = F.mse_loss(net_y, gtr_y, reduction='none')
    loss_all = loss_all.view(bs, npoints)
    loss = loss_all.mean()
    return loss, x

def smoothing_loss(original_decoder, decoder, opt_dec, beta,
                   num_update_step = 0,
                   boundary_loss_weight = 1.,
                   boundary_loss_num_points = 0,
                   boundary_loss_points_update_step = 1,
                   boundary_loss_use_surf_points = True,
                   grad_norm_weight = 1e-2,
                   grad_norm_num_points = 5000,
                   lap_loss_weight=1e-4,
                   lap_loss_threshold=50,
                   lap_loss_num_points=5000
                   ):
    num_update_step += 1
    if boundary_loss_weight > 0. and boundary_loss_num_points > 0:
        if num_update_step % boundary_loss_points_update_step == 0:
            boundary_points = None
        loss_y_boundary, boundary_points = loss_boundary(
            (lambda x: original_decoder(x)),
            (lambda x: decoder(x)),
            npoints=boundary_loss_num_points,
            x=boundary_points,
            dim=3,
            use_surf_points=boundary_loss_use_surf_points)
        loss_y_boundary = loss_y_boundary * boundary_loss_weight
    else:
        loss_y_boundary = torch.zeros(1).float().cuda()

    if grad_norm_weight > 0. and grad_norm_num_points > 0:
        loss_unit_grad_norm = loss_eikonal(
            lambda x: decoder(x),
            npoints=grad_norm_num_points,
            use_surf_points=False, invert_sampling=False
        )
        loss_unit_grad_norm *= grad_norm_weight
    else:
        loss_unit_grad_norm = torch.zeros(1).float().cuda()

    if lap_loss_weight > 0. and lap_loss_num_points > 0:
        loss_lap_scaling = loss_lap(
            (lambda x: original_decoder(x)),
            (lambda x: decoder(x)),
            npoints=lap_loss_num_points,
            beta=beta,
            masking_thr=lap_loss_threshold,
        )
        loss_lap_scaling = loss_lap_scaling * lap_loss_weight
    else:
        loss_lap_scaling = torch.zeros(1).float().cuda()

    loss = loss_unit_grad_norm + loss_y_boundary + loss_lap_scaling

    loss.backward()
    opt_dec.step()

    return {
        'loss': loss.detach().cpu().item(),
        'scalar/loss/loss': loss.detach().cpu().item(),
        'scalar/loss/loss_boundary': loss_y_boundary.detach().cpu().item(),
        'scalar/loss/loss_eikonal': loss_unit_grad_norm.detach().cpu().item(),
        'scalar/loss/loss_lap_scaling': loss_lap_scaling.detach().cpu().item(),
        'scalar/weight/loss_boundary': boundary_loss_weight,
        'scalar/weight/loss_eikonal': grad_norm_weight,
        'scalar/weight/loss_lap': lap_loss_weight,
    }

def loss_lap(
        gtr, net, deform=None,
        x=None, npoints=1000, dim=3,
        beta=1., masking_thr=10, return_mask=False, use_weights=False, weights=1
):
    """
    Matching the Laplacian between [gtr] and [net] on sampled points.

    :param gtr:
    :param net:
    :param deform:
    :param x:
    :param npoints:
    :param dim:
    :param use_surf_points:
    :param invert_sampling:
    :param beta:
    :param masking_thr:
    :param return_mask:
    :param use_weights:
    :param weights:
    :return:
    """
    if x is None:
        x, weights = igp_utils.sample_points(
            npoints, dim=dim, sample_surf_points=False,
            out_nf=gtr, inp_nf=None, deform=None, invert_sampling=False,
        )
        bs, npoints = x.size(0), x.size(1)
    else:
        if len(x.size()) == 2:
            bs, npoints = 1, x.size(0)
        else:
            bs, npoints = x.size(0), x.size(1)
    x = x.view(bs, npoints, dim)

    if deform is None:
        gtr_x = x
    else:
        gtr_x = deform(x, None)
    gtr_x = gtr_x.view(bs, npoints, dim).contiguous()
    if gtr_x.is_leaf:
        gtr_x.requires_grad = True
    else:
        gtr_x.retain_grad()
    gtr_y = gtr(gtr_x)
    lap_gtr = laplace(gtr_y, gtr_x, normalize=True).view(bs, npoints)

    if x.is_leaf:
        x.requires_grad = True
    else:
        x.retain_grad()
    net_y = net(x)
    lap_net = laplace(net_y, x, normalize=True).view(*lap_gtr.shape)

    diff = lap_gtr * beta - lap_net
    if masking_thr is not None:
        mask = ((torch.abs(lap_gtr) < masking_thr) &
                (torch.abs(lap_net) < masking_thr))
    else:
        mask = torch.ones_like(lap_gtr) > 0
    loss = F.mse_loss(diff, torch.zeros_like(diff), reduction='none')
    if use_weights:
        loss = loss * weights
    loss = loss[mask].mean()
    if return_mask:
        return loss, mask
    else:
        return loss


# TODO: NFGP END