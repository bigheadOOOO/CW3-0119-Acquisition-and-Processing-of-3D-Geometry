# coding: utf-8

import torch
import torch.nn.functional as F
from i3d_libs import diff_operators
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

    gradient = diff_operators.gradient(pred_sdf, vertices_posi)

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

    gradient = diff_operators.gradient(pred_sdf, coords)

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

    gradient = diff_operators.gradient(pred_sdf, coords)
    hessian = diff_operators.hessian(pred_sdf, coords)
    # principal directions
    pred_dirs = diff_operators.principal_directions(gradient, hessian)

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

    gradient = diff_operators.gradient(pred_sdf, coords)

    # mean curvature
    pred_curvature = diff_operators.divergence(gradient, coords)
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
