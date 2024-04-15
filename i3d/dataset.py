# coding: utf-8
from typing import Optional, Tuple, Union, Callable
import math
import numpy as np
import open3d as o3d
import open3d.core as o3c
from plyfile import PlyData
import torch
from torch.utils.data import Dataset
# from iso import levelset_sampling

def _sample_on_surface(vertices: torch.Tensor, n_points: int, device: str = torch.device("cpu")) -> (torch.Tensor, torch.Tensor):
    """Samples points in a torch tensor

    Parameters
    ----------
    vertices: torch.Tensor
        A mode-2 tensor where each row is a vertex.

    n_points: int
        The number of vertices to sample.

    device: str or torch.device
        The device where we should generate the indices of sampled points.

    Returns
    -------
    sampled: torch.tensor
        The points sampled from `vertices`. If
        `n_points` == `vertices.shape[0]`, then we simply return `vertices`.

    idx: torch.tensor
        The indices of points sampled from `vertices`. Naturally, these are
        row indices in `vertices`.

    See Also
    --------
    torch.randperm
    """
    if n_points == vertices.shape[0]:
        return vertices, torch.arange(end=n_points, step=1, device=device)
    idx = torch.randperm(vertices.shape[0], device=device)[:n_points]
    sampled = vertices[idx, ...]
    return sampled, idx


def _curv_segmentation(
    vertices: torch.Tensor,
    n_samples: int,
    bin_edges: np.array,
    proportions: np.array,
    device: str = torch.device("cpu")
):
    """Samples `n_points` points in the mesh based on their curvature.

    Parameters
    ----------
    vertices: torch.Tensor
        The vertices to sample. Note that each row is a vertex and the
        curvatures must be stored in the second-to-last column of `vertices`.

    n_samples: int
        Number of samples to fetch.

    bin_edges: np.array
        The [minimum, low-medium threshold, medium-high threshold, maximum]
        curvature values in `mesh`. These values define thresholds between low
        and medium curvature values, and medium to high curvatures.

    proportions: np.array
        The percentage of points to fetch for each curvature band per batch of
        `n_samples`.

    device: str, optional
        The device to store intermediate results. By default is
        `torch.device("cpu")`.

    Returns
    -------
    samples: torch.Tensor
        The vertices sampled from `mesh`.
    """
    all_curvatures = vertices[..., -2]

    # Assign vertices to buckets based on curvature
    bins = torch.bucketize(all_curvatures.cpu(), torch.Tensor(bin_edges), right=True)
    # print(bins)
    # Sample from each bucket
    samples = []
    for i in range(len(bin_edges) - 1):
        points_in_bin = vertices[bins == i+1]
        if (i == 0):
            n_low_curvature = int(math.floor(proportions[i] * n_samples))
            n_samples_bin = n_low_curvature
        elif (i == 1):
            n_med_curvature = int(math.ceil(proportions[i] * n_samples))
            n_samples_bin = n_med_curvature
        else:
            n_samples_bin = n_samples - (n_low_curvature + n_med_curvature)
        sampled_idx = torch.randperm(points_in_bin.shape[0], device=device)[:n_samples_bin]
        samples.append(points_in_bin[sampled_idx])

    return torch.row_stack(samples)


def _read_ply(path: str, with_curv: bool = False):
    """Reads a PLY file with position, normal and, optionally curvature data.

    ply file contains vertex data (x,y,z ), normal data ( nx,ny,nz) and the curvature stored in the `quality` vertex
    property.
    with_curv == True, set `quality` attribute.

    Parameters
    ----------
    path: str,
        Path to the ply file.

    with_curv: boolean, optional
        If the PLY file has curvatures, read them.

    Returns
    -------
    mesh: o3d.t.geometry.TriangleMesh
        The fully constructed Open3D Triangle Mesh. By default, the mesh is
        allocated on the CPU:0 device.

    vertices: torch.tensor
        The same vertex information as stored in `mesh`, augmented by the SDF
        values as the last column (a column of zeroes). Returned for easier,
        structured access. Note that this tensor is stored in CPU.

    See Also
    --------
    PlyData.read, o3d.t.geometry.TriangleMesh
    """
    # Reading the PLY file with curvature info
    n_columns = 7  # x, y, z, nx, ny, nz, sdf
    if with_curv:
        n_columns = 8  # x, y, z, nx, ny, nz, curvature, sdf
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=(num_verts, n_columns), dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        vertices[:, 3] = plydata["vertex"].data["nx"]
        vertices[:, 4] = plydata["vertex"].data["ny"]
        vertices[:, 5] = plydata["vertex"].data["nz"]
        if with_curv:
            vertices[:, 6] = plydata["vertex"].data["quality"]

        faces = np.stack(plydata["face"].data["vertex_indices"])

    # Converting the PLY data to open3d format
    device = o3c.Device("CPU:0")
    mesh = o3d.t.geometry.TriangleMesh(device)
    mesh.vertex["positions"] = o3c.Tensor(vertices[:, :3], dtype=o3c.float32)
    mesh.vertex["normals"] = o3c.Tensor(vertices[:, 3:6], dtype=o3c.float32)
    if with_curv:
        mesh.vertex["curvatures"] = o3c.Tensor(
            vertices[:, -2], dtype=o3c.float32
        )
    mesh.triangle["indices"] = o3c.Tensor(faces, dtype=o3c.int32)

    return mesh, torch.from_numpy(vertices).requires_grad_(False)


def _create_training_data(
    vertices: torch.tensor,
    n_on_surf: int,
    n_off_surf: int,
    domain_bounds: tuple = ([-1, -1, -1], [1, 1, 1]),
    scene: o3d.t.geometry.RaycastingScene = None,
    device: torch.device = torch.device("cpu"),
    no_sdf: bool = False,
    use_curvature: bool = False,
    curv_fractions: list = [],
    curv_thresholds: list = [],
):
    """Creates a set of training data with coordinates, normals and SDF
    values.

    Param
    ----------
    vertices: torch.tensor
        A mode-2 tensor with the mesh vertices.

    n_on_surf: int
        # of points to sample from the mesh.

    n_off_surf: int
        # of points to sample from the domain. Note that we sample points
        uniformely at random from the domain.

    domain_bounds: tuple[np.array, np.array]
        Bounds to use when sampling points from the domain.

    scene: o3d.t.geometry.RaycastingScene
        Open3D raycasting scene to use when querying SDF for domain points.

    device: str or torch.device, optional
        The compute device where `vertices` is stored. By default its
        torch.device("cpu").

    no_sdf: boolean, optional
        If using SIREN's original loss, set SDF = -1.

    use_curvature: boolean, optional
        Indicates if we must use the curvature to perform sampling on surface
        points. Note that we expect the curvature to be the second-to-last
        column in `vertices`.

    curv_fractions: list, optional
        The fractions of points to sample per curvature band. Only used when
        `use_curvature` is True.

    curv_thresholds: list
        The curvature values to use when defining low, medium and high
        curvatures. Only used when `use_curvature` is True.

    Returns
    -------
    full_points: torch.Tensor
        A tensor with the points sampled from the surface concatenated with the
        off-surface points.

    full_normals: torch.Tensor
        Tensor with the normals of points sampled from the surface concatenated
        with a tensor of zeroes, shaped  (`n_off_surf`, 3), since we don't
        calculate normals for off-surface points.

    full_sdf: torch.Tensor
        Tensor with the SDF values of on and off surface points.

    See Also
    --------
    _sample_on_surface, _lowMedHighCurvSegmentation
    """
    # project
    # projector = levelset_sampling.UniformProjection(
    #     max_points_per_pass=16000, proj_max_iters=10,
        # proj_tolerance=5e-5, knn_k=8)
    # Sample points on the surface
    if use_curvature:
        surf_points = _curv_segmentation(
            vertices, n_on_surf, curv_thresholds, curv_fractions,
            device=device
        )
    else:
        surf_points, _ = _sample_on_surface(vertices, n_on_surf, device=device)

    coord_dict = {
        "on_surf": [surf_points[..., :3], surf_points[..., 3:6], surf_points[..., -1]]
    }

    if n_off_surf > 0:
        domain_points = np.random.uniform(
            domain_bounds[0], domain_bounds[1],
            (n_off_surf, 3)
        )

        if not no_sdf:
            domain_points = o3c.Tensor(domain_points, dtype=o3c.Dtype.Float32)
            domain_sdf = torch.from_numpy(scene.compute_signed_distance(domain_points).numpy())
            domain_points = torch.from_numpy(domain_points.numpy())
        else:
            domain_sdf = torch.full((n_off_surf, 1), fill_value=-1, device=device)

        domain_points = torch.tensor(domain_points, device=device)
        coord_dict["off_surf"] = [
            domain_points, torch.zeros_like(domain_points, device=device), domain_sdf
        ]

    return coord_dict


def _calc_curv_bins(curvatures: torch.Tensor, percentiles: list) -> list:
    """Bins the curvature values according to `percentiles`.

    Parameters
    ----------
    curvatures: torch.Tensor
        Tensor with the curvature values for the vertices.

    percentiles: list
        List with the percentiles. Note that if any values larger than 1 in
        percentiles is divided by 100, since torch.quantile accepts only
        values in range [0, 1].

    Returns
    -------
    quantiles: list
        A list with len(percentiles) + 2 elements composed by the minimum, the
        len(percentiles) values and the maximum values for curvature.

    See Also
    --------
    torch.quantile
    """
    try:
        q = torch.quantile(
            curvatures,
            torch.Tensor(percentiles).to(curvatures.device)
        )
    except RuntimeError:
        percs = [None] * len(percentiles)
        for i, p in enumerate(percentiles):
            if p > 1.0:
                percs[i] = p / 100.0
                continue
            percs[i] = percentiles[i]
        q = torch.quantile(
            curvatures,
            torch.Tensor(percs).to(curvatures.device)
        )

    bins = [curvatures.min().item(), curvatures.max().item()]
    # Hack to insert elements of a list inside another list.
    bins[1:1] = q.data.tolist()
    return bins


class PointCloud(Dataset):
    """SDF Point Cloud dataset.

    Parameters
    ----------
    mesh_path: str
        Path to the base mesh.

    batch_size: integer, optional
        Used for fetching `batch_size` at every call of `__getitem__`. If set
        to 0 (default), fetches all on-surface points at every call.

    off_surface_sdf: number, optional
        Value to replace the SDF calculated by the sampling function for points
        with SDF != 0. May be used to replicate the behavior of Sitzmann et al.
        If set to `None` (default) uses the SDF estimated by the sampling
        function.

    off_surface_normals: torch.Tensor, optional
        Value to replace the normals calculated by the sampling algorithm for
        points with SDF != 0. May be used to replicate the behavior of Sitzmann
        et al. If set to `None` (default) uses the SDF estimated by the
        sampling function.

    use_curvature: boolean, optional
        Indicates if we must use the curvature to perform sampling on surface
        points. By default this is False.

    curv_fractions: list, optional
        The fractions of points to sample per curvature band. Only used when
        `use_curvature` is True.

    curv_percentiles: list, optional
        The curvature percentiles to use when defining low, medium and high
        curvatures. Only used when `use_curvature` is True.

    device: str, optional
        The device to store the tensors. By default its `torch.device("cpu")`.

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
    & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. Retrieved from http://arxiv.org/abs/2006.09661
    """
    def __init__(self, mesh_path: str,
                 batch_size: int,
                 off_surface_sdf: float = None,
                 off_surface_normals: torch.Tensor = None,
                 use_curvature: bool = False,
                 curv_fractions: list = [],
                 curv_percentiles: list = [],
                 device=torch.device("cpu")):
        super().__init__()
        self.use_curvature = use_curvature
        self.curv_fractions = curv_fractions
        self.batch_size = batch_size
        self.off_surface_normals = None
        self.off_surface_sdf = off_surface_sdf
        self.device = device

        if off_surface_normals is not None:
            if isinstance(off_surface_normals, list):
                self.off_surface_normals = torch.Tensor(off_surface_normals)

        print(f"Loading mesh \"{mesh_path}\".")
        print("Using curvatures? ", "YES" if use_curvature else "NO")
        mesh, vertices = _read_ply(
            mesh_path, with_curv=use_curvature
        )
        self.vertices = vertices.to(device)
        if not batch_size:
            self.batch_size = 2 * self.vertices.shape[0]

        print(f"Fetching {self.batch_size // 2} on-surface points per"
              " iteration.")
        print("Creating point-cloud and acceleration structures.")
        self.scene = None
        if off_surface_sdf is None:
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(mesh)

        # Binning the curvatures
        self.curv_bins = None
        if use_curvature:
            self.curv_bins = _calc_curv_bins(
                self.vertices[:, -2],
                curv_percentiles
            )

        self.samples = torch.zeros((self.batch_size, 7), device=device)
        print("Done preparing the dataset.")

    def __len__(self):
        return 2 * self.vertices.shape[0] // self.batch_size

    def __getitem__(self, idx):
        nonsurf = self.batch_size // 2
        noffsurf = self.batch_size - nonsurf
        samples = _create_training_data(
            vertices=self.vertices,
            n_on_surf=nonsurf,
            n_off_surf=noffsurf,
            scene=self.scene,
            device=self.device,
            no_sdf=self.off_surface_sdf is not None,
            use_curvature=self.use_curvature,
            curv_fractions=self.curv_fractions,
            curv_thresholds=self.curv_bins
        )
        self.samples[:nonsurf, :3] = samples["on_surf"][0]
        self.samples[:nonsurf, 3:6] = samples["on_surf"][1]
        self.samples[:nonsurf, -1] = samples["on_surf"][2]
        self.samples[noffsurf:, :3] = samples["off_surf"][0]
        self.samples[noffsurf:, 3:6] = samples["off_surf"][1]
        self.samples[noffsurf:, -1] = samples["off_surf"][2]

        return {
            "coords": self.samples[..., :3].float(),
        }, {
            "normals": self.samples[..., 3:6].float(),
            "sdf": self.samples[..., -1].unsqueeze(1).float()
        }


class PointCloudDeferredSampling(PointCloud):
    """Point Cloud with deferred SDF sampling, i.e. we don't sample SDFs at
    every iteration.

    Parameters
    ----------
    mesh_path: str
        Path to the base mesh.

    batch_size: integer, optional
        Used for fetching `batch_size` at every call of `__getitem__`. If set
        to 0 (default), fetches all on-surface points at every call.

    use_curvature: boolean, optional
        Indicates if we must use the curvature to perform sampling on surface
        points. By default this is False.

    curv_fractions: list, optional
        The fractions of points to sample per curvature band. Only used when
        `use_curvature` is True.

    curv_percentiles: list, optional
        The curvature percentiles to use when defining low, medium and high
        curvatures. Only used when `use_curvature` is True.

    device: str, optional
        The device to store the tensors. By default its `torch.device("cpu")`.

    See Also
    --------
    PointCloud
    """
    def __init__(
            self, mesh_path: str, cfg, batch_size: int, use_curvature: bool = False,
            curv_fractions: list = [], curv_percentiles: list = [],
            device=torch.device("cpu")
    ):
        super(PointCloudDeferredSampling, self).__init__(
            mesh_path, batch_size, use_curvature=use_curvature,
            curv_fractions=curv_fractions,
            curv_percentiles=curv_percentiles,
            device=device
        )
        self.refresh_sdf = True
        # The [beta] that controlls how smooth/sharp the output shape should be
        # If beta >  1, then the output shape will increase in curvature
        #               so it will be sharper
        # If beta < 1, then the output shape will decrease in curvature
        #               so it will be smoother.
        # beta should be > 0.
        self.cfg = cfg
        self.beta = getattr(self.cfg["NFGP"], "beta", 1.)

    def __getitem__(self, _):
        """Returns a batch of points to the caller."""
        nonsurf = self.batch_size // 2
        if self.refresh_sdf:
            noffsurf = self.batch_size - nonsurf
            samples = _create_training_data(
                vertices=self.vertices,
                n_on_surf=nonsurf,
                n_off_surf=noffsurf,
                scene=self.scene,
                device=self.device,
                use_curvature=self.use_curvature,
                curv_fractions=self.curv_fractions,
                curv_thresholds=self.curv_bins
            )

            off_surf_samples = samples["off_surf"]
            off_surf_samples = [v.to(self.device) for v in off_surf_samples]
            self.samples[noffsurf:, :3] = off_surf_samples[0]
            self.samples[noffsurf:, 3:6] = off_surf_samples[1]
            self.samples[noffsurf:, -1] = off_surf_samples[2]
            self.refresh_sdf = False
        else:
            samples = _create_training_data(
                vertices=self.vertices,
                n_on_surf=nonsurf,
                n_off_surf=0,
                scene=self.scene,
                device=self.device,
                use_curvature=self.use_curvature,
                curv_fractions=self.curv_fractions,
                curv_thresholds=self.curv_bins
            )

        self.samples[:nonsurf, :3] = samples["on_surf"][0]
        self.samples[:nonsurf, 3:6] = samples["on_surf"][1]
        self.samples[:nonsurf, -1] = samples["on_surf"][2]

        return {
            "coords": self.samples[..., :3].float(),
        }, {
            "normals": self.samples[..., 3:6].float(),
            "sdf": self.samples[..., -1].float().unsqueeze(1),
        }


if __name__ == "__main__":
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    p = PointCloud(
        "data/armadillo_curvs.ply", batch_size=10, use_curvature=True,
        curv_fractions=(0.2, 0.7, 0.1), curv_percentiles=(70, 95)
    )
    print(len(p))
    print(p.__getitem__(0))
    print(p.__getitem__(0))
    print(p.__getitem__(0))

    p = PointCloud("data/armadillo.ply", batch_size=10, use_curvature=False)
    print(len(p))
    print(p.__getitem__(0))
    print(p.__getitem__(0))
    print(p.__getitem__(0))

    p = PointCloudDeferredSampling("data/armadillo.ply", batch_size=10,
                                   use_curvature=False)
    print(len(p))
    print(p[0])
    print(p[0])
    print(p[0])