# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Main class for the implementation of the global alignment
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn

from cloud_opt.dust3r_opt.base_opt import BasePCOptimizer
from dust3r.utils.geometry import xy_grid, geotrf
from dust3r.utils.device import to_cpu, to_numpy


class PointCloudOptimizer(BasePCOptimizer):
    """Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, *args, optimize_pp=False, focal_break=20, **kwargs):
        super().__init__(*args, **kwargs)

        self.has_im_poses = True  # by definition of this class
        self.focal_break = focal_break

        # adding thing to optimize
        self.im_depthmaps = nn.ParameterList(
            torch.randn(H, W) / 10 - 3 for H, W in self.imshapes
        )  # log(depth)
        self.im_poses = nn.ParameterList(
            self.rand_pose(self.POSE_DIM) for _ in range(self.n_imgs)
        )  # camera poses
        self.im_focals = nn.ParameterList(
            torch.FloatTensor([self.focal_break * np.log(max(H, W))])
            for H, W in self.imshapes
        )  # camera intrinsics
        self.im_pp = nn.ParameterList(
            torch.zeros((2,)) for _ in range(self.n_imgs)
        )  # camera intrinsics
        self.im_pp.requires_grad_(optimize_pp)

        self.imshape = self.imshapes[0]
        im_areas = [h * w for h, w in self.imshapes]
        self.max_area = max(im_areas)

        # adding thing to optimize
        self.im_depthmaps = ParameterStack(
            self.im_depthmaps, is_param=True, fill=self.max_area
        )
        self.im_poses = ParameterStack(self.im_poses, is_param=True)
        self.im_focals = ParameterStack(self.im_focals, is_param=True)
        self.im_pp = ParameterStack(self.im_pp, is_param=True)
        self.register_buffer(
            "_pp", torch.tensor([(w / 2, h / 2) for h, w in self.imshapes])
        )
        self.register_buffer(
            "_grid",
            ParameterStack(
                [xy_grid(W, H, device=self.device) for H, W in self.imshapes],
                fill=self.max_area,
            ),
        )

        # pre-compute pixel weights
        self.register_buffer(
            "_weight_i",
            ParameterStack(
                [self.conf_trf(self.conf_i[i_j]) for i_j in self.str_edges],
                fill=self.max_area,
            ),
        )
        self.register_buffer(
            "_weight_j",
            ParameterStack(
                [self.conf_trf(self.conf_j[i_j]) for i_j in self.str_edges],
                fill=self.max_area,
            ),
        )

        # precompute aa
        self.register_buffer(
            "_stacked_pred_i",
            ParameterStack(self.pred_i, self.str_edges, fill=self.max_area),
        )
        self.register_buffer(
            "_stacked_pred_j",
            ParameterStack(self.pred_j, self.str_edges, fill=self.max_area),
        )
        self.register_buffer("_ei", torch.tensor([i for i, j in self.edges]))
        self.register_buffer("_ej", torch.tensor([j for i, j in self.edges]))
        self.total_area_i = sum([im_areas[i] for i, j in self.edges])
        self.total_area_j = sum([im_areas[j] for i, j in self.edges])

    def _check_all_imgs_are_selected(self, msk):
        assert np.all(
            self._get_msk_indices(msk) == np.arange(self.n_imgs)
        ), "incomplete mask!"

    def preset_pose(self, known_poses, pose_msk=None):  # cam-to-world
        self._check_all_imgs_are_selected(pose_msk)

        if isinstance(known_poses, torch.Tensor) and known_poses.ndim == 2:
            known_poses = [known_poses]
        for idx, pose in zip(self._get_msk_indices(pose_msk), known_poses):
            if self.verbose:
                print(f" (setting pose #{idx} = {pose[:3,3]})")
            self._no_grad(self._set_pose(self.im_poses, idx, torch.tensor(pose)))

        # normalize scale if there's less than 1 known pose
        n_known_poses = sum((p.requires_grad is False) for p in self.im_poses)
        self.norm_pw_scale = n_known_poses <= 1

        self.im_poses.requires_grad_(False)
        self.norm_pw_scale = False

    def preset_focal(self, known_focals, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, focal in zip(self._get_msk_indices(msk), known_focals):
            if self.verbose:
                print(f" (setting focal #{idx} = {focal})")
            self._no_grad(self._set_focal(idx, focal))

        self.im_focals.requires_grad_(False)

    def preset_principal_point(self, known_pp, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, pp in zip(self._get_msk_indices(msk), known_pp):
            if self.verbose:
                print(f" (setting principal point #{idx} = {pp})")
            self._no_grad(self._set_principal_point(idx, pp))

        self.im_pp.requires_grad_(False)

    def _get_msk_indices(self, msk):
        if msk is None:
            return range(self.n_imgs)
        elif isinstance(msk, int):
            return [msk]
        elif isinstance(msk, (tuple, list)):
            return self._get_msk_indices(np.array(msk))
        elif msk.dtype in (bool, torch.bool, np.bool_):
            assert len(msk) == self.n_imgs
            return np.where(msk)[0]
        elif np.issubdtype(msk.dtype, np.integer):
            return msk
        else:
            raise ValueError(f"bad {msk=}")

    def _no_grad(self, tensor):
        assert (
            tensor.requires_grad
        ), "it must be True at this point, otherwise no modification occurs"

    def _set_focal(self, idx, focal, force=False):
        param = self.im_focals[idx]
        if (
            param.requires_grad or force
        ):  # can only init a parameter not already initialized
            param.data[:] = self.focal_break * np.log(focal)
        return param

    def get_focals(self):
        log_focals = torch.stack(list(self.im_focals), dim=0)
        return (log_focals / self.focal_break).exp()

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.im_focals])

    def _set_principal_point(self, idx, pp, force=False):
        param = self.im_pp[idx]
        H, W = self.imshapes[idx]
        if (
            param.requires_grad or force
        ):  # can only init a parameter not already initialized
            param.data[:] = to_cpu(to_numpy(pp) - (W / 2, H / 2)) / 10
        return param

    def get_principal_points(self):
        return self._pp + 10 * self.im_pp

    def get_intrinsics(self):
        K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_principal_points()
        K[:, 2, 2] = 1
        return K

    def get_im_poses(self):  # cam to world
        cam2world = self._get_poses(self.im_poses)
        return cam2world

    def _set_depthmap(self, idx, depth, force=False):
        depth = _ravel_hw(depth, self.max_area)

        param = self.im_depthmaps[idx]
        if (
            param.requires_grad or force
        ):  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def get_depthmaps(self, raw=False):
        res = self.im_depthmaps.exp()
        if not raw:
            res = [dm[: h * w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def depth_to_pts3d(self):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)
        # project to world frame
        return geotrf(im_poses, rel_ptmaps)

    def get_pts3d(self, raw=False):
        res = self.depth_to_pts3d()
        if not raw:
            res = [dm[: h * w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def forward(self):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors().unsqueeze(1)
        proj_pts3d = self.get_pts3d(raw=True)

        # rotate pairwise prediction according to pw_poses
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)

        # compute the less
        li = (
            self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum()
            / self.total_area_i
        )
        lj = (
            self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum()
            / self.total_area_j
        )

        return li + lj


def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)


def ParameterStack(params, keys=None, is_param=None, fill=0):
    if keys is not None:
        params = [params[k] for k in keys]

    if fill > 0:
        params = [_ravel_hw(p, fill) for p in params]

    requires_grad = params[0].requires_grad
    assert all(p.requires_grad == requires_grad for p in params)

    params = torch.stack(list(params)).float().detach()
    if is_param or requires_grad:
        params = nn.Parameter(params)
        params.requires_grad_(requires_grad)
    return params


def _ravel_hw(tensor, fill=0):
    # ravel H,W
    tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

    if len(tensor) < fill:
        tensor = torch.cat(
            (tensor, tensor.new_zeros((fill - len(tensor),) + tensor.shape[1:]))
        )
    return tensor


def acceptable_focal_range(H, W, minf=0.5, maxf=3.5):
    focal_base = max(H, W) / (
        2 * np.tan(np.deg2rad(60) / 2)
    )  # size / 1.1547005383792515
    return minf * focal_base, maxf * focal_base


def apply_mask(img, msk):
    img = img.copy()
    img[msk] = 0
    return img


class GPSConstrainedPointCloudOptimizer(PointCloudOptimizer):
    """PointCloudOptimizer with GPS constraints for improved camera path estimation.

    Uses scale-invariant GPS constraints:
    - Displacement ratios: Relative distances between frames
    - Velocity directions: Normalized motion direction vectors
    - Heading changes: Angular changes in movement direction
    """

    def __init__(
        self,
        *args,
        gps_data=None,
        gps_weight=0.1,
        gps_velocity_weight=0.05,
        gps_heading_weight=0.05,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.gps_data = gps_data
        self.gps_weight = gps_weight
        self.gps_velocity_weight = gps_velocity_weight
        self.gps_heading_weight = gps_heading_weight

        # Precompute GPS constraints if data is available
        self.gps_constraints = None
        if gps_data is not None and len(gps_data) >= 2:
            self._precompute_gps_constraints()

    def _precompute_gps_constraints(self):
        """Precompute GPS constraints from GPS data."""
        from dust3r.utils.gps import compute_gps_constraints, compute_gimbal_yaw_changes

        self.gps_constraints = compute_gps_constraints(
            self.gps_data,
            device=self.device,
            min_displacement=0.1  # meters
        )

        if self.gps_constraints is not None:
            # Also compute gimbal yaw changes for hovering fallback
            self.gimbal_yaw_changes = compute_gimbal_yaw_changes(
                self.gps_data,
                device=self.device
            )

            is_hovering = self.gps_constraints.get('is_hovering', False)
            total_disp = self.gps_constraints.get('total_displacement', 0)
            if self.verbose:
                print(f"GPS constraints initialized:")
                print(f"  - Total displacement: {total_disp:.2f}m")
                print(f"  - Is hovering: {is_hovering}")
                print(f"  - GPS weight: {self.gps_weight}")
                print(f"  - Velocity weight: {self.gps_velocity_weight}")
                print(f"  - Heading weight: {self.gps_heading_weight}")

    def _compute_displacement_ratio_loss(self):
        """Compute loss comparing predicted vs GPS displacement ratios.

        This is scale-invariant: we compare relative distances, not absolute ones.
        """
        if self.gps_constraints is None:
            return 0.0

        gps_ratios = self.gps_constraints.get('displacement_ratios')
        if gps_ratios is None or len(gps_ratios) < 1:
            return 0.0

        # Get camera positions from poses
        im_poses = self.get_im_poses()  # [N, 4, 4] cam-to-world
        translations = im_poses[:, :3, 3]  # [N, 3]

        # Match frame IDs to pose indices
        # Assuming poses are in the same order as frame_ids
        n_poses = min(len(translations), len(gps_ratios) + 1)
        if n_poses < 2:
            return 0.0

        # Compute predicted displacements (XY only - ignore Z for zoomed video)
        pred_displacements = []
        for i in range(1, n_poses):
            d_xy = torch.norm(translations[i, :2] - translations[i-1, :2])
            pred_displacements.append(d_xy)

        pred_displacements = torch.stack(pred_displacements)
        total_pred = pred_displacements.sum() + 1e-8

        # Compute predicted ratios
        pred_ratios = pred_displacements / total_pred

        # Compute loss as L1 difference in ratios
        n_compare = min(len(pred_ratios), len(gps_ratios))
        loss = torch.abs(pred_ratios[:n_compare] - gps_ratios[:n_compare]).mean()

        return loss

    def _compute_velocity_direction_loss(self):
        """Compute loss comparing predicted vs GPS velocity directions.

        Uses cosine similarity between normalized velocity vectors.
        """
        if self.gps_constraints is None:
            return 0.0

        gps_directions = self.gps_constraints.get('velocity_directions')
        if gps_directions is None or len(gps_directions) < 1:
            return 0.0

        # Get camera positions from poses
        im_poses = self.get_im_poses()
        translations = im_poses[:, :3, 3]

        n_poses = min(len(translations), len(gps_directions) + 1)
        if n_poses < 2:
            return 0.0

        # Compute predicted velocity directions (XY only)
        pred_velocities = translations[1:n_poses, :2] - translations[:n_poses-1, :2]
        pred_magnitudes = torch.norm(pred_velocities, dim=1, keepdim=True)
        pred_directions = pred_velocities / (pred_magnitudes + 1e-8)

        # Compute cosine similarity loss (1 - cos_sim)
        n_compare = min(len(pred_directions), len(gps_directions))
        gps_dirs = gps_directions[:n_compare]

        # Only compute loss for frames with significant GPS movement
        gps_magnitudes = torch.norm(gps_dirs, dim=1)
        valid_mask = gps_magnitudes > 0.1  # GPS direction is valid

        if valid_mask.sum() == 0:
            return 0.0

        cos_sim = (pred_directions[:n_compare] * gps_dirs).sum(dim=1)
        loss = (1 - cos_sim)[valid_mask].mean()

        return loss

    def _compute_heading_change_loss(self):
        """Compute loss comparing predicted vs GPS heading changes.

        Heading changes are angular, so scale-invariant.
        For hovering, uses gimbal yaw changes as fallback.
        """
        is_hovering = self.gps_constraints is not None and self.gps_constraints.get('is_hovering', False)

        if is_hovering and self.gimbal_yaw_changes is not None:
            # Use gimbal yaw changes directly
            gps_heading_changes = self.gimbal_yaw_changes
        elif self.gps_constraints is not None:
            gps_heading_changes = self.gps_constraints.get('heading_changes')
        else:
            return 0.0

        if gps_heading_changes is None or len(gps_heading_changes) < 1:
            return 0.0

        # Get camera rotations from poses
        im_poses = self.get_im_poses()
        rotations = im_poses[:, :3, :3]

        n_poses = min(len(rotations), len(gps_heading_changes) + 2)
        if n_poses < 3:
            return 0.0

        # Compute predicted heading changes from rotation matrices
        pred_heading_changes = []
        for i in range(2, n_poses):
            R_prev = rotations[i-1]
            R_curr = rotations[i]
            R_rel = R_curr @ R_prev.T

            # Extract yaw angle from relative rotation (rotation around Z-axis)
            pred_yaw = torch.atan2(R_rel[1, 0], R_rel[0, 0])
            pred_heading_changes.append(pred_yaw)

        if len(pred_heading_changes) == 0:
            return 0.0

        pred_heading_changes = torch.stack(pred_heading_changes)

        # Compute angular difference loss (handle wraparound)
        n_compare = min(len(pred_heading_changes), len(gps_heading_changes))
        angular_diff = pred_heading_changes[:n_compare] - gps_heading_changes[:n_compare]

        # Normalize to [-pi, pi]
        angular_diff = torch.atan2(torch.sin(angular_diff), torch.cos(angular_diff))
        loss = torch.abs(angular_diff).mean()

        return loss

    def forward(self):
        """Forward pass with GPS constraints added to the alignment loss."""
        # Get base alignment loss
        base_loss = super().forward()

        # Add GPS constraints if available
        gps_loss = 0.0
        if self.gps_constraints is not None:
            # Displacement ratio loss (scale-invariant)
            if self.gps_weight > 0:
                disp_loss = self._compute_displacement_ratio_loss()
                gps_loss = gps_loss + self.gps_weight * disp_loss

            # Velocity direction loss
            if self.gps_velocity_weight > 0:
                vel_loss = self._compute_velocity_direction_loss()
                gps_loss = gps_loss + self.gps_velocity_weight * vel_loss

            # Heading change loss
            if self.gps_heading_weight > 0:
                head_loss = self._compute_heading_change_loss()
                gps_loss = gps_loss + self.gps_heading_weight * head_loss

        return base_loss + gps_loss
