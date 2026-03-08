# GPS coordinate utilities for CUT3R
# Provides scale-invariant GPS constraints for camera path optimization
#
# --------------------------------------------------------

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


# Earth radius in meters (for local coordinate conversion)
EARTH_RADIUS = 6371000.0


def lat_lon_to_local_xy(
    lat: float,
    lon: float,
    ref_lat: float,
    ref_lon: float
) -> Tuple[float, float]:
    """
    Convert GPS coordinates to local XY coordinates (meters) using Equirectangular projection.
    Suitable for small areas (< 10km).

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        ref_lat: Reference latitude (origin) in decimal degrees
        ref_lon: Reference longitude (origin) in decimal degrees

    Returns:
        (x, y): Local coordinates in meters, where x is East and y is North
    """
    x = EARTH_RADIUS * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    y = EARTH_RADIUS * np.radians(lat - ref_lat)
    return x, y


def gps_data_to_local_xy(
    gps_data: Dict[int, dict],
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, List[int]]:
    """
    Convert GPS data dictionary to local XY coordinates.

    Args:
        gps_data: Dictionary mapping frame_cnt to GPS info with 'latitude' and 'longitude'
        device: PyTorch device for output tensors

    Returns:
        xy_coords: Tensor of shape (N, 2) with local XY coordinates in meters
        frame_ids: List of frame IDs corresponding to each coordinate
    """
    if not gps_data:
        return None, []

    frame_ids = sorted(gps_data.keys())

    # Use first frame as reference
    ref_lat = gps_data[frame_ids[0]]['latitude']
    ref_lon = gps_data[frame_ids[0]]['longitude']

    xy_list = []
    for frame_id in frame_ids:
        lat = gps_data[frame_id]['latitude']
        lon = gps_data[frame_id]['longitude']
        x, y = lat_lon_to_local_xy(lat, lon, ref_lat, ref_lon)
        xy_list.append([x, y])

    xy_coords = torch.tensor(xy_list, dtype=torch.float32, device=device)
    return xy_coords, frame_ids


def compute_gps_displacement_ratios(
    xy_coords: torch.Tensor,
    min_total_displacement: float = 0.1
) -> torch.Tensor:
    """
    Compute scale-invariant displacement ratios from GPS coordinates.

    Each ratio is: displacement[i] / total_displacement
    This is invariant to scale (zoom level) but preserves relative motion.

    Args:
        xy_coords: Tensor of shape (N, 2) with XY coordinates
        min_total_displacement: Minimum total displacement to avoid division by zero

    Returns:
        ratios: Tensor of shape (N-1,) with displacement ratios
    """
    if xy_coords is None or len(xy_coords) < 2:
        return None

    # Compute displacements between consecutive frames
    displacements = torch.norm(xy_coords[1:] - xy_coords[:-1], dim=1)

    # Compute total displacement
    total_disp = displacements.sum()

    if total_disp < min_total_displacement:
        # Drone is hovering - return uniform ratios
        n = len(displacements)
        return torch.ones(n, device=xy_coords.device) / n

    # Compute ratios
    ratios = displacements / total_disp
    return ratios


def compute_gps_velocity_directions(
    xy_coords: torch.Tensor,
    min_displacement: float = 0.01
) -> torch.Tensor:
    """
    Compute normalized velocity direction vectors from GPS coordinates.

    Args:
        xy_coords: Tensor of shape (N, 2) with XY coordinates
        min_displacement: Minimum displacement to compute valid direction

    Returns:
        directions: Tensor of shape (N-1, 2) with normalized direction vectors
                   Returns zero vector for stationary segments
    """
    if xy_coords is None or len(xy_coords) < 2:
        return None

    # Compute velocity vectors
    velocities = xy_coords[1:] - xy_coords[:-1]

    # Compute magnitudes
    magnitudes = torch.norm(velocities, dim=1, keepdim=True)

    # Normalize, handling stationary cases
    directions = torch.where(
        magnitudes > min_displacement,
        velocities / (magnitudes + 1e-8),
        torch.zeros_like(velocities)
    )

    return directions


def compute_gps_heading_changes(
    xy_coords: torch.Tensor,
    min_displacement: float = 0.01
) -> torch.Tensor:
    """
    Compute heading (yaw) changes between consecutive segments.

    Args:
        xy_coords: Tensor of shape (N, 2) with XY coordinates
        min_displacement: Minimum displacement to compute valid heading

    Returns:
        heading_changes: Tensor of shape (N-2,) with heading changes in radians
                        Returns 0 for stationary segments
    """
    if xy_coords is None or len(xy_coords) < 3:
        return None

    # Compute velocity vectors
    velocities = xy_coords[1:] - xy_coords[:-1]  # (N-1, 2)

    # Compute headings (angle from x-axis)
    headings = torch.atan2(velocities[:, 1], velocities[:, 0])  # (N-1,)

    # Compute heading changes
    heading_changes = headings[1:] - headings[:-1]  # (N-2,)

    # Normalize to [-pi, pi]
    heading_changes = torch.atan2(
        torch.sin(heading_changes),
        torch.cos(heading_changes)
    )

    # Mask out changes where displacement is too small
    magnitudes = torch.norm(velocities, dim=1)
    valid_mask = (magnitudes[:-1] > min_displacement) & (magnitudes[1:] > min_displacement)
    heading_changes = torch.where(
        valid_mask,
        heading_changes,
        torch.zeros_like(heading_changes)
    )

    return heading_changes


def compute_gps_constraints(
    gps_data: Dict[int, dict],
    device: Optional[torch.device] = None,
    min_displacement: float = 0.1
) -> dict:
    """
    Compute all GPS constraints from GPS data.

    Args:
        gps_data: Dictionary mapping frame_cnt to GPS info
        device: PyTorch device for output tensors
        min_displacement: Minimum displacement threshold (meters)

    Returns:
        Dictionary containing:
        - 'xy_coords': Local XY coordinates (N, 2)
        - 'frame_ids': List of frame IDs
        - 'displacement_ratios': Scale-invariant ratios (N-1,)
        - 'velocity_directions': Normalized directions (N-1, 2)
        - 'heading_changes': Angular changes (N-2,)
        - 'is_hovering': Boolean indicating if drone is mostly stationary
    """
    if gps_data is None or len(gps_data) < 2:
        return None

    # Convert to local coordinates
    xy_coords, frame_ids = gps_data_to_local_xy(gps_data, device)

    if xy_coords is None:
        return None

    # Compute total displacement to detect hovering
    displacements = torch.norm(xy_coords[1:] - xy_coords[:-1], dim=1)
    total_disp = displacements.sum().item()
    is_hovering = total_disp < min_displacement * len(frame_ids)

    # Compute all constraints
    constraints = {
        'xy_coords': xy_coords,
        'frame_ids': frame_ids,
        'displacement_ratios': compute_gps_displacement_ratios(xy_coords, min_displacement),
        'velocity_directions': compute_gps_velocity_directions(xy_coords, min_displacement / 10),
        'heading_changes': compute_gps_heading_changes(xy_coords, min_displacement / 10),
        'is_hovering': is_hovering,
        'total_displacement': total_disp,
    }

    return constraints


def get_gimbal_yaw_constraints(
    gps_data: Dict[int, dict],
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Extract gimbal yaw values for use as rotation constraints when hovering.

    Args:
        gps_data: Dictionary mapping frame_cnt to GPS info with 'yaw' field
        device: PyTorch device for output tensors

    Returns:
        yaw_values: Tensor of shape (N,) with gimbal yaw in radians
    """
    if gps_data is None:
        return None

    frame_ids = sorted(gps_data.keys())
    yaw_values = []

    for frame_id in frame_ids:
        yaw_deg = gps_data[frame_id].get('yaw', 0.0)
        yaw_rad = np.radians(yaw_deg)
        yaw_values.append(yaw_rad)

    return torch.tensor(yaw_values, dtype=torch.float32, device=device)


def compute_gimbal_yaw_changes(
    gps_data: Dict[int, dict],
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute gimbal yaw changes between consecutive frames.
    Useful when GPS shows hovering but gimbal is rotating.

    Args:
        gps_data: Dictionary mapping frame_cnt to GPS info with 'yaw' field
        device: PyTorch device for output tensors

    Returns:
        yaw_changes: Tensor of shape (N-1,) with yaw changes in radians
    """
    yaw_values = get_gimbal_yaw_constraints(gps_data, device)

    if yaw_values is None or len(yaw_values) < 2:
        return None

    # Compute changes
    yaw_changes = yaw_values[1:] - yaw_values[:-1]

    # Normalize to [-pi, pi]
    yaw_changes = torch.atan2(
        torch.sin(yaw_changes),
        torch.cos(yaw_changes)
    )

    return yaw_changes
