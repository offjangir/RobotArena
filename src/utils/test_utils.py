

import argparse
import numpy as np
import torch
import numpy as np
import genesis as gs
from transforms3d.euler import euler2quat, quat2euler
import torch
import sapien.core as sapien

import numpy as np
import genesis as gs
import torch
from torch.nn import functional as F

def get_genesis_extrinsics(extrinsics, intrinsics):
    T = np.array(extrinsics)
    estimated_transform = T
    estimated_transform =  np.linalg.inv(T)
    Rotations_correction =  np.array([[1,0,0], [0,-1,0], [0,0,-1]])
    estimated_transform[:3, :3] =  estimated_transform[:3, :3] @ Rotations_correction
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]  # Extract focal lengths from intrinsic matrix
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    W, H = int(2 * cx), int(2 * cy) 
    
    fov_x = 2 * np.arctan(W / (2 * fx)) * (180 / np.pi)
    fov_y = 2 * np.arctan(H / (2 * fy)) * (180 / np.pi)
    
    return estimated_transform, fov_y, W, H


def reproject_to_plane(X_world ,K, T_world2cam, z_plane):
    # ---------- Cast inputs ----------
    X_world = np.asarray(X_world).reshape(3)
    K = np.asarray(K).reshape(3, 3)
    T_world2cam = np.asarray(T_world2cam).reshape(4, 4)
    # Split R and t out of the 4×4 for convenience
    R = T_world2cam[:3, :3]
    t = T_world2cam[:3, 3]
    # ---------- Step 1: world point → image pixel ----------
    X_cam = R @ X_world + t
    u_h = K @ X_cam
    u = u_h[:2] / u_h[2]
    # ---------- Step 2: back-project pixel to a camera-frame ray ----------
    u_homo = np.array([u[0], u[1], 1.0])
    ray_cam = np.linalg.inv(K) @ u_homo
    ray_world = R.T @ ray_cam
    # Camera centre in world frame
    cam_center_world = -R.T @ t
    # ---------- Step 3: intersect ray with the z-plane ----------
    if abs(ray_world[2]) < 1e-12:
        raise ValueError("Ray is parallel to the z-plane; intersection undefined.")
    t_scalar = (z_plane - cam_center_world[2]) / ray_world[2]
    X_new = cam_center_world + t_scalar * ray_world

    Z_orig = (R @ X_world + t)[2]
    Z_new  = (R @ X_new  + t)[2]
    scale_factor = Z_new / Z_orig
    return X_new, scale_factor 

def perturb_pos_lookat_rigid(pos, lookat, distance_cm=5):
    """
    Perturb both the camera position and lookat point in 6 cardinal directions in world coordinates.
    The world coordinate system is assumed to be: X = right, Y = forward, Z = up.
    
    Args:
        pos (np.ndarray): (3,) Camera position.
        lookat (np.ndarray): (3,) Look-at target.
        distance_cm (float): Perturbation distance in cm.

    Returns:
        dict: direction -> (new_pos, new_lookat)
    """
    print(f"perturb_pos_lookat_rigid: pos {pos}, lookat {lookat}, distance_cm {distance_cm}")
    
    distance = distance_cm / 100.0  # Convert to meters
    directions = {
        'up':       np.array([0.0, 0.0, distance]),   # Z is up
        'down':     np.array([0.0, 0.0, -distance]),  # Negative Z is down
        'forward':    np.array([distance, 0.0, 0.0]),   # X is right
        'backward':     np.array([-distance, 0.0, 0.0]),  # Negative X is left
        'right':  np.array([0.0, distance, 0.0]),   # Y is forward
        'left': np.array([0.0, -distance, 0.0]),  # Negative Y is backward
    }

    perturb = {
        direction: (pos + delta, lookat + delta)
        for direction, delta in directions.items()
    }
    print(f"perturb_pos_lookat_rigid: perturb {perturb}")

    return perturb


def find_link_indices(
    robot,
    names,
    global_idx:bool=False,
    return_link_names:bool=False
):
    if not names:
        if return_link_names:
            return [], []
        return []
    link_indices = list()
    link_names = list()
    for link in robot.links:
        flag = False
        for name in names:
            if name in link.name:
                flag = True
        if flag:
            link_names.append(link.name)
            if global_idx:
                link_indices.append(link.idx)
            else:
                link_indices.append(link.idx - robot.link_start)
    if return_link_names:
        return link_indices, link_names
    return link_indices

def is_grasping_two_finger(
    robot_entity,
    obj_entity,
    left_finger_idx,
    right_finger_idx,
    max_angle=10,
    min_force=0.05,
    max_plane_angle=30,
    obj_link_idx=None,
    **kwargs
):
    contacts = robot_entity.get_contacts(with_entity=obj_entity)
    if contacts['force_a'].shape[0] == 0:
        return False
    left_forces = torch.zeros(3, device=gs.device, dtype=gs.tc_float)
    right_forces = torch.zeros(3, device=gs.device, dtype=gs.tc_float)
    if obj_link_idx is None:
        left_forces  += torch.sum(contacts["force_a"] * ((contacts['link_a'] == left_finger_idx)[...,None]), dim=0)
        left_forces  += torch.sum(contacts["force_b"] * ((contacts['link_b'] == left_finger_idx)[...,None]), dim=0)
        right_forces += torch.sum(contacts["force_a"] * ((contacts['link_a'] == right_finger_idx)[...,None]), dim=0)
        right_forces += torch.sum(contacts["force_b"] * ((contacts['link_b'] == right_finger_idx)[...,None]), dim=0)
    else:
        left_forces  += torch.sum(contacts["force_a"] * ((contacts['link_a'] == left_finger_idx)  & (contacts['link_b'] == obj_link_idx))[...,None], dim=0)
        left_forces  += torch.sum(contacts["force_b"] * ((contacts['link_b'] == left_finger_idx)  & (contacts['link_a'] == obj_link_idx))[...,None], dim=0)
        right_forces += torch.sum(contacts["force_a"] * ((contacts['link_a'] == right_finger_idx) & (contacts['link_b'] == obj_link_idx))[...,None], dim=0)
        right_forces += torch.sum(contacts["force_b"] * ((contacts['link_b'] == right_finger_idx) & (contacts['link_a'] == obj_link_idx))[...,None], dim=0)

    link_quats = robot_entity.get_links_quat()
    # since the fingers are in the same plane we can use either left or right to define a gripper vec
    gripper_vec = quat_to_unit_vec(link_quats[left_finger_idx - robot_entity.link_start].unsqueeze(0))

    # Define the gripper plane normal (cross product of finger directions)
    link_pos = robot_entity.get_links_pos().unsqueeze(0)
    plane_normal = torch.cross(
        gripper_vec, 
        link_pos[:, left_finger_idx - robot_entity.link_start] - \
            link_pos[:, right_finger_idx - robot_entity.link_start],
        dim=-1
    )
    # Forces should be nearly perpendicular to the normal (cosine close to 0)
    left_cos_plane = F.cosine_similarity(plane_normal, left_forces.unsqueeze(0), dim=1)[0]
    right_cos_plane = F.cosine_similarity(plane_normal, right_forces.unsqueeze(0), dim=1)[0]

    cos_threshold = np.cos(np.deg2rad(90-max_plane_angle))
    forces_in_plane = (
        (-cos_threshold < left_cos_plane) & (left_cos_plane < cos_threshold) &
        (-cos_threshold < right_cos_plane) & (right_cos_plane < cos_threshold)
    )

    # calculate the angle between the force and the link
    left_ang = F.cosine_similarity(gripper_vec, left_forces.unsqueeze(0), dim=1)[0]
    right_ang = F.cosine_similarity(gripper_vec, right_forces.unsqueeze(0), dim=1)[0]
    has_left_contact = (-np.cos(np.deg2rad(max_angle)) < left_ang) & (left_ang < np.cos(np.deg2rad(max_angle))) & (left_forces.norm() > min_force)
    had_right_contact = (-np.cos(np.deg2rad(max_angle)) < right_ang) & (right_ang < np.cos(np.deg2rad(max_angle))) & (right_forces.norm() > min_force)
    
    grasping = has_left_contact & had_right_contact & forces_in_plane
    return grasping



def quat_to_unit_vec(q_batch):
    """
    Converts a batch of quaternions to unit direction vectors in 3D.
    
    Parameters
    ----------
        q_batch (numpy.ndarray | torch.Tensor)
            An (N, 4) array of N quaternions (w, x, y, z).
        
    Returns
    -------
        numpy.ndarray | torch.Tensor: 
            An (N, 3) array of unit vectors representing the direction of each quaternion.
    """
    # Normalize the quaternions
    if isinstance(q_batch, torch.Tensor):
        # Normalize the quaternions
        norms = torch.norm(q_batch, dim=1, keepdim=True)
        q_batch = q_batch / norms
        # Extract components
        w, x, y, z = q_batch[:, 0], q_batch[:, 1], q_batch[:, 2], q_batch[:, 3]
        # Compute the direction vector (forward Z-axis)
        v_z = torch.stack([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x**2 + y**2)
        ], dim=1)
        # Normalize the resulting direction vectors
        v_z = v_z / torch.norm(v_z, dim=1, keepdim=True)
    elif isinstance(q_batch, np.ndarray):
        norms = np.linalg.norm(q_batch, axis=1, keepdims=True)
        q_batch = q_batch / norms
        
        # Extract components
        w, x, y, z = q_batch[:, 0], q_batch[:, 1], q_batch[:, 2], q_batch[:, 3]
        
        # Compute the direction vector (forward Z-axis)
        v_z = np.stack([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x**2 + y**2)
        ], axis=1)
        # Normalize the resulting direction vectors
        v_z /= np.linalg.norm(v_z, axis=1, keepdims=True)
    return v_z


def apply_safety_limits(
    command: torch.Tensor,
    q_measured: torch.Tensor,
    v_measured: torch.Tensor,
    motors_soft_position_lower: torch.Tensor,
    motors_soft_position_upper: torch.Tensor,
    motors_velocity_limit: torch.Tensor,
    motors_effort_limit: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
) -> torch.Tensor:
    """Clip the command torque to ensure safe operation.

    A velocity limit v+ is enforced by bounding the commanded effort such that
    no effort can be applied to push the joint beyond the velocity limit, and
    a damping effort is applied if the joint is moving at a velocity beyond
    the limit, ie -kd * (v - v+).

    When the joint is near the soft limits x+/-, the velocities are bounded to
    keep the position from crossing the soft limits. The k_position term
    determines the scale of the bound on velocity, ie v+/- = -kp * (x - x+/-).
    These bounds on velocity are the ones determining the bounds on effort.

    The output command would never exceed the maximum effort, not even if
    needed to enforce safe operation.

    It acts on each actuator independently and only activate close to the
    position or velocity limits. Basically, the idea to the avoid moving faster
    when some prescribed velocity limit or exceeding soft position bounds by
    forcing the command torque to act against it. Still, it may not be enough
    to prevent such issue in practice as the command torque is bounded.

    Parameters
    ----------
    command:
        torque command input.
    q_measured:
        current position of the actuators.
    v_measured:
        current velocity of the actuators.
    kp: 
        scale of the velocity bound triggered by position limits.
    kd: 
        scale of the effort bound triggered by velocity limits.
    motors_soft_position_lower:
        soft lower position limit of the actuators.
    motors_soft_position_upper:
        soft upper position limit of the actuators.
    motors_velocity_limit:
        maximum velocity of the actuators.
    motors_effort_limit:
        maximum effort that the actuators can output.
        The command torque cannot exceed this limits,
            not even if needed to enforce safe operation.
    """
    # Computes velocity bounds based on margin from soft joint limit if any
    # print(-kp * (q_measured - motors_soft_position_lower))
    safe_velocity_lower = motors_velocity_limit * torch.clip(-kp * (q_measured - motors_soft_position_lower), -1.0, 1.0)
    safe_velocity_upper = motors_velocity_limit *  torch.clip(-kp * (q_measured - motors_soft_position_upper), -1.0, 1.0)

    # Computes effort bounds based on velocity and effort bounds
    safe_effort_lower = motors_effort_limit * torch.clip(-kd * (v_measured - safe_velocity_lower), -1.0, 1.0)
    safe_effort_upper = motors_effort_limit * torch.clip(-kd * (v_measured - safe_velocity_upper), -1.0, 1.0)

    # Clip command according to safe effort bounds
    if (command < safe_effort_lower).any() or (safe_effort_upper < command).any():
        print(command, safe_effort_lower, safe_effort_upper)
    res = torch.clip(command, safe_effort_lower, safe_effort_upper)
    return res
