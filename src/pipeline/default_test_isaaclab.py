"""
default_test_isaaclab.py
========================
IsaacLab counterpart of default_test.py.

IMPORTANT – IsaacSim must be launched **before** any isaaclab imports.
When running from the CLI the typical invocation is:

    ./isaaclab.sh -p src/pipeline/default_test_isaaclab.py \
        --headless --enable_cameras \
        --config configs/default.yaml --vla openvla --port 9010

The AppLauncher block below takes care of this.
"""

# ── Step 0: Launch IsaacSim (must happen before ANY isaaclab import) ──────
import argparse
import sys, os

# Ensure RobotArena root is on sys.path so ``from src.…`` works
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RobotArena default test – IsaacLab backend")
parser.add_argument("--config", type=str, default="configs/default.yaml")
parser.add_argument("--run_all", type=str, default="False")
parser.add_argument("--output_dir", type=str, default="./results")
parser.add_argument("--port", type=int, default=9010)
parser.add_argument("--vla", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Step 1: Normal imports (IsaacLab is now ready) ────────────────────────
import cv2
import yaml
import json
import math
import requests
import collections
import torch
import numpy as np
# Lightweight sapien.Pose replacement (position + quaternion algebra)
class _Pose:
    """Minimal replacement for sapien.Pose with position + quaternion (xyzw)."""
    __slots__ = ('p', 'q')

    def __init__(self, p=None, q=None):
        self.p = np.array(p, dtype=np.float64) if p is not None else np.zeros(3, dtype=np.float64)
        if q is not None:
            self.q = np.array(q, dtype=np.float64)  # xyzw
        else:
            self.q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # identity xyzw

    @staticmethod
    def _qmul(q1, q2):
        """Hamilton product of two quaternions in xyzw layout."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ], dtype=np.float64)

    @staticmethod
    def _qrot(q, v):
        """Rotate vector *v* by quaternion *q* (xyzw)."""
        qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
        q_conj = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)
        return _Pose._qmul(_Pose._qmul(q, qv), q_conj)[:3]

    def inv(self):
        q_conj = np.array([-self.q[0], -self.q[1], -self.q[2], self.q[3]], dtype=np.float64)
        p_inv = -_Pose._qrot(q_conj, self.p)
        return _Pose(p_inv, q_conj)

    def __mul__(self, other):
        q_new = self._qmul(self.q, other.q)
        p_new = self._qrot(self.q, other.p) + self.p
        return _Pose(p_new, q_new)

    def __repr__(self):
        return f"Pose(p={self.p}, q={self.q})"

# Provide a sapien-like namespace so existing code (`sapien.Pose(...)`) works
class _SapienCompat:
    Pose = _Pose
sapien = _SapienCompat()
from types import SimpleNamespace
from scipy.spatial.transform import Rotation as R

# transforms3d replacements using scipy (already imported)
def euler2quat(ai, aj, ak, axes='sxyz'):
    """Euler angles → quaternion (w, x, y, z) — matches transforms3d convention."""
    r = R.from_euler('xyz', [ai, aj, ak])
    q = r.as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz

def quat2euler(q, axes='sxyz'):
    """Quaternion (w, x, y, z) → euler angles — matches transforms3d convention."""
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    r = R.from_quat(q_xyzw)
    return r.as_euler('xyz')

def mat2quat(mat):
    """3x3 rotation matrix → quaternion (w, x, y, z) — matches transforms3d convention."""
    r = R.from_matrix(np.array(mat)[:3, :3])
    q = r.as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz

def quat2mat(q):
    """Quaternion (w, x, y, z) → 3x3 rotation matrix."""
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    return R.from_quat(q_xyzw).as_matrix()

def qmult(q1, q2):
    """Hamilton product of two quaternions in (w, x, y, z) layout."""
    q1_xyzw = np.array([q1[1], q1[2], q1[3], q1[0]])
    q2_xyzw = np.array([q2[1], q2[2], q2[3], q2[0]])
    r = R.from_quat(q1_xyzw) * R.from_quat(q2_xyzw)
    q = r.as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz
import json_numpy as json_np

from src.sim.simulator_isaaclab import SimulatorIsaacLab
from configs import robot_config


# ══════════════════════════════════════════════════════════════════════════
#  Utility helpers (re-used from test_utils but without Genesis dependency)
# ══════════════════════════════════════════════════════════════════════════

def get_extrinsics(extrinsics, intrinsics):
    """Compute camera→world transform, FOV, and resolution from numpy arrays."""
    T = np.array(extrinsics)
    T_inv = np.linalg.inv(T)
    correction = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    T_inv[:3, :3] = T_inv[:3, :3] @ correction

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    W, H = int(2 * cx), int(2 * cy)
    fov_y = 2 * np.arctan(H / (2 * fy)) * (180 / np.pi)
    return T_inv, (fx, fy), fov_y, W, H


def reproject_to_plane(X_world, K, T_world2cam, z_plane):
    """Project a 3-D world point onto z=z_plane through the camera."""
    T = np.array(T_world2cam)
    K = np.array(K)
    X_cam = T[:3, :3] @ np.array(X_world) + T[:3, 3]
    pixel = K @ X_cam
    pixel /= pixel[2]

    K_inv = np.linalg.inv(K)
    T_inv = np.linalg.inv(T)

    ray_cam = K_inv @ pixel
    ray_world = T_inv[:3, :3] @ ray_cam
    origin = T_inv[:3, 3]

    t = (z_plane - origin[2]) / ray_world[2]
    X_new = origin + t * ray_world

    scale_ratio = np.linalg.norm(X_cam) / np.linalg.norm(ray_cam * t)
    return tuple(X_new.tolist()), scale_ratio


def rotate6D_to_euler_xyz(v6: np.ndarray) -> np.ndarray:
    """Convert 6-D rotation representation to Euler xyz angles."""
    v6 = np.asarray(v6, dtype=np.float64)
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)
    return R.from_matrix(rot_mats).as_euler("xyz")


def apply_safety_limits(
    command, q_measured, v_measured,
    motors_soft_position_lower, motors_soft_position_upper,
    motors_velocity_limit, motors_effort_limit,
    kp, kd,
):
    """Clip gripper torque commands to respect position/velocity soft limits."""
    safe_vel_lo = motors_velocity_limit * torch.clip(
        -kp * (q_measured - motors_soft_position_lower), -1.0, 1.0)
    safe_vel_hi = motors_velocity_limit * torch.clip(
        -kp * (q_measured - motors_soft_position_upper), -1.0, 1.0)
    safe_eff_lo = motors_effort_limit * torch.clip(
        -kd * (v_measured - safe_vel_lo), -1.0, 1.0)
    safe_eff_hi = motors_effort_limit * torch.clip(
        -kd * (v_measured - safe_vel_hi), -1.0, 1.0)
    return torch.clip(command, safe_eff_lo, safe_eff_hi)


def mat2quat_wxyz(mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → quaternion (w, x, y, z)."""
    r = R.from_matrix(mat)
    q = r.as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# ══════════════════════════════════════════════════════════════════════════
#  DefaultTestIsaacLab
# ══════════════════════════════════════════════════════════════════════════

class DefaultTestIsaacLab:
    """
    Same pipeline as DefaultTest (Genesis) but drives SimulatorIsaacLab.
    Differences from the Genesis version are commented inline.
    """

    def __init__(self, args, **kwargs):
        self.args = args
        self.action_queue = collections.deque()
        self.pred_action_queue = collections.deque()

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #

    def setup(self):
        self.simulator = SimulatorIsaacLab(
            self.args.robot_args["name"], 1,
            show_viewer=False, add_robot=True,
        )

        self.target_asset = None
        self.desti_asset = None

        if self.args.scene_name == "default2":
            self.simulator.start_sim(default=self.args.default, special_light=True)
        else:
            self.simulator.start_sim(default=self.args.default)

        self.task_description = self.args.task_description
        self.target_image = self.args.background
        self.output_dir = self.args.output_dir
        self.target_rotation = None
        self.target_translation = None
        self.port = self.args.port
        self.use_initial_camera = True

        # ── Camera 0 ──
        self.estimated_transform, self.fs, fov, W, H = get_extrinsics(
            self.args.extrinsics, self.args.intrinsics)
        self.H, self.W = self.target_image.shape[:2]
        self.fov = fov

        if hasattr(self.args, "camera_pos") and hasattr(self.args, "camera_lookat"):
            self.use_initial_camera = False
            self.H = self.args.H
            self.W = self.args.W
            self.fov = self.args.fov
            self.camera_0 = self.simulator.add_camera(
                res=(self.W, self.H),
                pos=self.args.camera_pos,
                lookat=self.args.camera_lookat,
                fov=self.args.fov,
            )
        else:
            self.H, self.W = self.target_image.shape[:2]
            # Use intrinsics-derived focal length for the sensor
            fc = (self.fs[1] * 20.955) / self.H if self.fs is not None else self.fov
            self.camera_0 = self.simulator.add_camera(
                res=(self.W, self.H),
                pos=tuple(self.estimated_transform[:3, 3].tolist()),
                lookat=tuple((self.estimated_transform[:3, 3] - self.estimated_transform[:3, 2]).tolist()),
                fov=fc,
            )

        # ── Load scene objects ──
        for key, attributes in self.args.object_positions.items():
            count = int(key.split("_")[1])
            # IsaacLab needs USD assets.  We look for a USD subdirectory first,
            # falling back to the .glb file (which the spawner can also handle).
            usd_dir = os.path.join(self.args.asset_folder, key + "_usd")
            usd_file = os.path.join(usd_dir, "object.usd") if os.path.isdir(usd_dir) else None
            glb_file = os.path.join(self.args.asset_folder, key + ".glb")
            asset_file = usd_file if usd_file and os.path.exists(usd_file) else glb_file

            if self.args.object_properties:
                physics = self.args.object_properties[count]
            else:
                physics = None

            object_name = (physics["object_name"]
                           if physics and "object_name" in physics else key)

            # Skip emission-only objects (light markers) – IsaacLab has no
            # direct equivalent to Genesis Emission surfaces.
            if "emission" in attributes:
                print(f"[IsaacLab] Skipping emission entity '{object_name}' (no IsaacLab equivalent)")
                if "target" in object_name:
                    self.target_rotation = attributes["rotation"]
                    self.target_translation = attributes["translation"]
                if "destination" in object_name:
                    pass
                continue

            if not os.path.exists(asset_file):
                print(f"Warning: asset file not found – {asset_file}")
                continue

            if self.args.default:
                pos = attributes["translation"]
                scale = attributes["scale"]
                rotation = attributes["rotation"]
                # rotation from init_usd_scene: stored as 3×3 matrix → wxyz quat
                if isinstance(rotation, list) and len(rotation) == 3 and isinstance(rotation[0], list):
                    quat = mat2quat_wxyz(np.array(rotation))
                elif isinstance(rotation, list) and len(rotation) == 3:
                    # Euler angles (Genesis convention)
                    quat = mat2quat_wxyz(R.from_euler("xyz", rotation, degrees=False).as_matrix())
                else:
                    quat = mat2quat_wxyz(np.array(rotation))

                self.simulator.asset_addtion(
                    asset_file, pos=pos, quat=tuple(quat.tolist()),
                    scale=scale, physics=physics,
                )
                if "target" in object_name:
                    self.target_asset = self.simulator.asset_ID[scale][0]
                    self.target_rotation = attributes["rotation"]
                    self.target_translation = attributes["translation"]
                if "destination" in object_name:
                    self.desti_asset = self.simulator.asset_ID[scale][0]
            else:
                pos = attributes["translation"]
                pos_new, scale_factor = reproject_to_plane(
                    pos, self.args.intrinsics, self.args.extrinsics, 0.02)
                scale = attributes["scale"] * scale_factor
                rotation = mat2quat(np.array(attributes["rotation"]))  # xyzw from transforms3d
                # Convert to wxyz for IsaacLab
                quat_wxyz = (rotation[3], rotation[0], rotation[1], rotation[2])

                self.simulator.asset_addtion(
                    asset_file, pos=pos_new, scale=scale,
                    quat=quat_wxyz, physics=physics,
                )
                print(f"Loaded {key} at position {pos_new} with scale {scale}")
                if "target" in object_name:
                    self.target_asset = self.simulator.asset_ID[scale][0]
                    self.target_rotation = attributes["rotation"]
                    self.target_translation = attributes["translation"]
                if "destination" in object_name:
                    self.desti_asset = self.simulator.asset_ID[scale][0]

        # ── Camera 1 (bird's-eye / side) ──
        self.camera_1 = self.simulator.add_camera(
            res=(self.W, self.H),
            pos=self.args.camera_1_args["pos"],
            lookat=self.args.camera_1_args["lookat"],
            fov=self.args.camera_1_args["fov"],
        )

        # ── Build (IsaacLab: reset scene + initialize entities) ──
        print("Building scene …")
        self.simulator.build()

        # Set robot initial joint positions
        print("Setting robot position …")
        init_pos = self.args.robot_args["init_pos"]
        if isinstance(init_pos, torch.Tensor):
            init_pos_tensor = init_pos.clone().unsqueeze(0).to(self.simulator.device)
        else:
            init_pos_tensor = torch.tensor(init_pos, dtype=torch.float32,
                                           device=self.simulator.device).unsqueeze(0)
        self.simulator.robot.set_joint_position_target(init_pos_tensor)
        self.simulator.robot.write_data_to_sim()

        # One step to stabilise
        self.simulator.step()

        # ── Pre-compute useful body indices ──
        self._body_names = list(self.simulator.robot.body_names)
        ee_link_name = robot_config[self.args.robot_args["name"]]["ee_link"]
        self.ee_body_idx = self._body_idx(ee_link_name)
        self.base_body_idx = self._body_idx("base_link")
        self.left_finger_idx = self._body_idx("left_finger")
        self.right_finger_idx = self._body_idx("right_finger")

        # Joint grouping
        all_joints = list(self.simulator.robot.joint_names)
        self.arm_indices = [i for i, n in enumerate(all_joints) if "finger" not in n]
        self.gripper_indices = [i for i, n in enumerate(all_joints) if "finger" in n]

        # ── Transformation properties ──
        self.set_transformation_properties()
        print("Setup complete!")

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _body_idx(self, name):
        """Return body index for a given link/body name (partial match in either direction)."""
        for i, bn in enumerate(self._body_names):
            if name in bn or bn in name:
                return i
        raise ValueError(f"Body '{name}' not found in {self._body_names}")

    def _ee_pos(self):
        """Current end-effector position as numpy (3,)."""
        return self.simulator.robot.data.body_pos_w[0, self.ee_body_idx].cpu().numpy()

    def _ee_quat_wxyz(self):
        """Current end-effector quaternion (wxyz) as numpy (4,)."""
        return self.simulator.robot.data.body_quat_w[0, self.ee_body_idx].cpu().numpy()

    def _ee_quat_xyzw(self):
        """Convert wxyz → xyzw (for sapien Pose)."""
        q = self._ee_quat_wxyz()
        return np.array([q[1], q[2], q[3], q[0]])

    def _body_pos(self, idx):
        return self.simulator.robot.data.body_pos_w[0, idx].cpu().numpy()

    def _body_quat_xyzw(self, idx):
        q = self.simulator.robot.data.body_quat_w[0, idx].cpu().numpy()
        return np.array([q[1], q[2], q[3], q[0]])

    # ------------------------------------------------------------------ #
    #  Transformation bookkeeping (same maths as Genesis version)
    # ------------------------------------------------------------------ #

    def set_transformation_properties(self):
        self.prev_ee_pose_at_world = sapien.Pose(
            self._ee_pos(), self._ee_quat_xyzw())
        self.base_in_world_inv = sapien.Pose(
            self._body_pos(self.base_body_idx),
            self._body_quat_xyzw(self.base_body_idx),
        ).inv()
        self.base_in_world = sapien.Pose(
            self._body_pos(self.base_body_idx),
            self._body_quat_xyzw(self.base_body_idx),
        )
        self.prev_ee_pose_at_base = self.base_in_world_inv * self.prev_ee_pose_at_world
        self.prev_gripper = 0.0

    # ------------------------------------------------------------------ #
    #  Action preparation / parsing (same as Genesis version)
    # ------------------------------------------------------------------ #

    def prepare_action_payload(self, image):
        if self.args.model_name == "xvla":
            return {
                "language_instruction": self.task_description,
                "image0": json_np.dumps(image),
                "proprio": json_np.dumps(self._get_xvla_proprioception()),
                "domain_id": 0,
                "steps": 10,
            }
        elif self.args.model_name == "open_pi_zero":
            return {
                "instruction": self.task_description,
                "image": json_np.dumps(image),
                "proprio": json_np.dumps(self._get_openpi_proprioception()),
            }
        else:
            return {
                "instruction": self.task_description,
                "image": image.tolist(),
            }

    def reset_model(self):
        requests.post(f"http://localhost:{self.port}/reset",
                      json={"instruction": self.args.task_description})

    def set_task(self, task_description):
        requests.post(f"http://localhost:{self.port}/set_task",
                      json={"task_description": task_description})

    # ── Delta vs exact transforms ──

    def transform_actions_delta(self, raw_action, action):
        delta_quat = euler2quat(*raw_action["rotation_delta"])
        delta_pose = sapien.Pose(raw_action["world_vector"], delta_quat)
        cur_ee_at_world = sapien.Pose(self._ee_pos(), self._ee_quat_xyzw())
        cur_ee_at_base = self.base_in_world_inv * cur_ee_at_world
        target = (sapien.Pose(p=cur_ee_at_base.p) * delta_pose *
                  sapien.Pose(p=cur_ee_at_base.p).inv()) * self.prev_ee_pose_at_base
        self.final_pose = self.base_in_world * target
        self.prev_ee_pose_at_base = target
        self.prev_gripper = action["gripper"]
        self.prev_ee_pose_at_world = self.final_pose
        return self.final_pose

    def transform_actions_exact(self, raw_action, action):
        target_quat = euler2quat(*raw_action["rotation_delta"])
        target_in_base = sapien.Pose(raw_action["world_vector"], target_quat)
        self.final_pose = self.base_in_world * target_in_base
        self.prev_ee_pose_at_base = target_in_base
        self.prev_gripper = action["gripper"]
        self.prev_ee_pose_at_world = self.final_pose
        return self.final_pose

    def transform_actions(self, raw_action, action):
        if self.args.model_name == "xvla":
            return self.transform_actions_exact(raw_action, action)
        return self.transform_actions_delta(raw_action, action)

    # ── Action retrieval ──

    def get_action(self):
        if self.action_queue:
            image, rgb = self.get_image()
            raw_action, action = self.action_queue.popleft()
            if self.pred_action_queue:
                pred_action = self.pred_action_queue.popleft()
                self._update_xvla_proprioception(pred_action)
            self.transform_actions(raw_action, action)
            return self.final_pose, action["gripper"], image

        image, rgb = self.get_image()
        res = requests.post(f"http://localhost:{self.port}/act",
                            json=self.prepare_action_payload(image))
        response = res.json()

        if "raw_action" in response:
            raw_action = {k: np.array(v) for k, v in response["raw_action"].items()}
            action = {k: np.array(v) for k, v in response["action"].items()}
        else:
            action_seq = np.array(response["action"], dtype=np.float32)
            action_seq = [self.action_parser(np.array(i, dtype=np.float32)) for i in action_seq]
            self.action_queue.extend(action_seq)
            raw_action, action = self.action_queue.popleft()
            if self.args.model_name == "xvla":
                self._update_xvla_proprioception(np.array(self.pred_action_queue.popleft(), dtype=np.float32))

        self.transform_actions(raw_action, action)
        return self.final_pose, action["gripper"], image

    # ── Image rendering ──

    def get_image(self):
        """Render RGB + segmentation, composite onto background.

        Follows the same approach as init_usd_scene.py post_process_image():
        uses semantic segmentation mask to separate sim foreground (robot +
        objects) from the ground plane, then composites onto the real
        background image.
        """
        out = self.camera_0.render(rgb=True, depth=False, segmentation=True)
        if isinstance(out, dict):
            rgb = out.get("rgb")
            seg = out.get("segmentation")
        else:
            rgb = out
            seg = None

        if seg is not None and self.target_image is not None:
            # rgb from IsaacLab is RGB; convert to BGR for OpenCV ops
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # Mask: any pixel with semantic label > 0 is foreground
            # (Robot tagged "Robot", objects tagged "UsdObject";
            #  background/ground plane has label 0)
            if seg.ndim == 3:
                seg_single = seg[..., 0]
            else:
                seg_single = seg
            mask = (seg_single > 0).astype(np.uint8) * 255

            # Resize background to match rendered frame if needed
            bg = self.target_image
            if bg.shape[:2] != rgb_bgr.shape[:2]:
                bg = cv2.resize(bg, (rgb_bgr.shape[1], rgb_bgr.shape[0]))

            # Composite: foreground from sim, background from real image
            segmented = cv2.bitwise_and(rgb_bgr, rgb_bgr, mask=mask)
            background = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))
            blended = cv2.add(segmented, background)
            return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), rgb
        else:
            # No segmentation available – return raw RGB
            return rgb, rgb

    # ── Proprioception helpers ──

    def _get_openpi_proprioception(self):
        ep = self.prev_ee_pose_at_base
        return np.concatenate([ep.p, ep.q, np.array([self.prev_gripper])])

    def _init_xvla_proprioception(self):
        ep = self.prev_ee_pose_at_base
        p = torch.from_numpy(
            np.concatenate([ep.p, np.array([1, 0, 0, 1, 0, 0, 0])])
        ).to(dtype=torch.float32)
        p = torch.cat([p, torch.zeros_like(p)], dim=-1).numpy().copy()
        self.xvla_proprioception = p.copy()
        return p

    def _update_xvla_proprioception(self, action):
        self.xvla_proprioception[:10] = action[:10]
        return self.xvla_proprioception

    def _get_xvla_proprioception(self):
        if getattr(self, "xvla_proprioception", None) is None:
            self._init_xvla_proprioception()
        return self.xvla_proprioception

    # ── Action parsers ──

    def _xvla_retrieve_raw_action(self, action):
        wrapped = {
            "world_vector": action[:3],
            "rot_axangle": action[3:6],
            "gripper": action[6],
        }
        raw = {
            "world_vector": wrapped["world_vector"],
            "rotation_delta": action[3:6],
            "open_gripper": np.array([1.0 if wrapped["gripper"] > 0 else 0.0]),
        }
        return raw, wrapped

    def _xvla_action_parser(self, action):
        self.pred_action_queue.append(action)
        action_final = np.concatenate([
            action[:3],
            rotate6D_to_euler_xyz(action[3:9]) + np.array([0, math.pi / 2, 0]),
            np.array([1 if action[9] < 0.9 else -1]),
        ])
        return self._xvla_retrieve_raw_action(action_final)

    def _openpi_action_parser(self, action):
        raw = {
            "world_vector": action[:3],
            "rotation_delta": action[3:6],
            "open_gripper": action[6],
        }
        wrapped = {
            "world_vector": action[:3],
            "rot_axangle": action[3:6],
            "gripper": raw["open_gripper"],
        }
        return raw, wrapped

    def action_parser(self, action):
        if self.args.model_name == "xvla":
            return self._xvla_action_parser(action)
        elif self.args.model_name == "open_pi_zero":
            return self._openpi_action_parser(action)
        raise NotImplementedError(f"action_parser not implemented for {self.args.model_name}")

    # ------------------------------------------------------------------ #
    #  Jacobian-based IK (from init_usd_scene.py reference)
    # ------------------------------------------------------------------ #

    def _ik_step(self, target_pos_np, target_quat_wxyz_np, gripper_val):
        """
        Single Jacobian-based IK iteration + set gripper.
        `target_quat_wxyz_np` is currently unused (position-only IK) but
        kept for future orientation control.
        """
        robot = self.simulator.robot
        device = self.simulator.scene.device

        full_jacobian = robot.root_physx_view.get_jacobians()
        J = full_jacobian[:, self.ee_body_idx, :3, self.arm_indices]

        ee_pos = robot.data.body_pos_w[:, self.ee_body_idx]
        target = torch.tensor(target_pos_np, dtype=torch.float32,
                              device=device).unsqueeze(0)
        pos_error = target - ee_pos

        delta_q = torch.linalg.pinv(J) @ pos_error.unsqueeze(-1)
        delta_q = delta_q.squeeze(-1) * 0.8

        current_arm = robot.data.joint_pos[:, self.arm_indices]
        next_arm = current_arm + delta_q

        full_cmd = robot.data.joint_pos.clone()
        full_cmd[:, self.arm_indices] = next_arm

        if gripper_val > 0:
            for gi in self.gripper_indices:
                full_cmd[:, gi] = 0.037
        else:
            for gi in self.gripper_indices:
                full_cmd[:, gi] = 0.015

        robot.set_joint_position_target(full_cmd)

    # ------------------------------------------------------------------ #
    #  Main test loop
    # ------------------------------------------------------------------ #

    def run_default_test(self):
        imx = []
        base_states = []
        world_states = []
        object_states = {}

        print("Running Test on task:", self.args.task_description)
        self.set_task(self.args.task_description)
        self.reset_model()

        for i in range(65):
            if i >= 5:
                pose, gripper, image = self.get_action()

                # Record states
                base_states.append(np.concatenate([
                    self.prev_ee_pose_at_base.p,
                    self.prev_ee_pose_at_base.q,
                    np.atleast_1d(self.prev_gripper),
                ]))
                world_states.append(np.concatenate([
                    self.prev_ee_pose_at_world.p,
                    self.prev_ee_pose_at_world.q,
                    np.atleast_1d(self.prev_gripper),
                ]))

                # Object state logging
                for obj_name, obj in self.simulator.assets_entity.items():
                    if obj_name not in object_states:
                        object_states[obj_name] = []
                    root_pos = obj.data.root_pos_w[0].cpu().numpy()
                    root_quat = obj.data.root_quat_w[0].cpu().numpy()
                    root_vel = obj.data.root_lin_vel_w[0].cpu().numpy()
                    root_ang = obj.data.root_ang_vel_w[0].cpu().numpy()
                    object_states[obj_name].append(
                        np.concatenate([root_pos, root_quat, root_vel, root_ang]))
            else:
                pose = self.prev_ee_pose_at_world
                gripper = 0.0
                image = None

            # ── IK + gripper command ──
            self._ik_step(np.array(pose.p), np.array(pose.q), gripper)

            # Sub-step: run physics multiple times per action
            for _ in range(60):
                self.simulator.step()

            if image is not None:
                imx.append(image)

        # ── Save video ──
        folder = self.output_dir
        os.makedirs(folder, exist_ok=True)
        video_path = os.path.join(
            folder,
            f"test_{self.args.task_description}_{self.args.test_id}.mp4")
        self._save_video(imx, video_path, fps=5)

        # ── Save states ──
        world_states = np.array(world_states)
        base_states = np.array(base_states)
        # Uncomment to persist states:
        # np.savez(os.path.join(folder, f"test_{self.args.task_description}_{self.args.test_id}.npz"),
        #          world=world_states, base=base_states, **object_states)

    @staticmethod
    def _save_video(frames, path, fps=5):
        """Write a list of RGB numpy frames to an mp4 file using OpenCV."""
        if not frames:
            print("No frames to save.")
            return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Saved video ({len(frames)} frames) → {path}")

    # ------------------------------------------------------------------ #
    #  Entry point
    # ------------------------------------------------------------------ #

    def run(self):
        try:
            self.setup()
            self.run_default_test()
        except Exception:
            import traceback
            print(f"Error processing task '{self.args.task_description}' "
                  f"in scene '{self.args.scene_name}':\n{traceback.format_exc()}",
                  file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════
#  CLI entry
# ══════════════════════════════════════════════════════════════════════════

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    robot_args = {
        "name": "WidowX",
        "init_pos": torch.tensor(
            [-0.02014876, 0.04723017, 0.22625704, -0.00307271, 1.365988,
             -0.00168102, 0.037, 0.03699991], device="cuda:0",
        ),
    }
    camera_1_args = {
        "pos": (0.25, 0.25, 1.3),
        "lookat": (0.1, 0.1, 1.0),
        "fov": 80,
    }

    config = load_config(args_cli.config)
    base_folder = config["base_folder"]
    scene_name = config["scene_name"]
    output_folder = args_cli.output_dir
    port = args_cli.port
    model_name = args_cli.vla

    run_default = "default" in output_folder

    if str2bool(args_cli.run_all):
        scene_lists = os.listdir(os.path.join(base_folder, "bridge"))
    else:
        scene_lists = [scene_name] if isinstance(scene_name, str) else scene_name

    print("Scenes to process:", scene_lists)

    for arg in args_cli.__dict__:
        print(f"  {arg}: {args_cli.__dict__[arg]}")

    for sn in scene_lists:
        try:
            if sn.startswith("default"):
                if not run_default:
                    continue
                default = True
            elif sn.startswith("scene"):
                if run_default:
                    continue
                default = False
            else:
                continue

            out_dir = os.path.join(output_folder, "default_test", sn)
            if os.path.exists(out_dir):
                print(f"Skipping {sn} – results already exist.")
                continue

            data_folder = os.path.join(base_folder, "bridge", sn)
            asset_folder = os.path.join(base_folder, "assets", sn)
            background = os.path.join(base_folder, "scene_background", sn, "background.png")
            extrinsics = np.load(os.path.join(data_folder, "extrinsics.npy"))
            intrinsics = np.load(os.path.join(data_folder, "intrinsics.npy"))

            with open(os.path.join(data_folder, "masks", "transformations.json"), "r") as f:
                object_positions = json.load(f)

            phys_path = os.path.join(data_folder, "masks", "physical_properties.json")
            if not os.path.exists(phys_path):
                phys_path = os.path.join(data_folder, "masks", "result.json")
            with open(phys_path, "r") as f:
                physics_properties = json.load(f)

            task_path = os.path.join(data_folder, "lang.txt")
            with open(task_path, "r") as f:
                task_lines = [l.strip() for l in f.readlines()]

            for task_desc in task_lines:
                if "confidence" in task_desc:
                    continue
                for trial in range(1):
                    ns = SimpleNamespace(
                        default=default,
                        robot_args=robot_args,
                        background=cv2.imread(background),
                        task_description=task_desc,
                        camera_1_args=camera_1_args,
                        intrinsics=intrinsics,
                        extrinsics=extrinsics,
                        asset_folder=asset_folder,
                        object_positions=object_positions,
                        object_properties=physics_properties,
                        test_id=trial,
                        port=port,
                        scene_name=sn,
                        output_dir=out_dir,
                        model_name=model_name,
                    )
                    p = DefaultTestIsaacLab(ns)
                    p.run()
        except Exception:
            import traceback
            print(f"Error processing scene '{sn}':\n{traceback.format_exc()}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
    simulation_app.close()
