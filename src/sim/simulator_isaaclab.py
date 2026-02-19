"""
IsaacLab Simulator with an interface matching SimulatorGenesis.

IMPORTANT: Before importing this module, you must launch IsaacSim:
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

Then import and use this module.
"""

import numpy as np
import os
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.utils import convert_dict_to_backend

from configs import robot_config


def mat2quat_wxyz(mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion in (w, x, y, z) format."""
    r = R.from_matrix(mat)
    quat_xyzw = r.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


class IsaacLabCamera:
    """
    Wrapper around IsaacLab Camera to provide a Genesis-like camera interface.
    Supports set_pose(T) and render() returning an RGB image.
    """

    def __init__(self, sim, camera, cam_index=0):
        self.sim = sim
        self.camera = camera
        self.cam_index = cam_index
        self._pos = None
        self._target = None

    def set_pose(self, T):
        """Set camera pose from a 4x4 extrinsic matrix (camera-to-world)."""
        pos = T[:3, 3]
        target = pos - T[:3, 2]
        self._pos = pos
        self._target = target
        positions = torch.tensor([pos], dtype=torch.float32, device=self.sim.device)
        targets = torch.tensor([target], dtype=torch.float32, device=self.sim.device)
        self.camera.set_world_poses_from_view(positions, targets)

    def set_world_pose(self, pos, target):
        """Set camera pose from position and look-at target."""
        self._pos = pos
        self._target = target
        # Defer actual pose setting until the simulation is initialized (build).
        # Calling set_world_poses_from_view before scene.reset() causes
        # '_ALL_INDICES' AttributeError.

    def _apply_deferred_pose(self):
        """Apply the stored pose after the camera has been fully initialized."""
        if self._pos is not None and self._target is not None:
            positions = torch.tensor([self._pos], dtype=torch.float32, device=self.sim.device)
            targets = torch.tensor([self._target], dtype=torch.float32, device=self.sim.device)
            self.camera.set_world_poses_from_view(positions, targets)

    def render(self, rgb=True, depth=False, segmentation=False):
        """Update and return camera data. Returns RGB as numpy HxWx3 uint8 by default."""
        self.camera.update(dt=self.sim.get_physics_dt())
        outputs = {}
        if rgb and "rgb" in self.camera.data.output:
            rgb_data = self.camera.data.output["rgb"][self.cam_index].cpu().numpy()
            if rgb_data.shape[-1] == 4:
                rgb_data = rgb_data[..., :3]
            outputs["rgb"] = rgb_data
        if depth and "distance_to_image_plane" in self.camera.data.output:
            outputs["depth"] = self.camera.data.output["distance_to_image_plane"][self.cam_index].cpu().numpy()
        if segmentation and "semantic_segmentation" in self.camera.data.output:
            seg_data = self.camera.data.output["semantic_segmentation"][self.cam_index].cpu().numpy()
            # semantic_segmentation may be HxWx4 (RGBA) or HxWx1 or HxW — extract single channel
            if seg_data.ndim == 3 and seg_data.shape[-1] > 1:
                seg_data = seg_data[..., 0]
            elif seg_data.ndim == 3 and seg_data.shape[-1] == 1:
                seg_data = seg_data[..., 0]
            outputs["segmentation"] = seg_data
        if rgb and not depth and not segmentation:
            return outputs.get("rgb", None)
        return outputs


class SimulatorIsaacLab:
    """
    IsaacLab simulator with the same interface as SimulatorGenesis.

    Provides:
        - start_sim(default, special_light)
        - set_scene(default, special_light)
        - add_robot(default)
        - asset_addtion(asset_path, pos, quat, scale, physics, fixed)
        - get_asset_ID()
        - set_camera() / add_camera(pos, lookat, res, fov, GUI)
        - step()
        - build() — call after adding all entities, before simulation
        - scene  — the SimulationContext (has .step(), .device, etc.)
        - robot  — the Articulation object
        - assets_entity — dict of name -> RigidObject
        - asset_ID — dict of scale -> (RigidObject, mass, friction)
    """

    def __init__(self, robot_name, envs, add_robot=None, device=None, show_viewer=False, seed=None):
        self.robot_name = robot_name
        self.robot = None
        self.envs = envs
        self.seed = seed
        self.asset_path = None
        self.device = device or "cuda:0"
        self.show_viewer = show_viewer
        self.robot_addition = add_robot
        self.asset_ID = {}
        self.assets_entity = {}
        self.scene = None
        self._cameras = []
        self._rigid_objects = []
        self._built = False

    def get_asset_ID(self):
        return self.asset_ID

    # ------------------------------------------------------------------ #
    #  Scene setup
    # ------------------------------------------------------------------ #

    def start_sim(self, default=False, special_light=False):
        """Initialize simulation context and set up the scene."""
        sim_cfg = sim_utils.SimulationCfg(device=self.device)
        self.scene = sim_utils.SimulationContext(sim_cfg)
        self.set_scene(default=default, special_light=special_light)
        if self.robot_addition:
            self.add_robot(default=default)

    def set_scene(self, default=False, special_light=False):
        """Set up ground plane and lighting."""
        # Clean up any existing prims from a previous run
        from pxr import Usd
        stage = self.scene.stage
        for path in ["/World/Objects", "/World/Light", "/World/defaultGroundPlane", "/World/FloorPad"]:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                stage.RemovePrim(path)
        # Also remove old cameras and robot if present
        world_prim = stage.GetPrimAtPath("/World")
        if world_prim.IsValid():
            for prim in world_prim.GetChildren():
                name = prim.GetName()
                if name.startswith("CamOrigin_"):
                    stage.RemovePrim(prim.GetPath())

        prim_utils.create_prim("/World/Objects", "Xform")

        if special_light:
            cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        else:
            cfg = sim_utils.DistantLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0))
        cfg.func("/World/Light", cfg)

        if default:
            cfg = sim_utils.GroundPlaneCfg()
            cfg.func("/World/defaultGroundPlane", cfg, translation=(0, 0, 0.88))
            # Thin invisible collision pad on the ground to stop objects sinking
            pad_cfg = sim_utils.CuboidCfg(
                size=(10.0, 10.0, 0.01),
                visible=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
            pad_cfg.func("/World/FloorPad", pad_cfg, translation=(0, 0, 0.88 + 0.005))
        else:
            cfg = sim_utils.GroundPlaneCfg(visible=False)
            cfg.func("/World/defaultGroundPlane", cfg)
            pad_cfg = sim_utils.CuboidCfg(
                size=(10.0, 10.0, 0.01),
                visible=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
            pad_cfg.func("/World/FloorPad", pad_cfg, translation=(0, 0, 0.005))

        return self.scene

    # ------------------------------------------------------------------ #
    #  Asset management
    # ------------------------------------------------------------------ #

    def asset_addtion(self, asset_path, pos=None, quat=None, scale=None,
                      physics=None, fixed=False):
        """
        Add a rigid object asset to the scene.
        Interface matches SimulatorGenesis.asset_addtion().
        Returns the quaternion (matching Genesis convention).
        """
        if pos is None:
            pos = (1, 0.5, 1.0)
        if quat is None:
            quat = (0.717, 0.717, 0, 0)
        if scale is None:
            scale = 0.12

        name = (physics["object_name"]
                if physics and "object_name" in physics
                else Path(asset_path).stem)

        prim_name = name.replace(" ", "_").replace("/", "_")
        prim_path = f"/World/Objects/{prim_name}"

        common_props = {
            "scale": (scale,) * 3,
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
            "semantic_tags": [("class", "UsdObject")],
        }
        if physics and "mass" in physics:
            common_props["mass_props"] = sim_utils.MassPropertiesCfg(mass=physics["mass"])

        spawn_cfg = sim_utils.UsdFileCfg(usd_path=asset_path, **common_props)

        # add offset
        pos = list(pos)
        pos[2] += 0.05

        obj_cfg = RigidObjectCfg(
            prim_path=prim_path,
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=quat),
        )
        asset = RigidObject(cfg=obj_cfg)
        self._rigid_objects.append(asset)

        mass = physics["mass"] if physics and "mass" in physics else None
        friction = physics["friction"] if physics and "friction" in physics else None
        self.asset_ID[scale] = (asset, mass, friction)

        if name in self.assets_entity:
            name += f"_{len(self.assets_entity)}"
        self.assets_entity[name] = asset

        return quat

    # ------------------------------------------------------------------ #
    #  Robot
    # ------------------------------------------------------------------ #

    def add_robot(self, default=False):
        """Load and add a robot articulation to the scene."""
        cfg = robot_config[self.robot_name]
        robot_path = cfg["path"]

        position = cfg.get("pos", [0.0, 0.0, 0.0])
        quaternion = cfg.get("quat", [1.0, 0.0, 0.0, 0.0])

        if isinstance(position, dict):
            position = (position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0))
        if isinstance(quaternion, dict):
            quaternion = (quaternion.get("w", 1.0), quaternion.get("x", 0.0),
                          quaternion.get("y", 0.0), quaternion.get("z", 0.0))

        ext = os.path.splitext(robot_path)[1].lower()

        # IsaacLab needs a USD file; resolve URDF -> USD if possible
        usd_path = robot_path
        if ext == ".urdf":
            usd_candidate = robot_path.replace(".urdf", ".usd")
            if os.path.exists(usd_candidate):
                usd_path = usd_candidate
            elif "widowx" in self.robot_name.lower() or "wx250" in robot_path.lower():
                isaaclab_widowx = os.path.expanduser(
                    "~/user_data/IsaacLab/source/isaaclab_assets/data/Robots/WidowX/widowx.usd"
                )
                if os.path.exists(isaaclab_widowx):
                    usd_path = isaaclab_widowx
                else:
                    print(f"[WARNING] No USD found for {robot_path}, using URDF path directly")

        init_state_kwargs = {}
        if default:
            init_state_kwargs["pos"] = position
            init_state_kwargs["rot"] = quaternion

        # Per-joint actuator config (from working reference init_usd_scene.py)
        if "widowx" in self.robot_name.lower() or "wx250" in robot_path.lower():
            actuators = {
                'waist': ImplicitActuatorCfg(joint_names_expr=['waist'], stiffness=2595, damping=160),
                'shoulder': ImplicitActuatorCfg(joint_names_expr=['shoulder'], stiffness=1855, damping=398),
                'elbow': ImplicitActuatorCfg(joint_names_expr=['elbow'], stiffness=8536, damping=246),
                'forearm_roll': ImplicitActuatorCfg(joint_names_expr=['forearm_roll'], stiffness=8818, damping=375),
                'wrist_angle': ImplicitActuatorCfg(joint_names_expr=['wrist_angle'], stiffness=4442, damping=879),
                'wrist_rotate': ImplicitActuatorCfg(joint_names_expr=['wrist_rotate'], stiffness=10000, damping=541),
                'left_finger': ImplicitActuatorCfg(joint_names_expr=['left_finger'], stiffness=6771, damping=835),
                'right_finger': ImplicitActuatorCfg(joint_names_expr=['right_finger'], stiffness=8413, damping=736),
            }
            init_state_kwargs["joint_pos"] = {
                "left_finger": 0.02,
                "right_finger": 0.02,
            }
        else:
            actuators = {
                "all_joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=400.0,
                    damping=40.0,
                ),
            }

        robot_cfg = ArticulationCfg(
            prim_path=f"/World/{self.robot_name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                scale=(1.0, 1.0, 1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    fix_root_link=True,
                ),
                semantic_tags=[("class", "Robot")],
            ),
            init_state=ArticulationCfg.InitialStateCfg(**init_state_kwargs),
            actuators=actuators,
        )

        self.robot = Articulation(robot_cfg)
        return self.robot

    # ------------------------------------------------------------------ #
    #  Camera
    # ------------------------------------------------------------------ #

    def add_camera(self, pos=(2.5, -0.15, 2.42), lookat=(0.5, 0.5, 0.1),
                   res=(1280, 720), fov=30, GUI=False):
        """
        Add a camera to the scene. Mirrors Genesis scene.add_camera() interface.
        Returns an IsaacLabCamera wrapper.
        """
        cam_idx = len(self._cameras)
        prim_path_parent = f"/World/CamOrigin_{cam_idx:02d}"
        prim_utils.create_prim(prim_path_parent, "Xform")

        W, H = res
        camera_cfg = CameraCfg(
            prim_path=f"{prim_path_parent}/CameraSensor",
            update_period=0,
            height=H,
            width=W,
            data_types=[
                "rgb",
                "distance_to_image_plane",
                "semantic_segmentation",
            ],
            colorize_semantic_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=fov,
                focus_distance=400.0,
                vertical_aperture=20.955,
                horizontal_aperture=20.955 * (W / H),
                clipping_range=(0.1, 1.0e3),
            ),
        )
        camera = Camera(cfg=camera_cfg)
        wrapper = IsaacLabCamera(self.scene, camera, cam_index=0)
        wrapper.set_world_pose(pos, lookat)
        self._cameras.append(wrapper)
        return wrapper

    def set_camera(self, pos=(2.5, -0.15, 2.42), lookat=(0.5, 0.5, 0.1),
                   res=(1280, 720), fov=30):
        """Convenience method matching Genesis set_camera interface."""
        return self.add_camera(pos=pos, lookat=lookat, res=res, fov=fov)

    # ------------------------------------------------------------------ #
    #  Build & simulation control
    # ------------------------------------------------------------------ #

    def build(self):
        """
        Finalize the scene. Must be called after all entities/cameras are added.
        Equivalent to Genesis scene.build().
        """
        self.scene.reset()
        if self.robot is not None:
            self.robot.update(self.scene.get_physics_dt())
        for obj in self._rigid_objects:
            obj.update(self.scene.get_physics_dt())
        # Apply deferred camera poses now that the simulation is initialized
        for cam in self._cameras:
            cam._apply_deferred_pose()
            cam.camera.update(dt=self.scene.get_physics_dt())
        self._built = True

    def step(self):
        """Advance the simulation by one step and update all managed entities."""
        if self.robot is not None:
            self.robot.write_data_to_sim()
        self.scene.step()
        if self.robot is not None:
            self.robot.update(self.scene.get_physics_dt())
        for obj in self._rigid_objects:
            obj.update(self.scene.get_physics_dt())

    # ------------------------------------------------------------------ #
    #  Utility methods (matching Genesis interface)
    # ------------------------------------------------------------------ #

    def gs_transform_by_quat(self, pos, quat):
        """Transform position vector using quaternion rotation."""
        qw, qx, qy, qz = quat.unbind(-1)
        rot_matrix = torch.stack([
            1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
            2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
            2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2
        ], dim=-1).reshape(*quat.shape[:-1], 3, 3)
        return torch.matmul(rot_matrix, pos.unsqueeze(-1)).squeeze(-1)

    def transform_point_cloud(self, point_cloud, translation, quaternion):
        """Apply rotation and translation to a point cloud."""
        rotated_points = self.gs_transform_by_quat(point_cloud, quaternion)
        return rotated_points + translation

    def control_ee_pose(self, target_pos, target_quat):
        """Move end-effector to target position using Jacobian-based IK."""
        ee_link_name = robot_config[self.robot_name]["ee_link"]
        all_joints = self.robot.joint_names
        arm_indices = [i for i, name in enumerate(all_joints) if "finger" not in name]

        ee_idx = None
        for i, name in enumerate(self.robot.body_names):
            if ee_link_name in name:
                ee_idx = i
                break
        if ee_idx is None:
            ee_idx = self.robot.num_bodies - 1

        full_jacobian = self.robot.root_physx_view.get_jacobians()
        J = full_jacobian[:, ee_idx, :3, arm_indices]

        ee_pos = self.robot.data.body_pos_w[:, ee_idx]
        target = torch.tensor(target_pos, dtype=torch.float32, device=self.scene.device).unsqueeze(0)
        pos_error = target - ee_pos

        delta_q = torch.linalg.pinv(J) @ pos_error.unsqueeze(-1)
        delta_q = delta_q.squeeze(-1) * 0.8

        current_arm = self.robot.data.joint_pos[:, arm_indices]
        next_arm = current_arm + delta_q

        full_command = self.robot.data.joint_pos.clone()
        full_command[:, arm_indices] = next_arm
        self.robot.set_joint_position_target(full_command)

    def visualize_pc(self):
        """Placeholder — point cloud visualization not supported in headless IsaacLab."""
        pass