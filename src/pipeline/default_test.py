
import cv2
import os
import yaml
import argparse
import json
import requests
import torch
import random
import genesis as gs
import itertools
import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
from src.sim.simulator_genesis import SimulatorGenesis
from src.utils.test_utils import get_genesis_extrinsics, find_link_indices, is_grasping_two_finger, apply_safety_limits, rotate6D_to_euler_xyz, reproject_to_plane, resolve_overlap
from transforms3d.axangles import mat2axangle
from transforms3d.euler import euler2axangle, euler2mat, euler2quat, quat2euler, axangle2euler
from transforms3d.quaternions import axangle2quat, mat2quat, quat2axangle, quat2mat,qconjugate 
from transforms3d.quaternions import qmult, mat2quat, quat2mat
from transforms3d.quaternions import rotate_vector
import json_numpy as json_np
import collections
import math
from scipy.spatial.transform import Rotation as R

class DefaultTest:
    """
    Following bridge_demo's approach for object loading and scene setup,
    but using all other methods from default_test.py
    """
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.action_queue = collections.deque()
        self.pred_action_queue = collections.deque()

    def setup(self):
        # Simple simulator setup - following bridge_demo
        self.simulator = SimulatorGenesis(
            self.args.robot_args["name"], 1, show_viewer=False, add_robot=True
        )
        
        # Start simulation - following bridge_demo (no default parameter)
        self.target_asset = None
        self.desti_asset = None
        if self.args.scene_name == "default2":
            self.simulator.start_sim(default = self.args.default, special_light=True)
        else:
            self.simulator.start_sim(default = self.args.default)
        self.task_description = self.args.task_description
        self.target_image = self.args.background
        self.output_dir = self.args.output_dir
        self.target_rotation = None
        self.target_translation = None
        self.port = self.args.port
        self.use_initail_camera = True
        
        # Setup camera using extrinsics
        self.estimated_transform, fov, W, H = get_genesis_extrinsics(self.args.extrinsics, self.args.intrinsics)
        self.H, self.W = self.target_image.shape[:2]
        self.fov = fov
        
        if hasattr(self.args, "camera_pos") and hasattr(self.args, "camera_lookat"):
            self.use_initail_camera = False
            self.H = self.args.H
            self.W = self.args.W
            self.fov = self.args.fov
            self.camera_0 = self.simulator.scene.add_camera(
                res=(self.W, self.H),
                pos = self.args.camera_pos,
                lookat = self.args.camera_lookat,
                fov = self.args.fov,
                GUI=False,
            )
        else:
            self.H, self.W = self.target_image.shape[:2]  # get height and width from image
            self.fov = fov
            self.camera_0 = self.simulator.scene.add_camera(
                res=(self.W, self.H),
                pos = (0, 0, 0),
                lookat = (0, 0, 0),
                fov = self.fov,
                GUI=False,
            )
        
        # Load objects BEFORE building scene - following bridge_demo
        for key, attributes in self.args.object_positions.items():
            count = int(key.split("_")[1])
            asset_file = os.path.join(self.args.asset_folder, key + ".glb")
            
            if self.args.object_properties:
                physics = self.args.object_properties[count]
            else:
                physics = None

            object_name = physics["object_name"] if physics and "object_name" in physics else key
            if "emission" in attributes:
                asset = self.simulator.scene.add_entity(
                    gs.morphs.Box(
                        size = (0.03, 0.03, 0.03),
                        pos  = attributes["translation"],
                            ),
                        surface = gs.surfaces.Emission(
                            emissive=attributes["emission"]
                        )
                )
                self.simulator.assets_entity[object_name] = asset
                if "target" in object_name:
                    self.target_asset = asset
                    self.target_rotation = attributes["rotation"]
                    self.target_translation = attributes["translation"]
                if "destination" in object_name:
                    self.desti_asset = asset
                continue
            
            if self.args.default:
                if os.path.exists(asset_file):  
                    pos = attributes["translation"]
                    scale= attributes["scale"]
                    rotation = attributes["rotation"]
                    asset =  self.simulator.scene.add_entity(
                            morph=gs.morphs.Mesh(
                                file= asset_file,
                                scale = scale,
                                pos = pos,
                                euler = rotation,
                                convexify =True,
                        ),
                    )
                    self.simulator.assets_entity[object_name] = asset
                    if "target" in object_name:
                        self.target_asset = asset
                        self.target_rotation = attributes["rotation"]
                        self.target_translation = attributes["translation"]
                    if "destination" in object_name:
                        self.desti_asset = asset
            else:
                if os.path.exists(asset_file):
                    # Direct position loading
                    pos = attributes["translation"]
                    pos_new, scale_factor = reproject_to_plane(pos, self.args.intrinsics, self.args.extrinsics, 0.02)
                    # pos = np.array(pos)
                    scale = attributes["scale"] * scale_factor
                    rotation = mat2quat(np.array(attributes["rotation"]))
                    
                    # Add asset with simple call
                    self.simulator.asset_addtion(asset_file, pos=pos_new, scale=scale, quat=rotation, physics=physics)
                    print(f"Loaded {key} at position {pos} with scale {scale}")

                    if "target" in object_name:
                        self.target_asset = self.simulator.asset_ID[scale][0]
                        self.target_rotation = attributes["rotation"]
                        self.target_translation = attributes["translation"]
                    if "destination" in object_name:
                        self.desti_asset = self.simulator.asset_ID[scale][0]
                else:
                    print(f"Warning: File not found - {asset_file}")
        
        self.camera_1 = self.simulator.scene.add_camera(
            res=(self.W, self.H),
            pos=self.args.camera_1_args["pos"],
            lookat=self.args.camera_1_args["lookat"],
            fov=self.args.camera_1_args["fov"],
            GUI=False,
        )
        
        # BUILD SCENE - following bridge_demo order
        print("Building scene...")
        self.simulator.scene.build()
        
        # Set physics properties AFTER build - following bridge_demo
        print("Setting physics properties...")
        if not self.args.default:
            for _, value in self.simulator.get_asset_ID().items():
                for key, asset in self.simulator.asset_ID.items():
                    asset[0].set_mass(asset[1])
                    asset[0].set_friction(asset[2])
                    # Set restitution to 0.0 to prevent bouncing (default is often 0.5)
    
        # Set camera pose
        if self.use_initail_camera:
            self.camera_0.set_pose(self.estimated_transform)
        
        # Set robot initial position
        print("Setting robot position...")
        self.simulator.robot.set_qpos(self.args.robot_args["init_pos"])
        
        # Get finger indices
        self.left_finger_idx = find_link_indices(self.simulator.robot, ['left_finger'], global_idx=True)[0]
        self.right_finger_idx = find_link_indices(self.simulator.robot, ['right_finger'], global_idx=True)[0]
        
        # Single step to stabilize - following bridge_demo
        self.simulator.step()

        for _, value in self.simulator.get_asset_ID().items():
            asset = value[0]
            # check if the object is below the ground and if so, move it up
            aabb_min, aabb_max = asset.get_AABB()
            z_bottom = aabb_min[2]
            if z_bottom < 0:
                # 1. Get the current position (it's a CUDA tensor)
                current_pos = asset.get_pos() 

                # 2. Create the offset tensor directly on the same GPU device
                # Using -z_bottom + 0.01 to lift the object out of the floor
                offset = torch.tensor([0.0, 0.0, -z_bottom + 0.01], device=current_pos.device)

                # 3. Perform the addition and set the new position
                asset.set_pos(current_pos + offset)
        
        # Set transformation properties for action execution
        self.set_transformation_properties()

        # collision detector
        solver = self.simulator.scene.sim.rigid_solver
        solver.collider.detection()
        entity_keys = list(self.simulator.assets_entity.keys())
        for _ in range(5):
            moved = False
            for i in range(len(entity_keys)):
                entity = self.simulator.assets_entity[entity_keys[i]]
                for j in range(i + 1, len(entity_keys)):
                    other_entity = self.simulator.assets_entity[entity_keys[j]]
                    moved = moved or resolve_overlap(entity, other_entity, entity1_name=entity_keys[i], entity2_name=entity_keys[j], buffer=0.005)
            self.simulator.step()
        
        print("Setup complete!")

    def set_transformation_properties(self):
        self.ee_link = self.simulator.robot.get_link("ee_gripper_link")
        self.prev_ee_pose_at_world = sapien.Pose(
            self.simulator.robot.get_link("ee_gripper_link").get_pos().cpu().numpy(),
            self.simulator.robot.get_link("ee_gripper_link").get_quat().cpu().numpy()
        )
        self.base_in_world_inv = sapien.Pose(
            self.simulator.robot.get_link("base_link").get_pos().cpu().numpy(),
            self.simulator.robot.get_link("base_link").get_quat().cpu().numpy()
        ).inv()
        self.base_in_world = sapien.Pose(
            self.simulator.robot.get_link("base_link").get_pos().cpu().numpy(),
            self.simulator.robot.get_link("base_link").get_quat().cpu().numpy()
        )
        self.prev_ee_pose_at_base = self.base_in_world_inv * self.prev_ee_pose_at_world
        self.prev_gripper = 0.0
    
    def prepare_action_payload(self, image):
        if self.args.model_name == "xvla":
            payload = {
                "language_instruction": self.task_description,
                "image0": json_np.dumps(image),
                "proprio": json_np.dumps(self._get_xvla_proprioception()),
                "domain_id": 0,
                "steps": 10,
            }
        elif self.args.model_name == "open_pi_zero":
            payload = {
                "instruction": self.task_description,
                "image": json_np.dumps(image),
                "proprio": json_np.dumps(self._get_openpi_proprioception()),
            }
        else:
            payload = {
                "instruction": self.task_description,
                "image": image.tolist(),
            }
        return payload
    
    def reset_model(self):
        requests.post(f"http://localhost:{self.port}/reset",
        json={
            "instruction": self.args.task_description,
        })
    
    def set_task(self, task_description):
        requests.post(f"http://localhost:{self.port}/set_task",
        json={
            "task_description": task_description,
        })
    
    def transform_actions_delta(self, raw_action, action):
        delta_quat = euler2quat(*raw_action["rotation_delta"])
        delta_pose = sapien.Pose(raw_action["world_vector"], delta_quat)
        cur_ee_pose_at_world = sapien.Pose(self.ee_link.get_pos().cpu().numpy(), self.ee_link.get_quat().cpu().numpy())
        cur_ee_pose_at_base = self.base_in_world_inv * cur_ee_pose_at_world
        target_pose = (sapien.Pose(p=cur_ee_pose_at_base.p) * delta_pose * sapien.Pose(p=cur_ee_pose_at_base.p).inv()) * self.prev_ee_pose_at_base
        self.final_pose = self.base_in_world * target_pose
        self.prev_ee_pose_at_base = target_pose
        self.prev_gripper = action["gripper"]
        self.prev_ee_pose_at_world = self.final_pose
        return self.final_pose

    def transform_actions_exact(self, raw_action, action):
        target_quat = euler2quat(*raw_action["rotation_delta"])
        target_pose_in_base = sapien.Pose(raw_action["world_vector"], target_quat)
        self.final_pose = self.base_in_world * target_pose_in_base
        self.prev_ee_pose_at_base = target_pose_in_base
        self.prev_gripper = action["gripper"]
        self.prev_ee_pose_at_world = self.final_pose
        return self.final_pose

    def transform_actions(self, raw_action, action):
        if self.args.model_name == "xvla":
            return self.transform_actions_exact(raw_action, action)
        else:
            return self.transform_actions_delta(raw_action, action)
    
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
                action_pred = np.array(self.pred_action_queue.popleft(), dtype=np.float32)
                self._update_xvla_proprioception(action_pred)
        
        self.transform_actions(raw_action, action)
        return self.final_pose, action["gripper"], image
    
    def get_image(self):
        rgb, _, seg, _ = self.camera_0.render(rgb=True, depth=True, segmentation=True)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = (seg > 0).astype(np.uint8) * 255
        segmented = cv2.bitwise_and(rgb_bgr, rgb_bgr, mask=mask)
        background = cv2.bitwise_and(self.target_image, self.target_image, mask=cv2.bitwise_not(mask))
        blended = cv2.add(segmented, background)
        return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), rgb
    
    def _get_openpi_proprioception(self):
        ee_pose_wrt_base = self.prev_ee_pose_at_base
        return np.concatenate([ee_pose_wrt_base.p, ee_pose_wrt_base.q, np.array([self.prev_gripper])])
    
    def _init_xvla_proprioception(self):
        ee_pose_wrt_base = self.prev_ee_pose_at_base
        proprioception = torch.from_numpy(np.concatenate([ee_pose_wrt_base.p, np.array([1, 0, 0, 1, 0, 0, 0])])).to(dtype=torch.float32)
        proprioception = torch.cat([proprioception, torch.zeros_like(proprioception)], dim=-1).numpy().copy()
        self.xvla_proprioception = proprioception.copy()
        return proprioception
    
    def _update_xvla_proprioception(self, action):
        self.xvla_proprioception[:10] = action[:10]
        return self.xvla_proprioception

    def _get_xvla_proprioception(self):
        if getattr(self, 'xvla_proprioception', None) is None:
            self._init_xvla_proprioception()
        return self.xvla_proprioception

    def _xvla_retrieve_raw_action(self, action):
        wrapped_action = {
            "world_vector": action[:3],
            "rot_axangle": action[3:6],
            "gripper": action[6],
        }
        world_vector = wrapped_action["world_vector"]
        gripper_val = wrapped_action["gripper"]
        open_gripper_val = 1.0 if gripper_val > 0 else 0.0
        open_gripper = np.array([open_gripper_val])

        raw_action_retrieved = {
            "world_vector": world_vector,
            "rotation_delta": action[3:6],
            "open_gripper": open_gripper
        }

        return raw_action_retrieved, wrapped_action
    
    def _openpi_action_parser(self, action):
        raw_action = {
            "world_vector": action[:3],
            "rotation_delta": action[3:6],
            "open_gripper": action[6],
        }
        wrapped_action = {
            "world_vector": action[:3],
            "rot_axangle": action[3:6],
            "gripper": raw_action["open_gripper"]
        }
        return raw_action, wrapped_action
    
    def _xvla_action_parser(self, action):
        self.pred_action_queue.append(action)
        action_final = np.concatenate([
            action[:3],
            rotate6D_to_euler_xyz(action[3:9]) + np.array([0, math.pi / 2, 0]),
            np.array([1 if action[9] < 0.9 else -1])
        ])
        return self._xvla_retrieve_raw_action(action_final)

    def action_parser(self, action):
        if self.args.model_name == "xvla":
            return self._xvla_action_parser(action)
        elif self.args.model_name == "open_pi_zero":
            return self._openpi_action_parser(action)
        else:
            raise NotImplementedError(f"Action parser not implemented for model {self.args.model_name}")
    
    def run_default_test(self):
        imx = []
        results = {}
        base_states = []
        world_states = []
        object_states = {}
        reward = 0
        print("Running Test on task:", self.args.task_description)
        self.set_task(self.args.task_description)
        self.reset_model()
        
        enable_welding = True
        welded = False
        link_robot = np.array([self.simulator.robot.get_link("ee_gripper_link").idx], dtype=gs.np_int)
        rigid = self.simulator.scene.sim.rigid_solver
        
        for i in range(65):
            if i >= 5:
                pose, gripper, image = self.get_action()
                base_states.append(np.concatenate([self.prev_ee_pose_at_base.p, self.prev_ee_pose_at_base.q, np.array([self.prev_gripper]) if np.isscalar(self.prev_gripper) else self.prev_gripper]))
                world_states.append(np.concatenate([self.prev_ee_pose_at_world.p, self.prev_ee_pose_at_world.q, np.array([self.prev_gripper]) if np.isscalar(self.prev_gripper) else self.prev_gripper]))
                for key in self.simulator.assets_entity.keys():
                    if key not in object_states:
                        object_states[key] = []
                    asset = self.simulator.assets_entity[key]
                    pos = asset.get_pos().cpu().numpy()
                    quat = asset.get_quat().cpu().numpy()
                    vel = asset.get_vel().cpu().numpy()
                    ang = asset.get_ang().cpu().numpy()
                    object_states[key].append(np.concatenate([pos, quat, vel, ang]))
            else:
                pose = self.prev_ee_pose_at_world
                gripper = 0.0
                image = None
            
            des_q = self.simulator.robot.inverse_kinematics(
                link=self.ee_link,
                pos=np.array(pose.p),
                quat=np.array(pose.q),
            )
            if gripper > 0:
                des_q[6] = 0.037
                des_q[7] = 0.037
            else:
                des_q[6] = 0.00
                des_q[7] = 0.00

            self.simulator.robot.control_dofs_position(des_q)
            if gripper > 0:
                if enable_welding:
                    if welded:
                        welded = False
                        rigid.delete_weld_constraint(link_obj, link_robot)
                self.simulator.robot.control_dofs_force(np.array([10, 10]), dofs_idx_local=np.arange(6, 8))
                for _ in range(60):
                    self.simulator.scene.step()
            else:
                for _ in range(60):
                    self.simulator.robot.control_dofs_force(
                        apply_safety_limits(
                            torch.tensor([-10, -10]).to(device='cuda:0'),
                            self.simulator.robot.get_dofs_position()[6:],
                            self.simulator.robot.get_dofs_velocity()[6:],
                            motors_soft_position_lower=0.01,
                            motors_soft_position_upper=0.037,
                            motors_effort_limit=10,
                            motors_velocity_limit=0.05,
                            kp=100,
                            kd=0.5,
                        ),
                        dofs_idx_local=torch.tensor([6, 7]).to(device='cuda:0')
                    )
                    self.simulator.step()
                    
                    if enable_welding:
                        for e in self.simulator.scene.entities:
                            if e is self.simulator.robot:
                                continue
                            is_grasping = is_grasping_two_finger(
                                self.simulator.robot,
                                e,
                                self.left_finger_idx,
                                self.right_finger_idx
                            )
                            if is_grasping:
                                if not welded:
                                    # add suction / weld constraint
                                    link_obj = np.array([e.links[0].idx], dtype=gs.np_int)
                                    rigid.add_weld_constraint(link_obj, link_robot)
                                    welded = True
            
            if image is not None:
                imx.append(image)
        
        folder = self.output_dir
        os.makedirs(folder, exist_ok=True)
        gs.tools.animate(imx, os.path.join(folder, f"test_{self.args.task_description}_{self.args.test_id}.mp4"), fps=5)
        world_states = np.array(world_states)
        base_states = np.array(base_states)
        np.savez(os.path.join(folder, f"test_{self.args.task_description}_{self.args.test_id}.npz"), world=world_states, base=base_states, **object_states)

    def reward_reaching_cube(self, ee_link, asset):
        tcp_goal_dist = torch.linalg.norm(
            asset.get_pos() - ee_link.get_pos())
        reaching_reward = 1 - torch.tanh(5 * tcp_goal_dist)
        return reaching_reward

    def run(self):
        try:
            self.setup()
            self.camera_pos = self.camera_0.pos
            self.camera_lookat = self.camera_0.lookat
            self.run_default_test()
        except:
            import sys
            import traceback
            print(f"Error processing task '{self.args.task_description}' in scene '{self.args.scene_name}': {traceback.format_exc()}", file=sys.stderr)
        gs.destroy()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_config(config_path="config.yaml"):
    # Load YAML config
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# load from 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing Scene Generation')
    parser.add_argument('--config', type=str, default="configs/default.yaml", help='Path to the YAML config file')
    parser.add_argument('--run_all', type=str2bool, default=False, help='Run all tests')
    parser.add_argument('--output_dir', type=str, default="./results", help='Output directory for results')
    parser.add_argument('--port', type=int, default=9010, help='Port for the server')
    parser.add_argument('--vla', type=str, required=True, help='Name of the VLA model')
    args = parser.parse_args()
    robot_args = {
        "name": "WidowX",
        "init_pos": torch.tensor(
            [-0.02014876, 0.04723017, 0.22625704, -0.00307271, 1.365988,
            -0.00168102, 0.037, 0.03699991], device='cuda:0'
        )
    }
    camera_1_args = {
        "pos": (0.25, 0.25, 1.3),
        "lookat": (0.1, 0.1, 1.0),
        "fov": 80,
    }

    config = load_config(args.config)
    base_folder = config['base_folder']
    scene_name = config['scene_name']
    output_folder = args.output_dir
    port = args.port
    model_name = args.vla
    
    run_default = False
    if "default" in output_folder:
        run_default = True
    if args.run_all:
        scene_lists = os.listdir(os.path.join(base_folder, "bridge"))
    else:
        scene_lists = [scene_name] if type(scene_name) is str else scene_name
    
    print("Scenes to process:", scene_lists)
    
    for scene_name in scene_lists:
        try:
            if scene_name.startswith("default"):
                if run_default:
                    default = True
                else:
                    continue
            
            elif scene_name.startswith("scene"):
                if run_default:
                    continue
                else:
                    default = False
            else:
                continue
            
            if os.path.exists(os.path.join(output_folder, "default_test", scene_name)):
                print(f"Skipping {scene_name} as results already exist.")
                continue
            
            data_folder = os.path.join(base_folder, "bridge", scene_name)
            asset_folder = os.path.join(base_folder, "assets", scene_name)
            background = os.path.join(base_folder, "scene_background", scene_name, "background.png")
            extrinsics = np.load(os.path.join(data_folder, "extrinsics.npy"))
            intrinsics = np.load(os.path.join(data_folder, "intrinsics.npy"))
            
            with open(os.path.join(data_folder, "masks", "transformations.json"), 'r') as f:
                object_positions = json.load(f)
            
            if not os.path.exists(os.path.join(data_folder, "physical_properties.json")):
                with open(os.path.join(data_folder, "masks", "result.json"), 'r') as f:
                    physics_properties = json.load(f)
            else:
                with open(os.path.join(data_folder, "physical_properties.json"), 'r') as f:
                    physics_properties = json.load(f)
            
            task_path = os.path.join(data_folder, "lang.txt")
            with open(task_path, 'r') as file:
                task_lines = file.readlines()
            task_lines = [line.strip() for line in task_lines]
            
            for task_description in task_lines:
                if "confidence" in task_description:
                    continue
                for i in range(1):
                    args = SimpleNamespace(
                        default=default,
                        robot_args=robot_args,
                        background=cv2.imread(background),
                        task_description=task_description,
                        camera_1_args=camera_1_args,
                        intrinsics=intrinsics,
                        extrinsics=extrinsics,
                        asset_folder=asset_folder,
                        object_positions=object_positions,
                        object_properties=physics_properties,
                        test_id=i,
                        port=port,
                        scene_name=scene_name,
                        output_dir=os.path.join(output_folder, "default_test", scene_name),
                        model_name=model_name,
                    )
                    p = DefaultTest(args)
                    p.run()
        except:
            import sys
            import traceback
            print(f"Error processing scene '{scene_name}': {traceback.format_exc()}", file=sys.stderr)
