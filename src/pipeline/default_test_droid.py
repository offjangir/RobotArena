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
from src.utils.test_utils import get_genesis_extrinsics, reproject_to_plane, find_link_indices, is_grasping_two_finger, apply_safety_limits
from transforms3d.axangles import mat2axangle
from transforms3d.euler import euler2axangle, euler2mat, euler2quat, quat2euler
from transforms3d.quaternions import axangle2quat, mat2quat, quat2axangle, quat2mat,qconjugate 
from transforms3d.quaternions import qmult, mat2quat, quat2mat
from transforms3d.quaternions import rotate_vector
import time 
import logging
import base64
MIN_OPENING = 0.01
MAX_OPENING = 0.8
OPENING_DIRECTION = -1

def quat_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    half_angle = angle / 2.0
    return np.array([np.cos(half_angle), *(axis * np.sin(half_angle))])

def compute(actions, robot, dofs_idx_local):
    # NOTE: you are using sigmoid so make sure input is <0 for open and 0< for close
    dofs_pos_target = MIN_OPENING + (MAX_OPENING - MIN_OPENING) * (
        torch.sigmoid(actions) > 0.5
    ).repeat_interleave(2, dim=1)
    torques = (
        100 * (dofs_pos_target - robot.get_dofs_position(dofs_idx_local=dofs_idx_local))
        - 10 * robot.get_dofs_velocity(dofs_idx_local=dofs_idx_local)
    )
    torques = torques[:,0]
    # NOTE: clip to some reasonable value
    robot.control_dofs_force(torch.clip(torques[0], -20, 20), dofs_idx_local=dofs_idx_local)


class DefaulTest:
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.args = args
    
    def setup(self):
        self.simulator = SimulatorGenesis(
            self.args.robot_args["name"], 1, show_viewer=False, add_robot=True
        )
        self.target_asset = None
        self.desti_asset = None
        self.simulator.start_sim(default = self.args.default)
        self.task_description = self.args.task_description
        self.target_image =self.args.background
        self.output_dir = self.args.output_dir
        self.estimated_transform, fov, W, H = get_genesis_extrinsics(self.args.extrinsics, self.args.intrinsics)
        self.use_initail_camera = True
        self.target_rotation = None
        self.target_translation = None
        self.port = self.args.port
        
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
        
        for key, attributes in self.args.object_positions.items():
            count = int(key.split("_")[1])
            asset_file = os.path.join(self.args.asset_folder, key + ".glb") 
            if self.args.object_properties:
                physics = self.args.object_properties[count]
            else:
                physics = None
            object_name = physics["object_name"]
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
                    if "target" in object_name:
                        self.target_asset = asset
                        self.target_rotation = attributes["rotation"]
                        self.target_translation = attributes["translation"]
                    if "destination" in object_name:
                        self.desti_asset = asset
            else:
                if os.path.exists(asset_file):  
                    pos = attributes["translation"]
                    pos = np.array(pos)
                    scale = attributes["scale"]
                    rotation = mat2quat(np.array(attributes["rotation"]))
                    scale = self.simulator.asset_addtion(asset_file, pos=pos, scale=scale, quat=rotation, physics=physics)
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
            pos = self.args.camera_1_args["pos"],
            lookat = self.args.camera_1_args["lookat"],
            fov = self.args.camera_1_args["fov"],
            GUI=False,
        )
            
        self.reward_func = None
        self.simulator.scene.build()
        if not self.args.default:
            for key, asset in self.simulator.asset_ID.items():
                asset[0].set_mass(asset[1])
                asset[0].set_friction(asset[2])
        if self.use_initail_camera:
            self.camera_0.set_pose(self.estimated_transform)
        self.simulator.robot.set_qpos(self.args.robot_args["init_pos"][:7], qs_idx_local=np.arange(7))
        
        self.left_finger_idx = find_link_indices(self.simulator.robot, ['left_pad'], global_idx=True)[0]
        self.right_finger_idx = find_link_indices(self.simulator.robot, ['right_pad'], global_idx=True)[0]
        
        self.simulator.step()
        self.set_transformation_properties()  # Typo fixed: transformation not tranformation

    
    def set_transformation_properties(self):
        self.ee_link = self.simulator.robot.get_link("base")
        self.prev_ee_pose_at_world = sapien.Pose(self.simulator.robot.get_link("base").get_pos().cpu().numpy(), self.simulator.robot.get_link("base").get_quat().cpu().numpy())
        self.base_in_world_inv =  sapien.Pose(self.simulator.robot.get_link("link0").get_pos().cpu().numpy(), self.simulator.robot.get_link("link0").get_quat().cpu().numpy()).inv()
        self.base_in_world =sapien.Pose(self.simulator.robot.get_link("link0").get_pos().cpu().numpy(), self.simulator.robot.get_link("link0").get_quat().cpu().numpy())
        self.prev_ee_pose_at_base = self.base_in_world_inv * self.prev_ee_pose_at_world
        
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
    
    def transform_actions(self, raw_action, action):
        delta_quat = euler2quat(*raw_action["rotation_delta"])
        delta_pose  = sapien.Pose(raw_action["world_vector"],delta_quat)
        cur_ee_pose_at_world = sapien.Pose(self.ee_link.get_pos().cpu().numpy(), self.ee_link.get_quat().cpu().numpy())
        cur_ee_pose_at_base = self.base_in_world_inv * cur_ee_pose_at_world
        target_pose = (sapien.Pose(p=cur_ee_pose_at_base.p)* delta_pose * sapien.Pose(p=cur_ee_pose_at_base.p).inv()) * self.prev_ee_pose_at_base
        self.final_pose = self.base_in_world * target_pose
        self.prev_ee_pose_at_base = target_pose
        return self.final_pose
        
    def get_action(self):
        image, rgb = self.get_image()
        start_time = time.time()
        _, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        res = requests.post(f"http://localhost:{self.port}/act",
        json={
            "instruction": self.task_description,
            "image": jpg_as_text,  
        })
        response = res.json()
        raw_action = {k: np.array(v) for k, v in response["raw_action"].items()}
        action = {k: np.array(v) for k, v in response["action"].items()}
        self.transform_actions(raw_action, action)
        # self.final_pose = None
        # action["gripper"] = None
        return self.final_pose, action["gripper"],image
        
    def get_image(self):
        rgb, _, seg, _ = self.camera_0.render(rgb=True, depth=True, segmentation=True)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = (seg > 0).astype(np.uint8) * 255
        segmented = cv2.bitwise_and(rgb_bgr, rgb_bgr, mask=mask)
        background = cv2.bitwise_and(self.target_image, self.target_image, mask=cv2.bitwise_not(mask))
        blended = cv2.add(segmented, background)
        return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), rgb

    def run_default_test(self):
        imx = []
        results = {}
        reward = 0
        print("Running Test on task:", self.args.task_description)
        self.set_task(self.args.task_description) 
        self.reset_model()

        enable_welding = True
        welded = False
        link_robot = np.array([ self.simulator.robot.get_link("base").idx], dtype=gs.np_int)
        rigid = self.simulator.scene.sim.rigid_solver
        
        for i in range(150):
            pose, gripper, image = self.get_action()
            des_q = self.simulator.robot.inverse_kinematics(
                link=self.ee_link,
                pos=np.array(pose.p),
                quat=np.array(pose.q),
            )
            # Gripper positions are in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
            self.simulator.robot.control_dofs_position(des_q[:7], dofs_idx_local=np.arange(7))
            for i in range(10):
                compute(torch.tensor([[gripper]], dtype=torch.float32), self.simulator.robot, np.arange(7,9))
                self.simulator.step()
            # for _ in range(5):
            #     self.simulator.scene.step()
            # if gripper > 0:
            #     des_q[7:9] = np.array([0.00, 0.00])
            # else:
            #     # set gripper to joint limit of fully open(0.037)
            #     des_q[7:9] = np.array([0.037, 0.037])
            # simulator.robot.control_dofs_position(des_q[:9], dofs_idx_local=np.arange(9)) 
            
            # if gripper <= 0:
            #     if enable_welding:
            #         if welded:
            #             welded = False
            #             rigid.delete_weld_constraint(link_obj, link_robot)
            #     simulator.robot.control_dofs_force(np.array([-10, -10]), dofs_idx_local=np.arange(7,9))
            #     for i in range(10):
            #         simulator.step()
            # else:
            #     for i in range(10):
            #         simulator.robot.control_dofs_force(
            #             apply_safety_limits(
            #                 torch.tensor([10, 10]),
            #                 simulator.robot.get_dofs_position()[7:9],
            #                 simulator.robot.get_dofs_velocity()[7:9],
            #                 motors_soft_position_lower = 0.01,
            #                 motors_soft_position_upper = 0.037,
            #                 motors_effort_limit = 10,
            #                 motors_velocity_limit = 0.05,
            #                 kp = 100,
            #                 kd = 0.5,
            #             ),
            #             dofs_idx_local = torch.tensor([7,9], dtype=torch.int64)
            #         )
            #         simulator.step()
                    
            #         if enable_welding:
            #             for e in simulator.scene.entities:
            #                 if e is simulator.robot:
            #                     continue
            #                 is_grasping = is_grasping_two_finger(
            #                     simulator.robot,
            #                     e,
            #                     left_finger_idx,
            #                     right_finger_idx
            #                 )
            #                 if is_grasping:
            #                     if not welded:
            #                         link_obj = np.array([e.links[0].idx], dtype=gs.np_int)
            #                         rigid.add_weld_constraint(link_obj, link_robot)
            #                         welded = True

            imx.append(image)
        
        folder = self.output_dir
        os.makedirs(folder, exist_ok=True)
        gs.tools.animate(imx, os.path.join(folder, f"test_{self.args.task_description}_{self.args.test_id}.mp4"), fps=5)
        

    def run(self):
        self.setup()
        self.camera_pos = self.camera_0.pos
        self.camera_lookat = self.camera_0.lookat
        self.run_default_test()
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

# env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
# load from 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing Scene Generation')
    parser.add_argument('--config', type=str, default="configs/default1.yaml", help='Path to the YAML config file')
    parser.add_argument('--run_all', type=str2bool, default=False, help='Run all tests')
    parser.add_argument('--output_dir', type=str, default="./results_debug", help='Output directory for results')
    parser.add_argument('--port', type=int, default=9050, help='Port for the server')
    args = parser.parse_args()
    robot_args = {
        "name": "franka",
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
    
    run_default = False
    if "default" in output_folder:
        run_default = True
    if args.run_all:
        scene_lists = os.listdir(os.path.join(base_folder, "DROID"))
    else:
        scene_lists = [scene_name]

    print(scene_lists) 
    print(len(scene_lists))

    for scene_name in scene_lists:
        if scene_name.startswith("default"):
            if run_default:
                default = True
            else:
                continue
        
        elif scene_name.startswith("task") or scene_name.startswith("droid"):
            if run_default:
                continue
            else:
                default = False
        else:
            continue
        
        
        data_folder = os.path.join(base_folder, "DROID", scene_name)
        asset_folder = os.path.join(base_folder, "assets", scene_name)
        background = os.path.join(base_folder, "scene_background", scene_name, "background.png")

        # Check existence
        if not os.path.exists(data_folder):
            print(f"Data folder not found: {data_folder}, skipping...")
            continue  # if inside a loop over scenes

        if not os.path.exists(asset_folder):
            print(f"Asset folder not found: {asset_folder}, skipping...")
            continue

        if not os.path.exists(background):
            print(f"Background image not found: {background}, skipping...")
            continue

        # Check extrinsics and intrinsics files
        extrinsics_path = os.path.join(data_folder, "extrinsics.npy")
        intrinsics_path = os.path.join(data_folder, "intrinsics.npy")

        if not os.path.exists(extrinsics_path):
            print(f"Extrinsics file not found: {extrinsics_path}, skipping...")
            continue

        if not os.path.exists(intrinsics_path):
            print(f"Intrinsics file not found: {intrinsics_path}, skipping...")
            continue

        # Load data if all files exist
        extrinsics = np.load(extrinsics_path)
        intrinsics = np.load(intrinsics_path)

        with open(os.path.join(data_folder, "masks", "transformations_new.json"), 'r') as f:
            object_positions = json.load(f)
        
        if not os.path.exists(os.path.join(data_folder, "physical_properties.json")):
            with open(os.path.join(data_folder, "masks","result.json"), 'r') as f:
                physics_properties = json.load(f)
        else:
            with open(os.path.join(data_folder, "physical_properties.json"), 'r') as f:
                physics_properties = json.load(f)
        # Proprioception file
        joints_path = os.path.join(data_folder, "joints.npy")
        if os.path.exists(joints_path):
            proprioception_data = np.load(joints_path)
            init_pos = proprioception_data[0]
            robot_args["init_pos"] = torch.tensor(init_pos, device='cuda:0')
        else:
            print(f"Proprioception file not found: {joints_path}, skipping...")
            continue  # if inside a loop over scenes

        # Task file
        task_path = os.path.join(data_folder, "lang.txt")
        if os.path.exists(task_path):
            with open(task_path, 'r') as file:
                task_lines = file.readlines()
        else:
            print(f"Task file not found: {task_path}, skipping...")
            continue
        task_lines = [line.strip() for line in task_lines]
        
        for task_description in task_lines:
            if "confidence" in task_description:
                continue
            for i in range (1):
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
                    test_id = i,
                    port = port,
                    scene_name = scene_name,
                    output_dir = os.path.join(output_folder, "default_test",scene_name),
                )
                p = DefaulTest(args)
                p.run()