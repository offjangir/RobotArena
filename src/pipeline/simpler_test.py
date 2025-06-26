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
from src.sim.simulator_genesis import SimulatorGenesis, process_surface
from src.utils.test_utils import get_genesis_extrinsics, reproject_to_plane, find_link_indices, is_grasping_two_finger, apply_safety_limits
from transforms3d.axangles import mat2axangle
from transforms3d.euler import euler2axangle, euler2mat, euler2quat, quat2euler
from transforms3d.quaternions import axangle2quat, mat2quat, quat2axangle, quat2mat,qconjugate 
from transforms3d.quaternions import qmult, mat2quat, quat2mat
from transforms3d.quaternions import rotate_vector
from src.pipeline.default_test import DefaulTest, load_config, str2bool


class genesis_object:
    """
    Data class to hold details for a simulation object, derived from config.
    Includes initial position which was added in the previous refactor.
    """
    def __init__(self, name, file, initial_pos=None, quat=None, scale=None, mass=None, friction=None, surface=None):
        self.name = name
        self.file = file
        self.initial_pos = initial_pos # Keep initial_pos
        self.quat = quat
        self.scale = scale
        self.mass = mass
        self.friction = friction
        self.surface = surface

    def __str__(self):
        return (f"genesis_object(name='{self.name}', file='{self.file}', "
                f"initial_pos={self.initial_pos}, quat={self.quat}, scale={self.scale}, "
                f"mass={self.mass}, friction={self.friction}, surface={self.surface})")

class SimplerTest:
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.args = args
    
    def setup(self):
        self.simulator = SimulatorGenesis(
            self.args.robot_args["name"], 1, show_viewer=False, add_robot=True
        )
        self.target_asset = None
        self.desti_asset = None
        if self.args.scene_name == "default2":
            self.simulator.start_sim(default = True, special_light=True)
        else:
            self.simulator.start_sim(default = True)
        
        self.task_description = self.args.task_description
        self.target_image = cv2.imread(self.args.background_path)
        self.H, self.W = self.target_image.shape[:2]  # get height and width from image
        self.target_image = cv2.resize(self.target_image, (self.W, self.H))
        
        estimated_transform, fov, W, H = get_genesis_extrinsics(self.args.extrinsics, self.args.intrinsics)
        self.estimated_transform = estimated_transform
        self.camera_0 = self.simulator.scene.add_camera(
            res=(self.W, self.H),
            pos = (0, 0, 0),
            lookat = (0, 0, 0),
            fov = fov,
            GUI=False,
        )
        for key, attributes in self.args.object_positions.items():
            count = int(key.split("_")[1])
            asset_file = os.path.join(self.args.asset_folder, key + ".glb") 
            if self.args.object_properties:
                physics = self.args.object_properties[count]
            else:
                physics = None
            if os.path.exists(asset_file):  
                pos = attributes["translation"]
                scale= attributes["scale"]
                rotation = attributes["rotation"]
                object_name = physics["object_name"]
                if "target" in object_name and hasattr(self.args, "replace_object"):
                    asset = self.simulator.scene.add_entity(
                            morph=gs.morphs.Mesh(
                                file=self.args.replace_object.file,
                                scale=self.args.replace_object.scale,
                                pos= pos,
                                quat=self.args.replace_object.quat,
                                convexify=True,
                            ),
                            surface = process_surface(self.args.replace_object.surface),
                        )
                    self.target_asset = asset
                    continue
                if "emission" in attributes:
                    asset = self.simulator.scene.add_entity(
                        gs.morphs.Box(
                            size = (0.03, 0.03, 0.03),
                            pos  = pos,),
                            surface = gs.surfaces.Emission(
                                emissive=attributes["emission"]
                            )
                    )
                else:
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
                if "destination" in object_name:
                    self.desti_asset = asset
        
        self.camera_1 = self.simulator.scene.add_camera(
            res=(self.W, self.H),
            pos = self.args.camera_1_args["pos"],
            lookat = self.args.camera_1_args["lookat"],
            fov = self.args.camera_1_args["fov"],
            GUI=False,
        )
            
        self.reward_func = None
        self.simulator.scene.build()
        
        if hasattr(self.args, "replace_object"):
            self.target_asset.set_mass(self.args.replace_object.mass)
            self.target_asset.set_friction(self.args.replace_object.friction)
            
        self.camera_0.set_pose(self.estimated_transform)
        self.simulator.robot.set_qpos(self.args.robot_args["init_pos"])
        self.left_finger_idx = find_link_indices(self.simulator.robot, ['left_finger'], global_idx=True)[0]
        self.right_finger_idx = find_link_indices(self.simulator.robot, ['right_finger'], global_idx=True)[0]

        self.simulator.step()
        self.set_transformation_properties()  # Typo fixed: transformation not tranformation

    
    def set_transformation_properties(self):
        self.ee_link = self.simulator.robot.get_link("ee_gripper_link")
        self.prev_ee_pose_at_world = sapien.Pose(self.simulator.robot.get_link("ee_gripper_link").get_pos().cpu().numpy(), self.simulator.robot.get_link("ee_gripper_link").get_quat().cpu().numpy())
        self.base_in_world_inv =  sapien.Pose(self.simulator.robot.get_link("base_link").get_pos().cpu().numpy(), self.simulator.robot.get_link("base_link").get_quat().cpu().numpy()).inv()
        self.base_in_world =sapien.Pose(self.simulator.robot.get_link("base_link").get_pos().cpu().numpy(), self.simulator.robot.get_link("base_link").get_quat().cpu().numpy())
        self.prev_ee_pose_at_base = self.base_in_world_inv * self.prev_ee_pose_at_world
        
    
    def reset_model(self):
        requests.post(f"http://localhost:{self.args.port}/reset",
        json={
            "instruction": self.args.task_description, 
        })
    
    def set_task(self, task_description):
        requests.post(f"http://localhost:{self.args.port}/set_task",
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
        res = requests.post(f"http://localhost:{self.args.port}/act",
        json={
            "instruction": self.task_description,
            "image": image.tolist(),  
        })
        response = res.json()
        raw_action = {k: np.array(v) for k, v in response["raw_action"].items()}
        action = {k: np.array(v) for k, v in response["action"].items()}
        self.transform_actions(raw_action, action)
        return self.final_pose, action["gripper"],image
        
    def get_image(self):
        rgb, _, seg, _ = self.camera_0.render(rgb=True, depth=True, segmentation=True)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = (seg > 0).astype(np.uint8) * 255
        segmented = cv2.bitwise_and(rgb_bgr, rgb_bgr, mask=mask)
        background = cv2.bitwise_and(self.target_image, self.target_image, mask=cv2.bitwise_not(mask))
        blended = cv2.add(segmented, background)
        return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), rgb

    def run_default_test(self, default=False):
        imx = []
        results = {}
        reward = 0
        print("Running Test on task:", self.args.task_description)
        self.set_task(self.args.task_description) 
        self.reset_model()
        
        enable_welding = True
        welded = False
        link_robot = np.array([ self.simulator.robot.get_link("ee_gripper_link").idx], dtype=gs.np_int)
        rigid = self.simulator.scene.sim.rigid_solver
        
        for i in range(60):
            pose, gripper, image = self.get_action()
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
                self.simulator.robot.control_dofs_force(np.array([10, 10]), dofs_idx_local=np.arange(6,8))
                for i in range(60):
                    self.simulator.scene.step()
            else:
                for i in range(60):
                    self.simulator.robot.control_dofs_force(
                        apply_safety_limits(
                            torch.tensor([-10,-10]),
                            self.simulator.robot.get_dofs_position()[6:],
                            self.simulator.robot.get_dofs_velocity()[6:],
                            motors_soft_position_lower = 0.01,
                            motors_soft_position_upper = 0.037,
                            motors_effort_limit = 10,
                            motors_velocity_limit = 0.05,
                            kp = 100,
                            kd = 0.5,
                        ),
                        dofs_idx_local = torch.tensor([6,7]).to(device='cuda:0')
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
                                    
            reward += self.reward_reaching_cube(self.ee_link, self.target_asset)
            imx.append(image)
        print(f"Reward: {reward}")
        
        os.makedirs(self.args.output_dir, exist_ok=True)
        if self.args.test_type == "default":
            gs.tools.animate(imx, os.path.join(self.args.output_dir, f"default_test_{self.args.test_id}.mp4"), fps=5)
        elif self.args.test_type == "background":
            gs.tools.animate(imx, os.path.join(self.args.output_dir, f"background_test_{self.args.test_id}.mp4"), fps=5)
        else:
            gs.tools.animate(imx, os.path.join(self.args.output_dir, f"replace_to_{self.args.replace_object.name}_test_{self.args.test_id}.mp4"), fps=5)
            
        

    def reward_reaching_cube(self, ee_link, asset):
        tcp_goal_dist = torch.linalg.norm(
            asset.get_pos() - ee_link.get_pos())
        reaching_reward = 1 - torch.tanh(5 * tcp_goal_dist)
        return reaching_reward

    def run(self, default=False):
        self.setup()
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


def get_replace_object(config):
    base_folder = config.get('base_folder', './')
    scene_name = config.get('replace_name', 'default_scene')
    obj_cnt_to_replace = config.get('obj_cnt', None) # Get obj_cnt here
    data_folder = os.path.join(base_folder, "bridge", scene_name)
    asset_folder = os.path.join(base_folder, "assets", scene_name)
    # --- Load Transformations and Physics ---
    object_transforms = {}
    transformations_path = os.path.join(data_folder, "masks", "transformations.json")
    try:
        with open(transformations_path, 'r') as f:
            object_transforms = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Transformations file not found at {transformations_path}. Cannot perform object replacement.")

    physics_properties = {}
    physics_path_primary = os.path.join(data_folder, "physical_properties.json")
    physics_path_fallback = os.path.join(data_folder, "masks", "result.json")
    physics_to_load = None
    if os.path.exists(physics_path_primary):
        physics_to_load = physics_path_primary
    elif object_transforms is not None and os.path.exists(physics_path_fallback):
        physics_to_load = physics_path_fallback
    with open(physics_to_load, 'r') as f:
        physics_properties = json.load(f)
    replace_obj_details = None
    # Proceed only if object_cnt is specified and transformation/physics data was loaded
    found_count = None
    found_obj_key = None
    for key in object_transforms.keys():
        count = int(key.split("_")[1])
        if count == obj_cnt_to_replace:
            found_obj_key = key
            found_count = count
            break
    attributes = object_transforms.get(found_obj_key, {})
    physics = physics_properties[count] 
    # Determine surface/material key
    surface_key = 'material' if 'material' in physics else 'surface'
    material_data = physics.get(surface_key, {})
    # Construct the full path to the GLB file
    obj_glb_path = os.path.join(asset_folder, found_obj_key + ".glb")
    if not os.path.exists(obj_glb_path):
        print(f"Warning: Replacement object GLB file not found at {obj_glb_path}. Cannot perform object replacement.")
        replace_obj_details = None # Cannot use this object if GLB is missing
    else:
        replace_obj_details = genesis_object(
            name=physics.get('object_name', 'unknown_object').replace("_target", "").replace("_destination", ""),
            file=obj_glb_path, # Use the constructed path
            quat=mat2quat(np.array(attributes.get("rotation", np.eye(3)))),
            scale=attributes.get('scale', 1.0),
            mass=physics.get('mass', 1.0),
            friction=physics.get('friction', 0.5),
            surface=material_data,
        )
        print(f"Identified replacement object: {replace_obj_details.name}")
    
    return replace_obj_details

# load from 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing Scene Generation')
    parser.add_argument('--config', type=str, default="configs/simpler.yaml", help='Path to the YAML config file')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--output_dir', type=str, default="./results", help='Output directory for results')
    parser.add_argument('--port', type=int, default=9010, help='Port for the server')
    parser.add_argument('--run_all', type=str2bool, default=False, help='Run all scenes or a specific one')
    args = parser.parse_args()
    num_runs = args.num_runs
    robot_args = {
        "name": "WidowX",
        "init_pos": torch.tensor(
            [-0.02014876, 0.04723017, 0.22625704, -0.00307271, 1.365988,
            -0.00168102, 0.037, 0.03699991], device='cuda:0'
        )
    }
    
    task = "pickuptask"
    camera_1_args = {
        "pos": (0.25, 0.25, 1.3),
        "lookat": (0.1, 0.1, 1.0),
        "fov": 80,
    }
    config = load_config(args.config)
    base_folder = config['base_folder']
    scene_name = config['scene_name']
    replace_name = config['replace_name']
    output_dir = args.output_dir
    port = args.port
    
    if args.run_all:
        scene_lists = os.listdir(os.path.join(base_folder, "bridge"))
    else:
        scene_lists = [scene_name]
        
    for scene_name in scene_lists:
        if scene_name.startswith("default"):
            default = True
        else:
            continue
        
        data_folder = os.path.join(base_folder, "bridge", scene_name)
        asset_folder = os.path.join(base_folder, "assets", scene_name)
        background = os.path.join(base_folder, "scene_background", scene_name, "background.png")
        extrinsics = np.load(os.path.join(data_folder, "extrinsics.npy"))
        intrinsics = np.load(os.path.join(data_folder, "intrinsics.npy"))
        
        with open(os.path.join(data_folder, "masks", "transformations.json"), 'r') as f:
            object_positions = json.load(f)
        
        if not os.path.exists(os.path.join(data_folder, "physical_properties.json")):
            with open(os.path.join(data_folder, "masks","result.json"), 'r') as f:
                physics_properties = json.load(f)
        else:
            with open(os.path.join(data_folder, "physical_properties.json"), 'r') as f:
                physics_properties = json.load(f)
        
        task_path = os.path.join(data_folder, "lang.txt")
        with open(task_path, 'r') as file:
            task_lines = file.readlines()
        task_lines = [line.strip() for line in task_lines]
        task_description = task_lines[0]
        
        replace_obj = get_replace_object(config)
        object_name = replace_obj.name
        
        target_name = None
        for asset in physics_properties:
            if "target" in asset["object_name"]:
                target_name = asset["object_name"].replace("_target", "")
        
        new_task_desc = task_description.replace(target_name, object_name)
        
        if replace_obj is not None:
            for i in range (num_runs):
                args = SimpleNamespace(
                    test_type = "object_replacement",
                    robot_args=robot_args,
                    background_path=background,
                    task_description=new_task_desc,
                    task=task,
                    camera_1_args=camera_1_args,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    asset_folder=asset_folder,
                    object_positions=object_positions,
                    object_properties=physics_properties,
                    test_id = i,
                    port = port,
                    scene_name = scene_name,
                    output_dir = os.path.join(output_dir, "asset_test", scene_name),
                    replace_object=replace_obj,  # Pass the replacement object
                )
                p = SimplerTest(args)
                p.run()
        else:
            print("No replacement object found. Skipping this test.")