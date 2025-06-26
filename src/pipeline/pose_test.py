
import numpy as np
from src.sim.simulator_genesis import SimulatorGenesis
import cv2
import os
import yaml
import argparse
import json
import random
import requests
import torch
from types import SimpleNamespace
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation
import genesis as gs
import itertools
from src.pipeline.default_test import DefaulTest, load_config, str2bool


class PoseTest(DefaulTest):
    def __init__(self, config):
        super().__init__(config)
    
    def generate_pose_deterministic(self, args):
        # ---------------- original code ----------------
        random.seed(0)

        x_values = np.linspace(self.args.test_args["asset1_pos_range_x"][0],
                            self.args.test_args["asset1_pos_range_x"][1],
                            self.args.test_args["num_samples"])
        y_values = np.linspace(self.args.test_args["asset1_pos_range_y"][0],
                            self.args.test_args["asset1_pos_range_y"][1],
                            self.args.test_args["num_samples"])
        rot_values_z = np.array(self.args.test_args["asset1_rot_range_z"])

        grid_points = [[x, y, self.args.test_args["z"]]     # z is still 0.9
                    for x in x_values for y in y_values]

        # unique position-pairs
        position_pairs = [(grid_points[i], grid_points[j])
                        for i in range(len(grid_points))
                        for j in range(i + 1, len(grid_points))]
        # sort all pairs by 2-D Euclidean distance
        position_pairs.sort(
            key=lambda p: ((p[0][0] - p[1][0]) ** 2 + (p[0][1] - p[1][1]) ** 2) ** 0.5
        )

        # pick four representative distances: 0 %, 25 %, 75 %, 100 %
        def pick(frac):
            idx = int(round(frac * (len(position_pairs) - 1)))
            return position_pairs[idx]

        sampled_position_pairs = [
            pick(0.4),   # a bit close
            pick(0.5),   # a bit close
            pick(0.6),   # a bit close
            pick(0.7),   # a bit close
            pick(0.8),   # a bit far
            pick(0.9),   # a bit far
            pick(1.0),   # a bit far
        ]
        # ----------------------------------------------------------

        # original rotation-pair logic (unchanged)
        rotation_pairs = [(rot_values_z[i], rot_values_z[j])
                        for i in range(len(rot_values_z))
                        for j in range(i + 1, len(rot_values_z))]

        sampled_rotation_pairs = random.sample(rotation_pairs,
                                            min(self.args.test_args["num_samples"],
                                                len(rotation_pairs)))
        return sampled_position_pairs, sampled_rotation_pairs
    
    def reset_benchmark(self, asset1_pose, asset1_rot, asset2_pose, asset2_rot):
        if self.target_asset is None:
            raise ValueError("target_asset and desti_asset must be set before calling reset_benchmark.")
        if self.desti_asset is None:
            self.target_asset.set_pos(torch.tensor(asset1_pose[:3]))  # Ensure asset1_pose has at least 3 elements
            asset1_full_rot = [90, 0, asset1_rot] 
            self.target_asset.set_quat(torch.tensor(euler2quat(*asset1_full_rot)))  # Pass combined Euler angles directly
        else:
            self.target_asset.set_pos(torch.tensor(asset1_pose[:3]))  # Ensure asset1_pose has at least 3 elements
            self.desti_asset.set_pos(torch.tensor(asset2_pose[:3]))  # Ensure asset2_pose has at least 3 elements
            asset1_full_rot = [90, 0, asset1_rot]   # Prepend [90, 0] to asset1_rot
            self.target_asset.set_quat(torch.tensor(euler2quat(*asset1_full_rot)))  # Pass combined Euler angles directly
        self.simulator.robot.set_qpos(self.args.robot_args["init_pos"])
        self.simulator.step()
        
    def run_pose_test(self):
        positions, rotations = self.generate_pose_deterministic(self.args)
        test_id = self.args.test_id
        pose_dict = {}
        for trial_id, (position, rotation) in enumerate(zip(positions, rotations)):
            self.setup()
            asset1_pose = position[0]
            asset2_pose = position[1]
            asset1_rot = rotation[0]
            asset2_rot = rotation[1]
            self.args.test_id = f"{test_id}_{trial_id}"
            pose_dict[trial_id] = {
                    "asset1_pose": asset1_pose,
                    "asset1_rot": asset1_rot,
                    "asset2_pose": asset2_pose,
                    "asset2_rot": asset2_rot,
            }
            
            self.reset_benchmark(asset1_pose, asset1_rot, asset2_pose, asset2_rot)
            self.run_default_test()
            gs.destroy()
        
        # Save the pose_dict to a JSON file
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"pose_test_{test_id}.json")
        pose_dict_converted = convert_to_builtin_type(pose_dict)
        with open(output_file, 'w') as f:
            json.dump(pose_dict_converted, f, indent=4)

def convert_to_builtin_type(obj):
    if isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_type(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
# load from 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing Scene Generation')
    parser.add_argument('--config', type=str, default="configs/default.yaml", help='Path to the YAML config file')
    parser.add_argument('--run_all', type=str2bool, default=False, help='Run all scenes or a specific one')
    parser.add_argument('--output_dir', type=str, default="./results", help='Output directory for results')
    parser.add_argument('--port', type=int, default=9010, help='Port for the server')
    
    
    parser.add_argument
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
    
    test_args = {
        "num_samples": 8,
        "rotation_samples_z": 5, 
        "z": 0.9,
        "asset1_pos_range_x": [-0.2, -0.1],
        "asset1_pos_range_y": [-0.1, 0.1],
        "asset1_rot_range_z": [0, 90, 225, 45],
    }
    
    config = load_config(args.config)
    base_folder = config['base_folder']
    scene_name = config['scene_name']
    port = args.port
    output_dir = args.output_dir
    
    if args.run_all:
        scene_lists = os.listdir(os.path.join(base_folder, "bridge"))
    else:
        scene_lists = [scene_name]
    for scene_name in scene_lists:
        if scene_name.startswith("default"):
            default = True
        elif scene_name.startswith("scene"):
            default = False
            continue
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
        
        for task_description in task_lines:
            if "confidence" in task_description:
                continue
            background_image = cv2.imread(background)
            for i in range (1):
                args = SimpleNamespace(
                    default=default,
                    robot_args=robot_args,
                    background = background_image,
                    task_description=task_description,
                    test_args=test_args,
                    camera_1_args=camera_1_args,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    asset_folder=asset_folder,
                    object_positions=object_positions,
                    object_properties=physics_properties,
                    test_id = i,
                    port = port,
                    scene_name = scene_name,
                    output_dir = os.path.join(output_dir, "pose_test", scene_name),
                )
                p = PoseTest(args)
                p.run_pose_test()