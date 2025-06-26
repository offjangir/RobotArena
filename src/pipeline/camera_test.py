import cv2
import os
import yaml
import argparse
import json
import torch
import numpy as np
from types import SimpleNamespace
from src.utils.test_utils import get_genesis_extrinsics, perturb_pos_lookat_rigid
from src.pipeline.default_test import DefaulTest, load_config, str2bool
import genesis as gs

class CameraTest(DefaulTest):
    def __init__(self, config):
        super().__init__(config)
    
    def camera_setup(self):
        self.setup()
        self.camera_pos = self.camera_0.pos
        self.camera_lookat = self.camera_0.lookat
        gs.destroy()

# load from 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing Scene Generation')
    parser.add_argument('--config', type=str, default="configs/default.yaml", help='Path to the YAML config file')
    parser.add_argument('--run_all', type=str2bool, default=False, help='Run all scenes or a specific one')
    parser.add_argument('--output_dir', type=str, default="./results", help='Output directory for results')
    parser.add_argument('--port', type=int, default=9010, help='Port for the server')
    
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
    output_dir = args.output_dir
    port = args.port
    
    run_default = False
    if ("default" in output_dir):
        run_default = True
    
    if args.run_all:
        scene_lists = os.listdir(os.path.join(base_folder, "bridge"))
    else:
        scene_lists = [scene_name]
    for scene_name in scene_lists:
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
                    camera_1_args=camera_1_args,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    asset_folder=asset_folder,
                    object_positions=object_positions,
                    object_properties=physics_properties,
                    test_id = i,
                    port = port,
                    scene_name = scene_name,
                    output_dir = os.path.join( output_dir ,"camera_test", scene_name),
                )
                p = CameraTest(args)
                p.camera_setup()
                # perturb the camera position and lookat
                inital_extrinsics, fov, W, H = get_genesis_extrinsics(args.extrinsics, args.intrinsics)
                perturb = perturb_pos_lookat_rigid(p.camera_pos, p.camera_lookat, distance_cm=5)
                
                cnt = 0
                for key, value in perturb.items():
                    args.fov = fov
                    args.W = W
                    args.H = H
                    args.test_id = f"{i}_{key}"
                    args.camera_pos = value[0]
                    args.camera_lookat = value[1]
                    p = DefaulTest(args)
                    p.run()
                    cnt += 1
                    
            