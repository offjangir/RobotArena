import cv2
import os
import yaml
import argparse
import json
import torch
import numpy as np
from types import SimpleNamespace
from src.pipeline.default_test import DefaulTest, load_config, str2bool


def blend_target_image_color_space(image, blend_value):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Ensure image is float for multiplication, then clip and convert to uint8
    blended_image = (1 - blend_value) * image.astype(np.float32) + blend_value * image_rgb.astype(np.float32)
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    return blended_image

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
        default = False
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
        if os.path.exists(os.path.join(output_dir,"adv_background_test", scene_name)):
            continue
        task_path = os.path.join(data_folder, "lang.txt")
        with open(task_path, 'r') as file:
            task_lines = file.readlines()
        task_lines = [line.strip() for line in task_lines]
        
        for task_description in task_lines:
            if "confidence" in task_description:
                continue
            background_image = cv2.imread(background)
            blend_values_sanity = [0.33, 0.66, 1.0]
            
            for val_sanity in blend_values_sanity:
                current_target_image_sanity = blend_target_image_color_space(
                    background_image.copy(), val_sanity)
                for i in range (1):
                    args = SimpleNamespace(
                        default=default,
                        robot_args=robot_args,
                        background = current_target_image_sanity,
                        task_description=task_description,
                        camera_1_args=camera_1_args,
                        intrinsics=intrinsics,
                        extrinsics=extrinsics,
                        asset_folder=asset_folder,
                        object_positions=object_positions,
                        object_properties=physics_properties,
                        test_id = f"{val_sanity:.2f}_{i}",
                        scene_name = scene_name,
                        port = port,
                        output_dir = os.path.join(output_dir, "adv_background_test", scene_name),
                    )
                    p = DefaulTest(args)
                    p.run()