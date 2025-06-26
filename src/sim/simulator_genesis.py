import genesis as gs
import numpy as np
import torch
import trimesh
from configs import robot_config

def process_surface(surface):
    match surface:
        case "Default":
            return gs.surfaces.Default()
        case "Glass":
            return gs.surfaces.Glass()
        case "Water":
            return gs.surfaces.Water()
        case "Emission":
            return gs.surfaces.Emission()
        case "Plastic":
            return gs.surfaces.Plastic()
        case "Rough":
            return gs.surfaces.Rough()
        case "Smooth":
            return gs.surfaces.Smooth()
        case "Reflective":
            return gs.surfaces.Reflective()
        case "Metal":
            return gs.surfaces.Metal()
        case "Iron":
            return gs.surfaces.Iron()
        case "Aluminium":
            return gs.surfaces.Aluminium()
        case "Copper":
            return gs.surfaces.Copper()
        case "Gold":
            return gs.surfaces.Gold()
        case _:
            return gs.surfaces.Default()
        
class SimulatorGenesis:
    def __init__(self, robot_name, envs, add_robot=None, device=None, show_viewer=False, seed=None):
        self.robot_name = robot_name
        self.robot = None
        self.envs = envs
        self.seed = seed
        self.asset_path = None
        self.device = device
        self.show_viewer = show_viewer
        self.robot_addition = add_robot
        self.asset_ID = {}
    
    def get_asset_ID(self):
        return self.asset_ID

    def asset_addtion(self, asset_path, pos = None, 
                      quat = None, scale = None, 
                      physics = None, fixed = False):
        
        if pos is None:
            pos = (1, 0.5, 1.0)
        if quat is None:
            quat = (0.717, 0.717, 0, 0)
        if scale is None:
            scale = 0.12
        if physics is None:
            asset = (self.scene.add_entity(
            gs.morphs.Mesh(file=asset_path,
                        fixed=fixed,
                        pos= pos,
                        quat= quat,
                        scale= scale,
                        ),
            ),
            None,
            None,
            )
        else:
            if 'material' not in physics:
                surface = process_surface(physics["surface"])
            else:
                surface = process_surface(physics["material"])
            asset = (self.scene.add_entity(
            gs.morphs.Mesh(file=asset_path,
                        fixed=fixed,
                        pos= pos,
                        quat= quat,
                        scale= scale,
                        convexify=True,
                        ),
                surface = surface
            ),
            physics["mass"],
            physics["friction"],
            )
        self.asset_ID[scale] = asset
        return scale

    def set_scene(self, default, special_light = False):
        # Initializes the simulation scene with predefined viewer and physics options.
        viewer_options = gs.options.ViewerOptions(
            camera_pos=(3, -1, 2.5),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=30,
            max_FPS=60,
        )
        if special_light:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=0.005),
                viewer_options=viewer_options,
                vis_options=gs.options.VisOptions(
                        show_world_frame=False,
                        shadow=False,
                        lights = [{
                            'type': 'directional',
                            'dir': (-0.75, -0.5, -2.0),
                            'color': (1.0, 1.0, 1.0),
                            'intensity': 6.0
                        }]
                    ),
                rigid_options=gs.options.RigidOptions(dt=0.005),
                show_viewer=self.show_viewer,
                show_FPS=True,
            )
        else:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=0.005),
                viewer_options=viewer_options,
                vis_options=gs.options.VisOptions(
                        show_world_frame=False,
                        shadow=False,
                        ambient_light=(0.4, 0.4, 0.4),
                        # lights = [{'type': 'directional', 'dir': (-1, -1, -1), 'color': (1.0, 1.0, 1.0), 'intensity': 5.0}],
                    ),
                rigid_options=gs.options.RigidOptions(dt=0.005),
                show_viewer=self.show_viewer,
                show_FPS=True,
            )
        if default == True:
            plane = self.scene.add_entity(
                gs.morphs.Plane(
                    pos =(0, 0, 0.88),
                    visualization=False,
                )
            )
        else:
            plane = self.scene.add_entity(
                gs.morphs.Plane(
                    visualization=False,
                    )
            )
        return self.scene
    
    def add_robot(self, default=False):
        # Loads and adds a robot to the scene based on its configuration.
        robot_path = robot_config[self.robot_name]["path"]
        import os
        ext = os.path.splitext(robot_path)[1].lower()
        pos = robot_config[self.robot_name]["pos"]
        euler = robot_config[self.robot_name]["quat"]
        if ext == ".xml":
            self.robot = self.scene.add_entity(
                gs.morphs.MJCF(file=robot_path, 
                               pos= pos,
                               quat=euler,
                               )
            )
        elif ext == ".urdf":
            if default == True:
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file=robot_path,
                        fixed=True,
                        merge_fixed_links=True,
                        links_to_keep=[robot_config[self.robot_name]["ee_link"]],
                        pos=pos,
                        quat=euler,
                    )
                )
            else:
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file=robot_path,
                        fixed=True,
                        merge_fixed_links=True,
                        links_to_keep=[robot_config[self.robot_name]["ee_link"]],
                    )
                )
        else:
            return "Unknown file type"
        
        return self.robot
    
    def apply_pd_gains(self):
        """
        Applies PD gains to the robot's joints.
        This method should be called after the scene is built.
        """
        
        best_kp = np.array([4345, 2019, 3184, 4424, 3140, 10000, 100, 100])
        best_kv = np.array([224, 252, 593, 10, 893, 302, 10, 10])
        self.robot.set_dofs_kp(best_kp)
        self.robot.set_dofs_kv(best_kv)
    
    def gs_transform_by_quat(self, pos, quat):
        # Transforms a position vector using quaternion rotation.
        qw, qx, qy, qz = quat.unbind(-1)
        rot_matrix = torch.stack([
            1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
            2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
            2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2
        ], dim=-1).reshape(*quat.shape[:-1], 3, 3)
        
        return torch.matmul(rot_matrix, pos.unsqueeze(-1)).squeeze(-1)

    def transform_point_cloud(self, point_cloud, translation, quaternion):
        # Applies rotation and translation to a point cloud.
        rotated_points = self.gs_transform_by_quat(point_cloud, quaternion)
        return rotated_points + translation

    def control_ee_pose(self, target_pos, target_quat):
        # Moves the end-effector to a target position using inverse kinematics.
        q = self.robot.inverse_kinematics(
            link=self.robot.get_link(robot_config[self.robot_name]["ee_link"]),
            pos=target_pos,
            quat=target_quat,
        )
        self.robot.control_dofs_position(q)

    def start_sim(self, default=False, special_light = False):
        # Initializes simulation and sets up the environment.
        gs.init(backend=gs.gpu)
        self.set_scene(default=default, special_light=special_light)
        if self.robot_addition:
            self.add_robot(default=default)
    
    def set_camera(self):
        # Adds a camera to the scene with predefined parameters.
        self.scene.add_camera(
            pos=(2.5, -0.15, 2.42),
            lookat=(0.5, 0.5, 0.1),
            res=(1280, 720),
            fov=30,
            GUI=False,
        )

    def visualize_pc(self):
        # Displays the latest point cloud as debug spheres.
        self.scene.clear_debug_objects()
        for pc in self.latest_pc:
            self.scene.draw_debug_spheres(poss=pc, radius=0.001, color=(0.8, 0.8, 0, 0.7))

    def step(self):
        # Advances the simulation by one step.
        self.scene.step()