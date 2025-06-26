import yaml
import os

# Define the path to the YAML file
config_path = os.path.join(os.path.dirname(__file__), "robot_config.yaml")

# Load the YAML content
with open(config_path, "r") as f:
    robot_config = yaml.safe_load(f)