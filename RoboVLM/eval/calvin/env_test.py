import os

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"
import pyrender

from calvin_env.envs.play_table_env import get_env

path = "dataset/calvin_data/calvin_debug_dataset/validation"
env = get_env(path, show_gui=False)
print(env.get_obs())
