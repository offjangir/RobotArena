from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_image, load_depth, save_depth

DEVICE = 'cuda'
image_path = "/data/scene/data/scene_background/task_0044_user_0007_scene_0003_cfg_0005/background.png"
prompt_depth_path = "assets/example_images/arkit_depth.png"
image = load_image(image_path).to(DEVICE)
prompt_depth = load_depth(prompt_depth_path).to(DEVICE) # 192x256, ARKit LiDAR depth in meters

model = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vitl").to(DEVICE).eval()
depth = model.predict(image, prompt_depth) # HxW, depth in meters

save_depth(depth, prompt_depth=prompt_depth, image=image)