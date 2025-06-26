from PIL import Image
from vla import load_vla
import torch
import numpy as np
# Load the model
model = load_vla(
    'CogACT/CogACT-Base',                   # Options: CogACT-Small, -Base, -Large or local path
    load_for_training=False,
    action_model_type='DiT-B',              # Should match the chosen model
    future_action_window_size=15,
)

model.to('cuda:0').eval()

# Load your image
random_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
image = Image.fromarray(random_array)

prompt = "move sponge near apple"

# Predict Action (7-DoF)
actions, _ = model.predict_action(
    image,
    prompt,
    unnorm_key='fractal20220817_data',
    cfg_scale=1.5,
    use_ddim=True,
    num_ddim_steps=10,
)

print("Predicted actions:", actions)
