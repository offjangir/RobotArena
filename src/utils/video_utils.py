import cv2
import os
from tqdm import tqdm
import re
import json
def create_video_from_images(folder_path, output_video, fps=30):
    """Generates a video file from a sequence of images in a folder and writes the path to a JSON file."""
    folder_path = os.path.abspath(folder_path)
    sequence_name = os.path.basename(folder_path)
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = sorted([
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.lower().endswith(valid_extensions)
    ], key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(0)))
    
    if not image_files:
        print(f"[ERROR] No image files found in '{folder_path}'.")
        return

    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    output_video = output_video.replace('.json','.mp4')  # Same name as video but with .json extension
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    for image_file in image_files:
        img = cv2.imread(image_file)
        video_writer.write(img)
    
    video_writer.release()
    print(f"[INFO] Video file created: {output_video}")
    
    data = { output_video.replace('.json','.mp4'): output_video.replace('.json','.mp4')  # Same name as video but with .json extension
}
    output_json = output_video.replace('.mp4', '.json')
    if not os.path.exists(output_json):
        with open(output_json, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[INFO] JSON file '{output_json}' created successfully.")
    else:
        print(f"[INFO] File '{output_json}' already exists. Overwriting it.")
        with open(output_json, 'w') as f:
            json.dump(data, f, indent=4)

    print(f"[INFO] JSON file created at '{output_json}' with the video: {output_video}")
