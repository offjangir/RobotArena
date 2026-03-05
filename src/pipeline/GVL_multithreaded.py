import cv2
import os
from PIL import Image
import base64
import google.generativeai as genai
import random
import shutil
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import os
from distutils.util import strtobool
import matplotlib.pyplot as plt
import textwrap
import multiprocessing


def encode_image(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return None
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image


PROGRESS_EVAL_PROMPT = """You are an expert roboticist. The robot is performing the task of '{instruction}'.
Your task is to evaluate the robot's progress towards completing the overall task by analyzing the provided initial/current scene states and robot action trajectory.

***** Input *****
This is the initial state of the scene.
(1) Visual Observations:
{init_visual_observations}
(2) Object Layout:
{init_object_information}
(3) Contact Information:
{init_contact_information}

From the initial state, the robot has made progress represented by the following trajectory:
{history_information}

Such progress results in the current state of the scene.
(1) Visual Observations:
{current_visual_observations}
(2) Object Layout:
{current_object_information}
(3) Contact Information:
{current_contact_information}

***** Output *****
Based on the above information, please do the following:
## Reflection
Evaluate the overall progress of the robot toward completing the task by analizing the trajectory and comparing initial and current scene state.
Are the subgoals reasonable? Is the robot making meaningful progress towards the final goal?

## Progress Score
Predict a task completion percentage between 0 and 100.
    - In the initial state, the task completion percentage is 0.
    - If all necessary steps are completed, the task completion percentage is 100.
    - The score should increase when a **meaningful** subgoal is completed.
    and should decrease if an action **undoes or degrades** a previously completed meaningful subgoal.
    - The score should also decrease if the robot's action causes SEVERE damage to other objects in the environment(Minor disturbance is acceptable and can be ignored).
    In this part, just provide one score and one concise explaining sentence.

Make sure you use '## Reflection' and '## Progress Score' as section headers in your response.
Please use EXACTLY ONE number between 0 and 100 for the Progress Score."""


def build_progress_eval_prompt(
    inference_frames_base64,
    first_frame,
    task_description,
    init_object_information="(See initial scene image)",
    init_contact_information="No contact in initial state.",
    current_object_information="(See current scene image)",
    current_contact_information="(Infer from current scene image)",
    history_information="",
):
    """
    Build a Gemini multimodal prompt by formatting PROGRESS_EVAL_PROMPT
    and interleaving images at the visual observation placeholders.
    """
    current_frame = inference_frames_base64[-1] if len(inference_frames_base64) > 1 else first_frame

    # Build trajectory image description
    trajectory_img_desc = ""
    if len(inference_frames_base64) > 2:
        trajectory_img_desc = f"(See {len(inference_frames_base64) - 2} trajectory frames below)"

    # Format the text prompt with all placeholders bound
    formatted_prompt = PROGRESS_EVAL_PROMPT.format(
        instruction=task_description,
        init_visual_observations="[Initial scene image attached]",
        init_object_information=init_object_information,
        init_contact_information=init_contact_information,
        history_information=history_information + ("\n" + trajectory_img_desc if trajectory_img_desc else ""),
        current_visual_observations="[Current scene image attached]",
        current_object_information=current_object_information,
        current_contact_information=current_contact_information,
    )

    # Split formatted prompt at image insertion points and interleave images
    parts = formatted_prompt.split("[Initial scene image attached]")
    prompt_parts = []

    # Part before initial image
    prompt_parts.append({"text": parts[0]})
    # Initial scene image
    prompt_parts.append({"mime_type": "image/jpeg", "data": first_frame})

    remainder = parts[1] if len(parts) > 1 else ""

    # Insert trajectory frames right after history_information section
    history_split = remainder.split("[Current scene image attached]")
    prompt_parts.append({"text": history_split[0]})

    # Trajectory frames (intermediate, chronological)
    if len(inference_frames_base64) > 2:
        prompt_parts.append({"text": "Trajectory frames (in chronological order):\n"})
        for j, frame_b64 in enumerate(inference_frames_base64[1:-1], start=1):
            prompt_parts.append({"text": f"Trajectory Frame {j}: [IMG]\n"})
            prompt_parts.append({"mime_type": "image/jpeg", "data": frame_b64})
        prompt_parts.append({"text": "\n"})

    # Current scene image
    prompt_parts.append({"mime_type": "image/jpeg", "data": current_frame})

    # Remainder after current image
    if len(history_split) > 1:
        prompt_parts.append({"text": history_split[1]})

    return prompt_parts


def generate_gemini_prompt_inference_only(inference_frames_base64, first_frame, task_description):
    """Legacy prompt: per-frame scoring with shuffled frames."""
    prompt_parts = []

    # Task description header
    prompt_parts.append({"text": f"You are an expert roboticist tasked to predict task completion\n"
                                 f"percentages for frames of a robot for the task of {task_description}.\n"
                                 f"The task completion percentages are between 0 and 100, where 100\n"
                                 f"corresponds to full task completion. Note that these frames are\n"
                                 f"in random order, so please pay attention to the individual frames\n"
                                 f"when reasoning about task completion percentage.\n"})
    prompt_parts.append({"text": "Initial robot scene: [IMG]\n"})
    prompt_parts.append({"mime_type": "image/jpeg", "data": first_frame})
    prompt_parts.append({"text": "In the initial robot scene, the task completion percentage is 0.\n"})    

    # Ground Truth Video Frames with scores and images
    prompt_parts.append({"text": f"Now, for the task of {task_description}, output the task completion\n"
                                 f"percentage for the following frames that are presented in random\n"
                                 f"order. Penalize the progression score if the robot just falls down and does not do anything or does random actions. You response will only contain output for each frame do not deviate from the output format, format your response as follows: "
                                 f" Frame {{i}}: Frame Description: {{What is the status describe}}, Task Completion Percentages: {{predicted_percentage}}%\n"})
    # Inference frames
    prompt_parts.append({"text": "\nNow, predict the task completion percentage for the following each inference frame:\n"})

    for j, inference_image in enumerate(inference_frames_base64):
        prompt_parts.append({"text": f"Inference Frame {j}: [IMG]\n"})
        prompt_parts.append({"mime_type": "image/jpeg", "data": inference_image})
        prompt_parts.append({"text": f"Predicted Task Completion Percentage for Inference Frame {j}: \n"})
    
    prompt_parts.append({"text": f"Predict for in total of {j} frames: \n"})
    return prompt_parts



def process_video_to_frames_and_data(video_path, num_frames_to_sample = 24, output_dir="temp_frames"):
    """Extracts all frames and their base64 encodings from a video."""

    frame_data = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return [], 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    num_frames_to_sample = int(duration)

    print(f"Total number of frames in video: {num_frames_to_sample}")
    if frame_count < num_frames_to_sample:
        print("Error: Video has fewer frames than requested samples.")
        exit()

    frame_indices = np.linspace(0, frame_count - 1, num=num_frames_to_sample, dtype=int)
    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if ret:
            base64_image = encode_image(frame)
            if base64_image:
                frame_data.append((base64_image, f"frame_{frame_index:04d}.jpg"))
            else:
                print(f"Error encoding frame {frame_index}. Skipping.")
        else:
            print(f"Warning: Could not read frame {frame_index}.")

    cap.release()
    return frame_data, frame_count


def generate_and_save_visualizations(
    original_images_b64,
    frame_indices,
    frame_scores,
    task_description,
    output_video_path,
    frames_save_dir,
    fps=5,
):
    """
    Generates a video AND saves individual frames in a side-by-side 
    Camera + Graph format as shown in the user's reference image.
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    os.makedirs(frames_save_dir, exist_ok=True)

    # Decode frames
    decoded_frames = []
    for img_b64 in original_images_b64:
        img_data = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        decoded_frames.append(cv2.imdecode(img_array, cv2.IMREAD_COLOR))

    num_frames = len(decoded_frames)
    h, w = decoded_frames[0].shape[:2]

    # Interpolate scores for a smooth graph
    all_scores = np.full(num_frames, np.nan)
    for idx, score in zip(frame_indices, frame_scores):
        all_scores[idx] = score
    if np.isnan(all_scores[0]): all_scores[0] = 0.0
    
    nans = np.isnan(all_scores)
    x_known = np.where(~nans)[0]
    y_known = all_scores[~nans]
    all_scores_interp = np.interp(np.arange(num_frames), x_known, y_known)

    # Define canvas (Camera Width + Graph Width)
    chart_w = 600  # Fixed width for the graph to ensure readability
    canvas_w = w + chart_w
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (canvas_w, h))

    for t in range(num_frames):
        current_score = all_scores_interp[t]
        
        # --- Create the Plot (Right Side) ---
        dpi = 100
        fig = Figure(figsize=(chart_w / dpi, h / dpi), dpi=dpi)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Style the plot to match the reference image
        ax.plot(np.arange(t + 1), all_scores_interp[:t + 1], color='blue', linestyle='--', marker='o', markersize=4)
        ax.set_xlim(0, num_frames)
        ax.set_ylim(0, 100)
        ax.set_title("Frame ID vs Task Completion Score", fontsize=12)
        ax.set_xlabel("Frame Progression", fontsize=10, fontweight='bold')
        ax.set_ylabel("Score (%)", fontsize=10, fontweight='bold')
        ax.grid(True, which='both', linestyle='-', alpha=0.5)

        fig.tight_layout()
        canvas.draw()
        
        # Convert Matplotlib figure to OpenCV image
        # chart_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        # chart_img = chart_img.reshape(h, chart_w, 3)
        # chart_bgr = cv2.cvtColor(chart_img, cv2.COLOR_RGB2BGR)
        # Convert Matplotlib figure to OpenCV image using the modern buffer_rgba
        rgba_buffer = canvas.buffer_rgba()
        chart_img = np.asarray(rgba_buffer)

        # chart_img is now (H, W, 4). Convert RGBA to BGR for OpenCV.
        chart_bgr = cv2.cvtColor(chart_img, cv2.COLOR_RGBA2BGR)

        # Ensure it fits your defined chart dimensions
        chart_bgr = cv2.resize(chart_bgr, (chart_w, h))

        # --- Combine Images ---
        combined = np.hstack((decoded_frames[t], chart_bgr))

        # 1. Save individual frame to the folder
        frame_filename = os.path.join(frames_save_dir, f"side_by_side_{t:04d}.jpg")
        cv2.imwrite(frame_filename, combined)

        # 2. Write to video
        out.write(combined)
        plt.close(fig)

    out.release()
    print(f"Frames saved to: {frames_save_dir}")


def generate_progress_video(
    original_images_b64,
    frame_indices,
    frame_scores,
    task_description,
    output_path,
    fps=5,
):
    """
    Generate a visualization video that shows each original frame
    alongside an animated progress-score line chart and progress bar.

    Args:
        original_images_b64: list of base64-encoded JPEG frames (in order)
        frame_indices: list of frame indices that have scores
        frame_scores: list of corresponding scores (0-100)
        task_description: task string for the title
        output_path: path to write the output .mp4 file
        fps: frames per second for the output video
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from io import BytesIO

    if len(frame_indices) == 0:
        print("No scores to visualize. Skipping video generation.")
        return

    # Decode all original frames
    decoded_frames = []
    for img_b64 in original_images_b64:
        img_data = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        decoded_frames.append(img)

    num_frames = len(decoded_frames)
    h, w = decoded_frames[0].shape[:2]

    # Interpolate scores for every frame
    all_scores = np.full(num_frames, np.nan)
    for idx, score in zip(frame_indices, frame_scores):
        all_scores[idx] = score
    # Fill score 0 for frame 0 if not present
    if np.isnan(all_scores[0]):
        all_scores[0] = 0.0
    # Forward-fill + linear interpolation
    nans = np.isnan(all_scores)
    x_known = np.where(~nans)[0]
    y_known = all_scores[~nans]
    all_scores_interp = np.interp(np.arange(num_frames), x_known, y_known)

    # Chart dimensions (right panel)
    chart_w = max(400, w // 2)
    chart_h = h
    canvas_w = w + chart_w

    import subprocess, tempfile
    tmp_avi = output_path.replace('.mp4', '_tmp.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(tmp_avi, fourcc, fps, (canvas_w, chart_h))

    for t in range(num_frames):
        # --- Left: original frame ---
        frame_bgr = decoded_frames[t].copy()
        current_score = all_scores_interp[t]

        # Draw progress bar on the original frame
        bar_x, bar_y = 10, h - 40
        bar_w_max = w - 20
        bar_h_px = 24
        # Background bar (dark gray)
        cv2.rectangle(frame_bgr, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + bar_h_px), (50, 50, 50), -1)
        # Filled bar (green-to-yellow gradient based on score)
        fill_w = int(bar_w_max * current_score / 100.0)
        g = int(255 * min(current_score / 50.0, 1.0))
        r = int(255 * max(0, (100 - current_score) / 50.0 - 1.0))
        bar_color = (0, max(100, g), min(255, 50 + int(current_score * 2)))
        cv2.rectangle(frame_bgr, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h_px), bar_color, -1)
        # Border
        cv2.rectangle(frame_bgr, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + bar_h_px), (200, 200, 200), 1)
        # Score text on bar
        cv2.putText(frame_bgr, f"Progress: {current_score:.0f}%", (bar_x + 5, bar_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # --- Right: animated chart ---
        dpi = 100
        fig = Figure(figsize=(chart_w / dpi, chart_h / dpi), dpi=dpi)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, num_frames - 1)
        ax.set_ylim(0, 105)
        ax.set_xlabel('Frame', fontsize=10)
        ax.set_ylabel('Progress Score (%)', fontsize=10)
        title_text = task_description if len(task_description) < 50 else task_description[:47] + '...'
        ax.set_title(f'Progress: {title_text}', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot scores up to current frame
        x_vals = np.arange(t + 1)
        y_vals = all_scores_interp[:t + 1]
        ax.fill_between(x_vals, y_vals, alpha=0.2, color='royalblue')
        ax.plot(x_vals, y_vals, color='royalblue', linewidth=2)
        # Current point
        ax.plot(t, current_score, 'o', color='red', markersize=8, zorder=5)
        ax.annotate(f'{current_score:.0f}%', (t, current_score),
                    textcoords='offset points', xytext=(8, 8),
                    fontsize=11, fontweight='bold', color='red')

        # Mark evaluated points that have been reached
        for fi, fs in zip(frame_indices, frame_scores):
            if fi <= t:
                ax.plot(fi, fs, 's', color='green', markersize=6, zorder=4)

        fig.tight_layout()
        canvas.draw()
        chart_img = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        chart_img = chart_img.reshape(int(fig.get_figheight() * dpi), int(fig.get_figwidth() * dpi), 4)
        chart_bgr = cv2.cvtColor(chart_img, cv2.COLOR_RGBA2BGR)
        chart_bgr = cv2.resize(chart_bgr, (chart_w, chart_h))
        plt.close(fig)

        # Combine left + right
        combined = np.zeros((chart_h, canvas_w, 3), dtype=np.uint8)
        combined[:h, :w] = frame_bgr
        combined[:chart_h, w:w + chart_w] = chart_bgr

        out.write(combined)

    out.release()
    # Re-encode to valid H.264 mp4 via ffmpeg
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', tmp_avi, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', '-crf', '23', output_path],
            check=True, capture_output=True,
        )
        os.remove(tmp_avi)
        print(f"Progress visualization video saved to: {output_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ffmpeg re-encode failed ({e}), keeping raw avi: {tmp_avi}")
        if os.path.exists(tmp_avi):
            os.rename(tmp_avi, output_path)


def draw_wrapped_text(img, text, pos, max_width, line_height=20, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(0, 255, 0), thickness=1):
    """Draw wrapped text on an image at the given position."""
    # Estimate character width (depends on font and scale)
    char_width = 10  # You can fine-tune this if needed
    max_chars_per_line = max_width // char_width

    wrapped_text = textwrap.wrap(text, width=max_chars_per_line)
    x, y = pos
    for i, line in enumerate(wrapped_text):
        y_offset = y + i * line_height
        cv2.putText(img, line, (x, y_offset), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return img

def load_rollout_states(npz_path, num_sampled_frames):
    """
    Load recorded object positions and robot (EE) states from a .npz file.

    Returns:
        rollout_data: dict with keys:
            'object_names' : list[str]  – names of tracked objects
            'ee_states'    : np.ndarray (T, 8) – [pos(3), quat(4), gripper(1)]
            'object_states': dict[str -> np.ndarray (T, 13)]
                             Each row: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        or None if the file does not exist.
    The arrays are resampled to *num_sampled_frames* so that index *i*
    corresponds to sampled video frame *i*.
    """
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    reserved = {'world', 'base'}
    object_names = [k for k in data.keys() if k not in reserved]

    T = data['world'].shape[0]  # original number of timesteps
    # Build indices that map each sampled frame to the nearest original step
    sample_indices = np.linspace(0, T - 1, num=num_sampled_frames, dtype=int)

    rollout = {
        'object_names': object_names,
        'ee_states': data['world'][sample_indices],       # (N, 8)
        'object_states': {k: data[k][sample_indices] for k in object_names},
    }
    return rollout


def format_object_info(rollout, frame_idx, include_robot=True, include_objects=True):
    """
    Build a human-readable string describing every object's position at a
    given frame index, plus the robot end-effector state.
    """
    if rollout is None:
        return "(No recorded state data available)"
    lines = []
    # Robot EE
    if include_robot:
        ee = rollout['ee_states'][frame_idx]
        pos, quat, grip = ee[:3], ee[3:7], ee[7]
        grip_str = "open" if grip > 0 else "closed"
        lines.append(f"Robot end-effector: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                     f"quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}), "
                     f"gripper={grip_str}")
    # Objects
    if include_objects:
        for name in rollout['object_names']:
            obj = rollout['object_states'][name][frame_idx]
            p, q, v = obj[:3], obj[3:7], obj[7:10]
            speed = np.linalg.norm(v)
            moving = f", moving (speed={speed:.3f})" if speed > 0.01 else ""
            lines.append(f"{name}: pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), "
                         f"quat=({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}){moving}")
    if not lines:
        return "(No state data provided for this ablation)"
    return "\n".join(lines)


def format_contact_info(rollout, frame_idx, contact_dist_thresh=0.05):
    """
    Heuristic contact detection: if the EE is within *contact_dist_thresh*
    of an object's position, report it as a likely contact / grasp.
    """
    if rollout is None:
        return "(No recorded state data available)"
    ee_pos = rollout['ee_states'][frame_idx][:3]
    grip_val = rollout['ee_states'][frame_idx][7]
    contacts = []
    for name in rollout['object_names']:
        obj_pos = rollout['object_states'][name][frame_idx][:3]
        dist = np.linalg.norm(ee_pos - obj_pos)
        if dist < contact_dist_thresh:
            if grip_val <= 0:
                contacts.append(f"Gripper is closed near '{name}' (dist={dist:.3f}m) – likely grasping.")
            else:
                contacts.append(f"End-effector is very close to '{name}' (dist={dist:.3f}m).")
    if not contacts:
        return "No detected contact between robot and objects."
    return "\n".join(contacts)


def format_history_info(rollout, start_idx, end_idx):
    """
    Summarise the robot's trajectory between two frame indices.
    """
    if rollout is None:
        return f"The robot has executed {end_idx} steps from the initial state."
    ee_start = rollout['ee_states'][start_idx][:3]
    ee_end = rollout['ee_states'][end_idx][:3]
    displacement = np.linalg.norm(ee_end - ee_start)
    grip_changes = 0
    for t in range(start_idx + 1, end_idx + 1):
        if (rollout['ee_states'][t][7] > 0) != (rollout['ee_states'][t - 1][7] > 0):
            grip_changes += 1
    lines = [
        f"The robot has executed {end_idx - start_idx} steps.",
        f"End-effector moved {displacement:.3f}m total displacement.",
        f"Gripper state changed {grip_changes} time(s) during this period.",
    ]
    # Report which objects moved significantly
    for name in rollout['object_names']:
        obj_s = rollout['object_states'][name][start_idx][:3]
        obj_e = rollout['object_states'][name][end_idx][:3]
        d = np.linalg.norm(obj_e - obj_s)
        if d > 0.01:
            lines.append(f"'{name}' moved {d:.3f}m from ({obj_s[0]:.3f},{obj_s[1]:.3f},{obj_s[2]:.3f}) "
                         f"to ({obj_e[0]:.3f},{obj_e[1]:.3f},{obj_e[2]:.3f}).")
    return "\n".join(lines)


def shuffle(inference_frames_base64):
    original_indices = list(range(len(inference_frames_base64)))
    shuffled = list(zip(original_indices, inference_frames_base64))
    random.shuffle(shuffled)
    shuffled_indices, shuffled_frames_base64 = zip(*shuffled)
    return shuffled_indices, shuffled_frames_base64


def zero_shot(args):
    import re
    import os
    import copy

    inference_video_file = args.inference_video_file
    api_key = args.key 
    video_name = os.path.basename(inference_video_file).replace(".mp4", "")
    scene_name = os.path.basename(os.path.dirname(inference_video_file))
    base_dir = args.base_dir
    task_file_path = os.path.join(base_dir, scene_name, "lang.txt")
    with open(task_file_path, "r") as f:
        task = f.read().strip()

    save_dir = os.path.join(args.dir, args.policy, args.test, scene_name, "zero_shot", video_name)
    annotated_dir = os.path.join(save_dir, "annotated_inference_frames")
    os.makedirs(annotated_dir, exist_ok=True)
    save_score_path = os.path.join(save_dir, "save_scores.json")
    print("---------------")
    print(f"Task: {task}")
    print(f"Video Name: {video_name}")
    print(f"Scene Name: {scene_name}")
    print(f"Save Directory: {save_score_path}")
    print(f"Annotated Directory: {annotated_dir}")
    print("---------------")

    if os.path.exists(save_score_path):
        print(f"Score json already exists: {save_score_path}")
        print("Exiting...")
        return

    inference_frame_data, _ = process_video_to_frames_and_data(inference_video_file, output_dir="inference_frames_temp")
    original_images = [img for img, _ in inference_frame_data]
    inference_frames_base64 = [img for img, _ in inference_frame_data]
    first_frame = inference_frames_base64[0]

    # Load recorded object positions & robot state from companion .npz
    npz_path = inference_video_file.replace(".mp4", ".npz")
    rollout = load_rollout_states(npz_path, num_sampled_frames=len(inference_frames_base64))
    if rollout is not None:
        print(f"Loaded rollout data from {npz_path} "
              f"({len(rollout['object_names'])} objects, {rollout['ee_states'].shape[0]} steps)")
    else:
        print(f"No .npz found at {npz_path} – using image-only evaluation")

    save_scores = {}
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=args.model)

    # Evaluate progress at every sampled frame (like GVL_multithreaded.py)
    num_frames = len(inference_frames_base64)
    eval_points = list(range(1, num_frames))

    for i in range(args.frequency):
        save_scores[i] = {}
        frame_indices = []
        frame_scores = []
        frame_reflections = []

        for eval_idx in eval_points:
            # Build trajectory frames up to eval_idx
            trajectory_frames = inference_frames_base64[:eval_idx + 1]
            current_frame = inference_frames_base64[eval_idx]

            # Build state descriptions from recorded NPZ data
            # Respect ablation mode
            ablation = getattr(args, 'ablation', 'full')
            use_obj = ablation in ('full', 'no_robot') and rollout is not None
            use_robot = ablation in ('full', 'no_object') and rollout is not None
            use_any = use_obj or use_robot

            if use_obj and use_robot:
                init_obj_info = format_object_info(rollout, 0)
                cur_obj_info = format_object_info(rollout, eval_idx)
            elif use_obj:  # no_robot: only object lines
                init_obj_info = format_object_info(rollout, 0, include_robot=False)
                cur_obj_info = format_object_info(rollout, eval_idx, include_robot=False)
            elif use_robot:  # no_object: only robot lines
                init_obj_info = format_object_info(rollout, 0, include_objects=False)
                cur_obj_info = format_object_info(rollout, eval_idx, include_objects=False)
            else:
                init_obj_info = "(See initial scene image above)"
                cur_obj_info = "(See current scene image above)"

            init_contact = format_contact_info(rollout, 0) if use_any else "No contact in initial state."
            cur_contact = format_contact_info(rollout, eval_idx) if use_any else "(Infer from current scene image)"
            history_info = format_history_info(rollout, 0, eval_idx) if use_any else f"The robot has executed {eval_idx} steps from the initial state."

            prompt = build_progress_eval_prompt(
                inference_frames_base64=trajectory_frames,
                first_frame=first_frame,
                task_description=task,
                init_object_information=init_obj_info,
                init_contact_information=init_contact,
                current_object_information=cur_obj_info,
                current_contact_information=cur_contact,
                history_information=history_info,
            )

            try:
                response = model.generate_content(prompt)
                response_text = response.text
            except Exception as e:
                print(f"Error generating content at frame {eval_idx}: {e}")
                try:
                    response = model.generate_content(prompt)
                    response_text = response.text
                except Exception as e2:
                    print(f"Retry also failed: {e2}")
                    continue

            # Parse Progress Score from response
            score_match = re.search(r'##\s*Progress\s*Score.*?(\d+)', response_text, re.DOTALL | re.IGNORECASE)
            reflection_match = re.search(r'##\s*Reflection\s*(.*?)(?=##\s*Progress\s*Score)', response_text, re.DOTALL | re.IGNORECASE)

            if score_match:
                completion_pct = float(score_match.group(1))
                completion_pct = min(100, max(0, completion_pct))
                frame_indices.append(int(eval_idx))
                frame_scores.append(completion_pct)
                reflection = reflection_match.group(1).strip() if reflection_match else ""
                frame_reflections.append(reflection)
                print(f"Frame {eval_idx}: Progress Score: {completion_pct:.0f}%")
            else:
                print(f"Could not parse Progress Score from response at frame {eval_idx}")
                print(f"Response: {response_text[:200]}...")

        if args.debug:
            trial_annotation_path = os.path.join(annotated_dir, f'{i}')
            os.makedirs(trial_annotation_path, exist_ok=True)
            for idx, img_b64 in enumerate(original_images):
                img_data = base64.b64decode(img_b64)
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                annotated_img = img.copy()
                # Find the closest eval score for this frame
                score_text = "N/A"
                if idx in frame_indices:
                    score_idx = frame_indices.index(idx)
                    score_text = f"{frame_scores[score_idx]:.0f}%"
                text = f"Progress: {score_text}"
                annotated_img = draw_wrapped_text(annotated_img, text, pos=(10, 30), max_width=200)
                cv2.imwrite(os.path.join(trial_annotation_path, f"frame_{idx:03d}.jpg"), annotated_img)

        # Plot and save
        if len(frame_indices) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(frame_indices, frame_scores, marker='o', linestyle='-', color='blue')
            plt.title(f"Progress Score - {task}")
            plt.xlabel("Frame ID")
            plt.ylabel("Progress Score (%)")
            plt.ylim(0, 100)
            plt.grid(True)
            plot_path = os.path.join(save_dir, f"frame_vs_score_{i}.png")
            print(f"Plot saved to: {plot_path}")
            plt.savefig(plot_path)
            plt.close()

        save_scores[i]['frame_indices'] = frame_indices
        save_scores[i]['frame_scores'] = frame_scores
        save_scores[i]['reflections'] = frame_reflections

    os.makedirs(save_dir, exist_ok=True)
    with open(save_score_path, "w") as f:
        import json
        json.dump(save_scores, f, indent=4)

    # Generate progress visualization video for the last trial
    last_trial = max(save_scores.keys(), key=lambda k: int(k))
    # if 'frame_indices' in save_scores[last_trial] and len(save_scores[last_trial]['frame_indices']) > 0:
    #     viz_video_path = os.path.join(save_dir, "progress_visualization.mp4")
    #     generate_progress_video(
    #         original_images_b64=original_images,
    #         frame_indices=save_scores[last_trial]['frame_indices'],
    #         frame_scores=save_scores[last_trial]['frame_scores'],
    #         task_description=task,
    #         output_path=viz_video_path,
    #         fps=5,
    #     )
    if 'frame_indices' in save_scores[last_trial] and len(save_scores[last_trial]['frame_indices']) > 0:
        viz_video_path = os.path.join(save_dir, "progress_visualization.mp4")
        
        # NEW: Folder for individual styled frames
        frames_out_dir = os.path.join(save_dir, "styled_frames")
        
        generate_and_save_visualizations(
            original_images_b64=original_images,
            frame_indices=save_scores[last_trial]['frame_indices'],
            frame_scores=save_scores[last_trial]['frame_scores'],
            task_description=task,
            output_video_path=viz_video_path,
            frames_save_dir=frames_out_dir, # Pass the new folder path
            fps=5,
        )

import copy
def worker(chunk, args):
    for video in chunk:
        local_args = copy.deepcopy(args)
        local_args.inference_video_file = video
        zero_shot(local_args)
    
if __name__ == '__main__':
    # Example usage:
    parser = argparse.ArgumentParser(description='Processing Scene Generation')
    parser.add_argument('--inference', type=str, help='Inference Folder of Test')
    parser.add_argument('--base_dir', type=str, help='bridge_base_dir')

    parser.add_argument('--task', type=str, help='Task Description')
    parser.add_argument('--key', type=str, help='API Key')
    parser.add_argument('--zero', type=lambda x: bool(strtobool(x)), help='single_shot or zero_shot')
    parser.add_argument('--frequency', type=int, default=1, help='Frequency value')
    parser.add_argument('--dir', type=str, help='Directory path')
    parser.add_argument('--scene', type=str, help='Scene name')
    parser.add_argument('--take_top_50', type=lambda x: bool(strtobool(x)), help='Whether to take top 50%')
    parser.add_argument('--test', type=str, help='Test mode')
    parser.add_argument('--policy', type=str, help='Policy name')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), help='Enable debug mode')
    parser.add_argument('--ablation', type=str, default='full',
                        choices=['full', 'no_object', 'no_robot', 'none'],
                        help='Ablation mode: full=all info, no_object=no object positions, '
                             'no_robot=no robot EE state, none=image-only (old prompt behavior)')
    args = parser.parse_args()
    import os
    import glob
    num_cores = 64
    video_files = []
    for subdir in os.listdir(args.inference):
        full_path = os.path.join(args.inference, subdir)
        if os.path.isdir(full_path):
            videos = glob.glob(os.path.join(full_path, '*.mp4'))
            video_files.extend(videos)

    # Filter to a single scene if --scene is specified
    if args.scene:
        video_files = [v for v in video_files if os.path.basename(os.path.dirname(v)) == args.scene]

    # Sort by the immediate subdirectory name
    video_files.sort(key=lambda x: os.path.basename(os.path.dirname(x)))
    # Create empty lists for each core
    core_chunks = [[] for _ in range(num_cores)]

    # Assign videos in round-robin style
    for idx, video in enumerate(video_files):
        core_idx = idx % num_cores
        core_chunks[core_idx].append(video)
    
    for chunk in core_chunks:
        processes = []
        p = multiprocessing.Process(target=worker, args=(chunk,args))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()