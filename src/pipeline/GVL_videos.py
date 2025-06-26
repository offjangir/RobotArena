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

def encode_image(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return None
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

def generate_gemini_prompt_for_inference(gt_frame_data, inference_frames_base64, first_frame, task_description):
    prompt_parts = []

    # Task description header
    prompt_parts.append({"text": f"You are an expert roboticist tasked to predict task completion\n"
                                 f"percentages for frames of a robot for the task of {task_description}.\n"
                                 f"The task completion percentages are between 0 and 100, where 100\n"
                                 f"corresponds to full task completion. We provide several examples of\n"
                                 f"the robot performing the task at various stages and their\n"
                                 f"corresponding task completion percentages. Note that these frames are\n"
                                 f"in random order, so please pay attention to the individual frames\n"
                                 f"when reasoning about task completion percentage. Next we give the examples followed by initial frame\n\n"})
    for i, (base64_image, score) in enumerate(gt_frame_data):
        prompt_parts.append({"text": f"Frame {i}: Frame: [IMG], Task Completion Percentages: {score:.2f}%\n"})
        prompt_parts.append({"mime_type": "image/jpeg", "data": base64_image})


    prompt_parts.append({"text": "Initial robot scene: [IMG]\n "})
    prompt_parts.append({"mime_type": "image/jpeg", "data": first_frame})
    prompt_parts.append({"text": "In the initial robot scene, the task completion percentage is 0.\n"})    

    prompt_parts.append({"text": f"Now, for the task of {task_description}, output the task completion\n"
                                 f"percentage for the following frames that are presented in random\n"
                                 f"order. Make sure to not predict highly deviating values. You response will only contain output for each frame do not deviate from the structure, format your response as follows:"
                                 f" Frame {{i}}: Frame Description: {{What is the status describe}}, Task Completion Percentages: {{predicted_percentage}}%\n"})
# Frame Description: {{What is the status describe}}.
    prompt_parts.append({"text": "\nNow, predict the task completion percentage for the following inference frames(Do not give any empty or None answers):\n"})

    for j, inference_image in enumerate(inference_frames_base64):
        prompt_parts.append({"text": f"Inference Frame {j}: [IMG]\n"})
        prompt_parts.append({"mime_type": "image/jpeg", "data": inference_image})
        prompt_parts.append({"text": f"Predicted Task Completion Percentage for Inference Frame {j}: \n"})
    prompt_parts.append({"text": f"Predict for in total of {len(inference_frames_base64)-1} frames i.e your output should have in total {len(inference_frames_base64)-1} frames \n"})
    return prompt_parts


def generate_gemini_prompt_inference_only(inference_frames_base64,first_frame,task_description):
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
                                 f"order. You response will only contain output for each frame do not deviate from the output format, format your response as follows: "
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

    print(f"Total number of frames in video: {frame_count}")
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

import textwrap

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
    gt_video_file = args.gt  
    inference_video_file = args.inference 
    task = args.task 
    api_key = args.key 

    video_name = os.path.basename(inference_video_file.replace('.mp4', ''))
    save_dir = os.path.join(args.dir, args.policy, args.test, args.scene, "zero_shot", video_name)
    annotated_dir = os.path.join(save_dir, "annotated_inference_frames")
    os.makedirs(annotated_dir, exist_ok=True)
    save_score_path = os.path.join(save_dir, "save_scores.json")

    inference_frame_data, _ = process_video_to_frames_and_data(inference_video_file, output_dir="inference_frames_temp")
    original_images = [img for img, _ in inference_frame_data]
    inference_frames_base64 = [img for img, _ in inference_frame_data]
    first_frame = inference_frames_base64[0]
    save_scores = {}
    average_scores = []
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=args.model)
    for i in range(args.frequency):
        save_scores[i] = {}

        shuffled_indices,shuffled_frames_base64 = shuffle(inference_frames_base64)
        prompt = generate_gemini_prompt_inference_only(list(shuffled_frames_base64), first_frame, task)
        try: 
            response = model.generate_content(prompt)
            raw_scores = response.text.splitlines()  # Adjust depending on response format

        except Exception as e:
            print(f"Error generating content: {e}")
            try:
                response = model.generate_content(prompt)
                raw_scores = response.text.splitlines()  # Adjust depending on response format

            except Exception as e:
                continue
            

        save_scores[i]['response'] = response.text
        save_scores[i]['raw_scores'] = raw_scores
        save_scores[i]['shuffled_inference_indices'] = shuffled_indices
        remapped_scores = [None] * len(raw_scores)
        for idx, original_idx in enumerate(shuffled_indices):
            try: 
                remapped_scores[original_idx] = raw_scores[idx]
            except IndexError:
                #  restart the loop if there is an index error
                print(f"IndexError: {idx} is out of range for remapped_scores")

                continue

        os.makedirs(save_dir, exist_ok=True)
        frame_indices = []
        frame_scores = []

        for idx, (img_b64, score) in enumerate(zip(original_images, remapped_scores)):
            text = copy.deepcopy(score)
            try: 
                frame_no_match = re.search(r'Frame (\d+):', score)
                completion_pct_match = re.search(r'Task Completion Percentages: ([\d.]+)%', score)
                if frame_no_match and completion_pct_match:
                    frame_no = int(frame_no_match.group(1))
                    completion_pct = float(completion_pct_match.group(1))
                    frame_indices.append(idx)
                    frame_scores.append(completion_pct)
                    print(f"Frame {idx}: Task Completion Percentage: {completion_pct:.2f}%")
            except Exception as e:
                print(f"Error processing frame {idx}: {e}")
                i -= 1
                continue

        if args.debug:
            trial_annotation_path =os.path.join(annotated_dir,f'{i}')
            os.makedirs(trial_annotation_path, exist_ok=True)
            for idx, (img_b64, score) in enumerate(zip(original_images, remapped_scores)):
                img_data = base64.b64decode(img_b64)
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                annotated_img = img.copy()
                text = f"Score: {score}"
                annotated_img = draw_wrapped_text(annotated_img, text, pos=(10, 30), max_width=200)
                cv2.imwrite(os.path.join(trial_annotation_path, f"frame_{idx:03d}.jpg"), annotated_img)
        # Plot and save
        plt.figure(figsize=(10, 5))
        plt.plot(frame_indices, frame_scores, marker='o', linestyle='-', color='blue')
        plt.title("Frame ID vs Task Completion Score")
        plt.xlabel("Frame ID")
        plt.ylabel("Score (%)")
        plt.ylim(0, 100)
        plt.grid(True)

        plot_path = os.path.join(save_dir, f"frame_vs_score_{i}.png")
        print(f"Plot saved to: {plot_path}")
        plt.savefig(plot_path)
        plt.close()
        save_scores[i]['frame_indices'] = frame_indices
        save_scores[i]['frame_scores'] = frame_scores
    
    # Calculate Average of average scores
    with open(save_score_path, "w") as f:
        import json
        json.dump(save_scores, f, indent=4)

def shuffle_one_shot(gt_processed_data):
    original_indices_gt = list(range(len(gt_processed_data)))
    shuffled = list(zip(original_indices_gt, gt_processed_data))
    random.shuffle(shuffled)
    original_indices, shuffled_frames_base64_gt = zip(*shuffled)
    return original_indices, shuffled_frames_base64_gt  

def one_shot(args):
    import re
    import os
    import copy
    gt_video_file = args.gt  
    inference_video_file = args.inference 
    task = args.task 
    api_key = args.key 
    video_name = os.path.basename(inference_video_file).split('.')[0]
    save_dir = os.path.join(args.dir, args.policy, args.test, args.scene, "one_shot", video_name)
    annotated_dir = os.path.join(save_dir, "annotated_inference_frames")
    os.makedirs(annotated_dir, exist_ok=True)
    save_score_path = os.path.join(save_dir, "save_scores.json")
    save_scores = {}
    average_scores = []
    gt_frame_data, _ = process_video_to_frames_and_data(gt_video_file,  output_dir="gt_frames_temp")
    inference_frame_data, _ = process_video_to_frames_and_data(inference_video_file, output_dir="inference_frames_temp")
    original_images = [img for img, _ in inference_frame_data]
    inference_frames_base64 = [img for img, _ in inference_frame_data]
    first_frame = inference_frames_base64[0]

    gt_processed_data = []
    for idx, (base64_image, _) in enumerate(gt_frame_data):
        score = (idx * 100) / (len(gt_frame_data) - 1) if len(gt_frame_data) > 1 else 100.0
        gt_processed_data.append((base64_image, score))

    shuffled_indices_gt, shuffled_frames_base64_gt  = shuffle_one_shot(copy.deepcopy(gt_processed_data))
    shuffled_inference_indices, shuffled_frames_base64 = shuffle_one_shot(copy.deepcopy(inference_frames_base64))

    for freq in range(args.frequency):
        save_scores[freq] = {}
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(args.model)
        prompt = generate_gemini_prompt_for_inference(copy.deepcopy(list(shuffled_frames_base64_gt)), copy.deepcopy(list(shuffled_frames_base64)), first_frame, task)


        response = model.generate_content(prompt)
        raw_scores = response.text.splitlines()
        save_scores[freq]['response'] = response.text
        save_scores[freq]['raw_scores'] = raw_scores
        save_scores[freq]['shuffled_inference_indices'] = shuffled_inference_indices

        remapped_scores = [None] * len(raw_scores)
        # print(f"length of raw_scores: {len(raw_scores)}")
        # print(f"length of shuffled_inference_indices: {len(shuffled_inference_indices)}")
        # print("shuffled_inference_indices: ", shuffled_inference_indices)
        for idx, original_idx in enumerate(shuffled_inference_indices):
            try: 
                remapped_scores[original_idx] = raw_scores[idx]
            except IndexError:
                continue
        os.makedirs(save_dir, exist_ok=True)
        frame_indices = []
        frame_scores = []

        for idx, (img_b64, score) in enumerate(zip(original_images, remapped_scores)):
            text = copy.deepcopy(score)
            try: 
                frame_no_match = re.search(r'Frame (\d+):', score)
                completion_pct_match = re.search(r'Task Completion Percentages: ([\d.]+)%', score)
                if frame_no_match and completion_pct_match:
                    frame_no = int(frame_no_match.group(1))
                    completion_pct = float(completion_pct_match.group(1))
                    frame_indices.append(idx)
                    frame_scores.append(completion_pct)
                    print(f"Frame {idx}: Task Completion Percentage: {completion_pct:.2f}%")
            except Exception as e:
                print(f"Error processing frame {idx}: {e}")
                import pdb; pdb.set_trace()

        if args.debug:
            trial_annotation_path =os.path.join(annotated_dir,f'{freq}')
            os.makedirs(trial_annotation_path, exist_ok=True)
            for idx, (img_b64, score) in enumerate(zip(original_images, remapped_scores)):
                img_data = base64.b64decode(img_b64)
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                annotated_img = img.copy()
                text = f"Score: {copy.deepcopy(score)}"
                annotated_img = draw_wrapped_text(annotated_img, text, pos=(10, 30), max_width=200)

                cv2.imwrite(os.path.join(trial_annotation_path, f"frame_{idx:03d}.jpg"), annotated_img)
        # Plot and save
        plt.figure(figsize=(10, 5))
        plt.plot(frame_indices, frame_scores, marker='o', linestyle='-', color='blue')
        plt.title("Frame ID vs Task Completion Score")
        plt.xlabel("Frame ID")
        plt.ylabel("Score (%)")
        plt.ylim(0, 100)
        plt.grid(True)

        plot_path = os.path.join(save_dir, f"frame_vs_score_{freq}.png")
        # print(f"Plot saved to: {plot_path}")
        plt.savefig(plot_path)
        plt.close()
        save_scores[freq]['frame_indices'] = frame_indices
        save_scores[freq]['frame_scores'] = frame_scores
    
    # Save the scores to a file json
    # Calculate Average of average scores
    with open(save_score_path, "w") as f:
        import json
        json.dump(save_scores, f, indent=4)


def check_shuffling():
    inference_frames_base64 = [f"frame_{i}" for i in range(5)]

    # Step 2: Shuffle using your function
    shuffled_indices, shuffled_frames = shuffle_one_shot(inference_frames_base64)

    # Step 3: Simulate model output
    dummy_model_scores = [f"score_for_{frame}" for frame in shuffled_frames]

    # Step 4: Remap to original order
    remapped_scores = [None] * len(dummy_model_scores)
    for idx, original_idx in enumerate(shuffled_indices):
        remapped_scores[original_idx] = dummy_model_scores[idx]

    # Final sanity check
    print("Original frames:        ", inference_frames_base64)
    print("Shuffled indices:       ", shuffled_indices)
    print("Shuffled frames:        ", shuffled_frames)
    print("Dummy model scores:     ", dummy_model_scores)
    print("Remapped scores:        ", remapped_scores)


    inference_frames_base64 = [f"frame_{i}" for i in range(5)]

    # Step 2: Shuffle
    shuffled_indices, shuffled_frames = shuffle(inference_frames_base64)

    # Step 3: Simulate dummy model output
    dummy_model_scores = [f"score_for_{frame}" for frame in shuffled_frames]

    # Step 4: Remap back to original order
    remapped_scores = [None] * len(dummy_model_scores)
    for idx, original_idx in enumerate(shuffled_indices):
        remapped_scores[original_idx] = dummy_model_scores[idx]

    # Final print
    print("Original frames:     ", inference_frames_base64)
    print("Shuffled indices:    ", shuffled_indices)
    print("Shuffled frames:     ", shuffled_frames)
    print("Dummy model scores:  ", dummy_model_scores)
    print("Remapped scores:     ", remapped_scores)
    
if __name__ == '__main__':
    # Example usage:
    parser = argparse.ArgumentParser(description='Processing Scene Generation')
    parser.add_argument('--gt', type=str, help='Gt Video')
    parser.add_argument('--inference', type=str, help='Inference Video')
    parser.add_argument('--task', type=str, help='Task Description')
    parser.add_argument('--key', type=str, help='API Key')
    parser.add_argument('--zero', type=lambda x: bool(strtobool(x)), help='single_shot or zero_shot')
    parser.add_argument('--one', type=lambda x: bool(strtobool(x)), help='single_shot or zero_shot')
    # Add the missing arguments
    parser.add_argument('--frequency', type=int, default=1, help='Frequency value')
    parser.add_argument('--dir', type=str, help='Directory path')
    parser.add_argument('--scene', type=str, help='Scene name')
    parser.add_argument('--take_top_50', type=lambda x: bool(strtobool(x)), help='Whether to take top 50%')
    parser.add_argument('--test', type=str, help='Test mode')
    parser.add_argument('--policy', type=str, help='Policy name')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), help='Enable debug mode')
    args = parser.parse_args()

    gt_video_file = args.gt  
    inference_video_file = args.inference 
    task = args.task 
    api_key = args.key 
                
    if args.zero:
        zero_shot(args)

    if args.one:
        one_shot(args) 
    # check_shuffling()



    
        