import numpy as np
import json
import argparse
import os
from distutils.util import strtobool
import matplotlib.pyplot as plt
from collections import Counter
SUCCESS_TOP = 30

def main(args):
    metrics = {}
    all_data = {}

    for scene in args.scene_list:
        scene_path = os.path.join(args.test_path, scene, args.zero_or_one)
        all_data[scene] = {}
        metrics[scene] = {}

        for video in os.listdir(scene_path):
            #  check if the vodeo is a directory
            #  skip if it is not a directory
            json_path = os.path.join(scene_path, video, "save_scores.json")
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            video_scores = []
            video_indices = []
            run_names = []

            for key in json_data:

                try:
                    if len(json_data[key]) == 0:
                        continue
                    if len(json_data[key]['frame_scores']) == 0:
                        continue
                    scores = json_data[key]['frame_scores']
                    video_scores.append(scores)
                    run_names.append(key)

                    video_indices.append(json_data[key]['frame_indices'])
                except KeyError as e:
                    print(f"KeyError: {e}")
                    print(scene)
                    print(video)
                    continue

                # if all scores are 0, skip this run
                # if np.sum(scores) == 0:
                #     continue
                # elif np.mean(scores) == 100:
                #     continue
                # else: 
                #     pass

            
            if len(video_scores) == 0:
                print(f"Skipping {video} in {scene} because it has no scores")
                continue

            lengths = [len(scores) for scores in video_scores]
            count = Counter(lengths)
            most_common_length = count.most_common(1)[0][0]
            for score,indices in zip(video_scores, video_indices):
                if len(score) !=most_common_length:
                    video_scores.remove(score)
                    video_indices.remove(indices)
            
            lengths = [len(scores) for scores in video_scores]
            run_stack = np.stack(video_scores)  # shape: (num_runs, num_frames)

            median_run = np.median(run_stack, axis=0)  # shape: (num_frames,)


            all_data[scene][video] = {
                'scores': run_stack,
                'indices': video_indices[0],
            }
            # find the top succes_top % scores of median run
            
            num_top = max(1, int(len(median_run) * SUCCESS_TOP / 100))
            top_scores = np.sort(median_run)[-num_top:]
            avg_top_score = np.mean(top_scores) 

            # average of last 10 frames
            last_10_frames = median_run[-num_top:]
            avg_last_10_frames = np.mean(last_10_frames)
            metrics[scene][video] = {
                'avg': np.mean(np.mean(run_stack, axis=0)), 
                'avg_m': np.mean(median_run),
                'avg_ts': avg_top_score,
                'avg_last_10': avg_last_10_frames
            }
            if args.debug:
                plot(video_indices[0], median_run, os.path.join(scene_path, video, "median_run.png"))
                plot(video_indices[0], np.mean(run_stack, axis=0), os.path.join(scene_path, video, "average_run.png"))


    averaged_metrics = {}
    for scene in metrics:
        avg = []
        avg_m = []
        avg_ts = []
        avg_last_10 = []
        for video in metrics[scene]:
            avg.append(metrics[scene][video]['avg'])
            avg_m.append(metrics[scene][video]['avg_m'])
            avg_ts.append(metrics[scene][video]['avg_ts'])
            avg_last_10.append(metrics[scene][video]['avg_last_10'])
        averaged_metrics[scene] = {
            'avg': np.mean(avg),
            'avg_m': np.mean(avg_m),
            'avg_ts': np.mean(avg_ts),
            'avg_last_10': np.mean(avg_last_10)
        }
            
    
    with open(os.path.join(args.test_path, f'metrics_all_{args.zero_or_one}.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    with open(os.path.join(args.test_path, f'metrics_{args.zero_or_one}.json'), 'w') as f:
        json.dump(averaged_metrics, f, indent=4)

def plot(indices,scores, path):
    plt.figure(figsize=(10, 5))
    plt.plot(indices, scores, marker='o', linestyle='-', color='blue')
    plt.title("Frame ID vs Task Completion Score")
    plt.xlabel("Frame ID")
    plt.ylabel("Score (%)")
    plt.ylim(0, 100)
    plt.grid(True)

    plt.savefig(path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/data/scene/openvla/eval_paper_latest_generate')
    parser.add_argument('--test_name', type=str, default='default_test')
    parser.add_argument('--zero', type=lambda x: bool(strtobool(x)), help='single_shot or zero_shot')
    parser.add_argument('--one', type=lambda x: bool(strtobool(x)), help='single_shot or zero_shot')
    parser.add_argument('--policy', type=str, help='Policy name')
    parser.add_argument('--all_scenes', type=lambda x: bool(strtobool(x)), help='do on all scenes or specific scene')
    parser.add_argument('--scene_name', type=str, help='Scene name')
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), help='debug')

    
    args = parser.parse_args()
    args.test_path = os.path.join(args.base_dir, args.policy, args.test_name)
    if args.zero:
        if args.all_scenes:
            scene_list = os.listdir(args.test_path)
        else:
            scene_list = [args.scene_name]
        args.scene_list = scene_list
        args.zero_or_one = 'zero_shot'
        main(args)
    
    if args.one:
        if args.all_scenes:
            scene_list = os.listdir(args.test_path)
        else:
            scene_list = [args.scene_name]
        args.scene_list = scene_list
        args.zero_or_one = 'one_shot'

        main(args)
    
