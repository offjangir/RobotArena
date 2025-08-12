import json
import argparse
import os

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_averages(data):
    total = {}
    count = 0

    for scene, metrics in data.items():
        for key, value in metrics.items():
            total[key] = total.get(key, 0) + value
        count += 1

    averaged = {key: total[key] / count for key in total}
    return averaged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True, help='Path to metrics JSON file')
    args = parser.parse_args()
    
    json_path = os.path.join(args.base_dir)
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    data = load_json(json_path)
    averaged = compute_averages(data)

    print("\n=== Average Metrics Across Scenes ===")
    for k, v in averaged.items():
        print(f"{k}: {v:.2f}")
    # save metric in the original json file below the content that was loaded
    data['averaged'] = averaged
    # Save the averaged metrics back to the JSON file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\nAveraged metrics saved to {json_path}")
        

if __name__ == "__main__":
    main()
