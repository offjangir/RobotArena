import json
import argparse
import os
import numpy as np

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_stats(data):
    metrics_dict = {}

    for scene, metrics in data.items():
        for key, value in metrics.items():
            metrics_dict.setdefault(key, []).append(value)

    averaged = {k: float(np.mean(v)) for k, v in metrics_dict.items()}
    stds = {k: float(np.std(v, ddof=1)) for k, v in metrics_dict.items()}
    sems = {k: float(np.std(v, ddof=1) / np.sqrt(len(v))) for k, v in metrics_dict.items()}

    return averaged, stds, sems

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True, help='Path to metrics JSON file')
    parser.add_argument('--out_file', default="results.json", help='Path to save results JSON file')
    args = parser.parse_args()

    json_path = os.path.join(args.base_dir)
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    data = load_json(json_path)
    averaged, stds, sems = compute_stats(data)

    results = {
        "averaged": averaged,
        "stds": stds,
        "sems": sems
    }
    out_file = os.path.join(os.path.dirname(json_path), "results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {out_file}")
if __name__ == "__main__":
    main()
