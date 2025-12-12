import glob
import json


def main():
    paths = sorted(glob.glob("results/results_rank*.json"))
    all_records = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            all_records.extend(json.load(f))

    merged_path = "results/results.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"Merged {len(paths)} files into {merged_path} with {len(all_records)} records")


if __name__ == "__main__":
    main()
