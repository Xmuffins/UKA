import json
import sys


def calc_avg_emo_point(jsonl_file):
    total_success = 0.0
    total_emo = 0.0
    total_len = 0.0
    count = 0

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception as exc:
                print(f"Warning: failed to parse line: {line}\nError: {exc}")
                continue

            if "emo_point" not in data:
                continue

            emo_point = data["emo_point"]
            total_emo += emo_point
            total_success += 1 if emo_point > 0.5 else 0
            total_len += len(data.get("history", []))
            count += 1

    if count == 0:
        print("No valid emo_point found.")
        return

    print(f"Average emo_point: {total_emo / count:.4f} (count={count})")
    print(f"Success rate: {total_success / count:.4f} (count={count})")
    print(f"Average turn: {total_len / (2 * count):.4f} (count={count})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_score.py <result_file.jsonl>")
    else:
        calc_avg_emo_point(sys.argv[1])
