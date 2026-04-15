"""
Prepare ERR Training Data from multi-turn TRIAL pipeline outputs.

Input files are JSONL produced by attack/main.py (engage or explain mode),
each line containing a {"messages": [...], "original_prompt": "..."} entry.

The trainer (defense/err/data.py) handles the train/val split internally (90/10).

Usage:
    python defense/scripts/prepare_err_data.py \
        --benign_responses  data/conversations/benign_d-ds_vs_hf-llama_engage.jsonl \
        --harmful_responses data/conversations/jbb_d-ds_vs_hf-llama_explain.jsonl \
        --output_dir        defense/datasets/train_data
"""
import argparse
import json
import os


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_messages(item):
    """Return {"messages": [...]} from a TRIAL engage/explain output entry."""
    if "messages" in item:
        msgs = item["messages"]
        if msgs and isinstance(msgs[0], dict) and "role" in msgs[0]:
            return {"messages": msgs}
    return None


def load_and_extract(path):
    raw = load_jsonl(path)
    extracted = [extract_messages(item) for item in raw]
    skipped = sum(1 for e in extracted if e is None)
    extracted = [e for e in extracted if e is not None]
    if skipped:
        print(f"  WARNING: skipped {skipped} entries with no 'messages' field")
    return extracted


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ERR training data from multi-turn TRIAL pipeline outputs"
    )
    parser.add_argument("--benign_responses", type=str, required=True,
                        help="JSONL from engage mode (benign multi-turn conversations)")
    parser.add_argument("--harmful_responses", type=str, required=True,
                        help="JSONL from explain mode (harmful multi-turn conversations)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for train data")
    args = parser.parse_args()

    print(f"Loading benign responses from {args.benign_responses}...")
    benign = load_and_extract(args.benign_responses)
    print(f"  -> {len(benign)} examples")

    print(f"Loading harmful responses from {args.harmful_responses}...")
    harmful = load_and_extract(args.harmful_responses)
    print(f"  -> {len(harmful)} examples")

    save_jsonl(benign,  os.path.join(args.output_dir, "benign_train.jsonl"))
    save_jsonl(harmful, os.path.join(args.output_dir, "harmful_train.jsonl"))

    print(f"\nSaved:")
    print(f"  Benign:  {len(benign)} examples")
    print(f"  Harmful: {len(harmful)} examples")


if __name__ == "__main__":
    main()
