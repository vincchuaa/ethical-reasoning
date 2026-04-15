import jsonlines
from pathlib import Path
from typing import List, Dict


def load_jsonl(filepath: str) -> List[Dict]:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    data = []
    with jsonlines.open(filepath, mode='r') as reader:
        for obj in reader:
            data.append(obj)
    if not data:
        raise ValueError(f"File is empty: {filepath}")
    return data
