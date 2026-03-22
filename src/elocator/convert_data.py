"""Convert existing training data to the new JSONL format with raw FEN strings."""

import json
import pickle
import sys
from utils import fen_decode


def convert_train_json(input_path, output_path):
    """Convert train.json (780-element encoded arrays) to JSONL with raw FENs."""
    print(f"Loading {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Converting {len(data)} records...")
    converted = 0
    with open(output_path, 'w') as out:
        for i, record in enumerate(data):
            fen = fen_decode(record['FEN'])
            accuracy = record['Accuracy']
            out.write(json.dumps({"fen": fen, "accuracy": accuracy}) + '\n')
            converted += 1
            if converted % 100000 == 0:
                print(f"  {converted}/{len(data)}")

    print(f"Wrote {converted} records to {output_path}")


def convert_pickle(input_path, output_path):
    """Convert lichess_gameData.pkl (780-element encoded arrays) to JSONL."""
    print(f"Loading {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Converting {len(data)} records...")
    converted = 0
    with open(output_path, 'w') as out:
        for record in data:
            fen = fen_decode(record['FEN'])
            accuracy = record['Accuracy']
            out.write(json.dumps({"fen": fen, "accuracy": accuracy}) + '\n')
            converted += 1
            if converted % 100000 == 0:
                print(f"  {converted}/{len(data)}")

    print(f"Wrote {converted} records to {output_path}")


def validate_sample(jsonl_path, n=10):
    """Print a few records from the converted JSONL for manual inspection."""
    print(f"\nSample records from {jsonl_path}:")
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            record = json.loads(line)
            print(f"  [{i}] fen={record['fen'][:60]}...  accuracy={record['accuracy']:.2f}")


if __name__ == "__main__":
    import os

    data_dir = "./data"

    # Convert train.json if it exists
    train_json = os.path.join(data_dir, "train.json")
    train_jsonl = os.path.join(data_dir, "train.jsonl")
    if os.path.exists(train_json):
        convert_train_json(train_json, train_jsonl)
        validate_sample(train_jsonl)
    else:
        print(f"Skipping {train_json} (not found)")

    # Convert lichess pickle if it exists
    lichess_pkl = os.path.join(data_dir, "lichess_gameData.pkl")
    lichess_jsonl = os.path.join(data_dir, "lichess.jsonl")
    if os.path.exists(lichess_pkl):
        convert_pickle(lichess_pkl, lichess_jsonl)
        validate_sample(lichess_jsonl)
    else:
        print(f"Skipping {lichess_pkl} (not found)")
