"""Deduplicate training data by FEN position.

When the same FEN appears multiple times (common in opening theory), we keep
the record with the median accuracy — this gives a stable label for positions
that show up across many games.
"""

import json
import argparse
import sys
from collections import defaultdict
import statistics


def dedup_jsonl(input_path, output_path):
    """Deduplicate JSONL by FEN, keeping median accuracy per position."""
    # Group accuracies by FEN
    fen_accuracies = defaultdict(list)
    total = 0

    print(f"Reading {input_path}...")
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            fen_accuracies[record['fen']].append(record['accuracy'])
            total += 1

    unique = len(fen_accuracies)
    dupes = total - unique
    print(f"Total records: {total:,}")
    print(f"Unique FENs: {unique:,}")
    print(f"Duplicates: {dupes:,} ({dupes/total*100:.1f}%)")

    # Analyze duplication
    dup_counts = [len(v) for v in fen_accuracies.values() if len(v) > 1]
    if dup_counts:
        print(f"\nDuplicated positions: {len(dup_counts):,}")
        print(f"Max occurrences: {max(dup_counts)}")
        print(f"Avg occurrences for dups: {sum(dup_counts)/len(dup_counts):.1f}")

        # Show top duplicated positions
        top_dups = sorted(fen_accuracies.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        print(f"\nTop 5 most duplicated FENs:")
        for fen, accs in top_dups:
            print(f"  {fen[:60]}...  count={len(accs)}  "
                  f"mean_acc={sum(accs)/len(accs):.2f}  "
                  f"std_acc={statistics.stdev(accs) if len(accs) > 1 else 0:.2f}")

    # Write deduplicated data (median accuracy)
    print(f"\nWriting {unique:,} records to {output_path}...")
    with open(output_path, 'w') as f:
        for fen, accs in fen_accuracies.items():
            median_acc = statistics.median(accs)
            f.write(json.dumps({"fen": fen, "accuracy": round(median_acc, 4)}) + '\n')

    print(f"Done. Removed {dupes:,} duplicates ({dupes/total*100:.1f}% reduction)")
    return unique


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate JSONL training data by FEN")
    parser.add_argument("--input", default="./data/train.jsonl",
                        help="Input JSONL file")
    parser.add_argument("--output", default="./data/train_dedup.jsonl",
                        help="Output deduplicated JSONL file")
    args = parser.parse_args()
    dedup_jsonl(args.input, args.output)
