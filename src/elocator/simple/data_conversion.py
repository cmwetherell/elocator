import pickle
import torch

# If your fen_decode and fen_to_tensor are in elocator.utils, import them accordingly:
from elocator.utils import fen_decode, fen_to_tensor

# Paths
OLD_PICKLE_PATH = "data/lichess_gameData.pkl"
OUTPUT_TENSOR_PATH = "data/lichess_gameData_converted.pt"

def main():
    # 1. Load old data from pickle
    with open(OLD_PICKLE_PATH, 'rb') as f:
        old_data = pickle.load(f)  # This should give you a list of dicts, similar to your JSON structure

    # 2. Decode FEN and convert to tensor
    converted_records = []
    for idx, record in enumerate(old_data):
        if idx % 100000 == 0:
            print(f"Processed {idx} records")

        # record["FEN"] is stored in an encoded format; decode it
        decoded_fen = fen_decode(record["FEN"])
        
        # Convert decoded FEN to a tensor
        fen_tensor = fen_to_tensor(decoded_fen)
        
        # Create a new record with tensor FEN
        converted_records.append({
            "FEN": fen_tensor,
            "Move": record["Move"],
            "ScoreBefore": record["ScoreBefore"],
            "ScoreAfter": record["ScoreAfter"],
            "Accuracy": record["Accuracy"],
            "Elo": record["Elo"]
        })

    # 3. Build final data structure
    tensor_data = {
        "FENs": torch.stack([entry["FEN"] for entry in converted_records]),
        "Metadata": [
            {
                "Move": entry["Move"],
                "ScoreBefore": entry["ScoreBefore"],
                "ScoreAfter": entry["ScoreAfter"],
                "Accuracy": entry["Accuracy"],
                "Elo": entry["Elo"]
            }
            for entry in converted_records
        ]
    }

    # 4. Save the final data
    torch.save(tensor_data, OUTPUT_TENSOR_PATH)
    print(f"Converted {len(converted_records)} positions and saved to {OUTPUT_TENSOR_PATH}")

if __name__ == "__main__":
    main()
