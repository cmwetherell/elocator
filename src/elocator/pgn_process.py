import chess
import chess.engine
import chess.pgn
import json
import os

from utils import calculate_win_percentage_change

# Engine configuration
depth = 18
threads = 8


def load_jsonl_data(file_path):
    """Load data from a JSONL file (one JSON object per line)."""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def append_jsonl_data(file_path, records):
    """Append records to a JSONL file."""
    with open(file_path, 'a') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')


def load_json_data(file_path, data_type):
    """Load position_eval cache (still regular JSON)."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return [] if data_type == 'list' else {}


def save_json_data(file_path, data):
    """Save position_eval cache (still regular JSON)."""
    with open(file_path, 'w') as file:
        json.dump(data, file)


def get_last_position(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return int(file.read().strip())
    return 0


def save_last_position(file_path, position):
    with open(file_path, 'w') as file:
        file.write(str(position))


def main():
    pgn_file = "./data/filtered.pgn"
    train_data_file = './data/train.jsonl'
    position_scores_file = './data/position_eval.json'
    last_position_file = './data/last_position.txt'

    # Load existing data
    position_scores = load_json_data(position_scores_file, 'dict')
    last_position = get_last_position(last_position_file)

    with open(pgn_file) as pgn, chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish") as engine:
        engine.configure({"Threads": threads})

        # Move to the last read position in the file
        pgn.seek(last_position)

        game_count = 0
        batch = []  # accumulate records before writing
        while True:
            current_position = pgn.tell()  # Save current position
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            process_game(game, engine, batch, position_scores)
            game_count += 1
            print(game_count)

            if game_count % 10 == 0:
                # Save data and current file position after every 10 games
                append_jsonl_data(train_data_file, batch)
                batch = []
                save_json_data(position_scores_file, position_scores)
                save_last_position(last_position_file, current_position)
                print(f"Processed {game_count} games")

        # Final save
        if batch:
            append_jsonl_data(train_data_file, batch)
        save_json_data(position_scores_file, position_scores)
        save_last_position(last_position_file, current_position)


def process_game(game, engine, gameData, position_scores):
    board = game.board()
    for move in game.mainline_moves():
        fen_before_move = board.fen()
        move_san = board.san(move)
        player_perspective = board.turn

        # Check if the position before the move is already evaluated
        if fen_before_move in position_scores:
            score_before_white_pov = position_scores[fen_before_move]
        else:
            # Evaluate position before the move
            info_before = engine.analyse(board, chess.engine.Limit(depth=depth))
            score_before_white_pov = info_before["score"].white().score(mate_score=100000)
            position_scores[fen_before_move] = score_before_white_pov

        # Convert score to the current player's perspective
        score_before = score_before_white_pov if player_perspective == chess.WHITE else -score_before_white_pov

        board.push(move)

        fen_after_move = board.fen()
        # Check if the position after the move is already evaluated
        if fen_after_move in position_scores:
            score_after_white_pov = position_scores[fen_after_move]
        else:
            # Evaluate position after the move
            info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
            score_after_white_pov = info_after["score"].white().score(mate_score=100000)
            position_scores[fen_after_move] = score_after_white_pov

        # Convert score to the current player's perspective
        score_after = score_after_white_pov if player_perspective == chess.WHITE else -score_after_white_pov

        # Calculate the accuracy of the move
        move_accuracy = calculate_win_percentage_change(score_before, score_after)

        gameData.append({
            "fen": fen_before_move,
            "move": move_san,
            "score_before": score_before,
            "score_after": score_after,
            "accuracy": move_accuracy,
            "elo": game.headers["WhiteElo"] if player_perspective == chess.WHITE else game.headers["BlackElo"]
        })


if __name__ == "__main__":
    main()
