import chess
import chess.engine
import chess.pgn
import json
import os

from utils import fen_encoder, calculate_win_percentage_change

# Engine configuration
depth = 20
threads = 32

def load_json_data(file_path, data_type):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return [] if data_type == 'list' else {}


def save_json_data(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

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
    train_data_file = './data/train.json'
    position_scores_file = './data/position_eval.json'
    last_position_file = './data/last_position.txt'

    # Load existing data
    gameData = load_json_data(train_data_file, 'list')
    position_scores = load_json_data(position_scores_file, 'dict')
    last_position = get_last_position(last_position_file)

    with open(pgn_file) as pgn, chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish") as engine:
        engine.configure({"Threads": threads})

        # Move to the last read position in the file
        pgn.seek(last_position)

        game_count = 0
        while True:
            current_position = pgn.tell()  # Save current position
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            process_game(game, engine, gameData, position_scores)
            game_count += 1
            print(game_count)

            if game_count % 10 == 0:
                # Save data and current file position after every 100 games
                save_json_data(train_data_file, gameData)
                save_json_data(position_scores_file, position_scores)
                save_last_position(last_position_file, current_position)
                print(f"Processed {game_count} games")

        # Final save
        save_json_data(train_data_file, gameData)
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
            score_before_white_pov = info_before["score"].white().score(mate_score=100000)  # Always in white's perspective
            # Store the score in the dictionary
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
            score_after_white_pov = info_after["score"].white().score(mate_score=100000)  # Always in white's perspective
            # Store the score in the dictionary
            position_scores[fen_after_move] = score_after_white_pov

        # Convert score to the current player's perspective
        score_after = score_after_white_pov if player_perspective == chess.WHITE else -score_after_white_pov

        # Calculate the accuracy of the move
        move_accuracy = calculate_win_percentage_change(score_before, score_after)

        gameData.append({
            "FEN": fen_encoder(fen_before_move),
            "Move": move_san,
            "ScoreBefore": score_before,
            "ScoreAfter": score_after,
            "Accuracy": move_accuracy,
            "Elo": game.headers["WhiteElo"] if player_perspective == chess.WHITE else game.headers["BlackElo"]
        })

if __name__ == "__main__":
    main()
