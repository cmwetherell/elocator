import chess
import chess.pgn
import io
import re
from utils import calculate_win_percentage_change, fen_encoder
import json
import pickle

def extract_eval(comment):
    if comment is None:
        return None
    match = re.search(r'\[%eval ([^]]+)\]', comment)
    if not match:
        return None
    eval_str = match.group(1)
    if eval_str.startswith('#'):
        # Mate found, extract number after '#'
        try:
            mate_in_moves = int(eval_str[1:])
            return 100000
        except ValueError:
            return None  # Just in case there's an error converting to integer
    else:
        try:
            # Convert evaluation to a floating-point number representing centipawns
            return round(float(eval_str) * 100)
        except ValueError:
            return None

def main():
    gameData = []
    file_path = 'data/lichess.pgn'

    with open(file_path, 'rb') as f:
        pgn = io.TextIOWrapper(f, encoding='utf-8', errors='ignore')
        i = 0
        while True:
            game = chess.pgn.read_game(pgn)
            i += 1
            if game is None:
                break

            if (
                (int(game.headers['WhiteElo']) > 2300)
                & (int(game.headers['BlackElo']) > 2300)
                & (
                    (game.headers['Event'] == 'Rated Rapid game')
                    | (game.headers['Event'] == 'Rated Blitz game')
                )
            ):
                node = game
                previous_eval = None
                while node.variations:
                    next_node = node.variation(0)
                    board = node.board()
                    move_san = board.san(next_node.move)
                    current_eval = extract_eval(next_node.comment)

                    if 'w' in board.fen():
                        turn = 'w'
                    elif 'b' in board.fen():
                        turn = 'b'
                    else:
                        turn = None

                    if current_eval is None:
                        # skip to the next game
                        break

                    # Determine player perspective and corresponding Elo
                    player_perspective = chess.WHITE if board.turn == chess.BLACK else chess.BLACK
                    player_elo = game.headers["WhiteElo"] if player_perspective == chess.WHITE else game.headers["BlackElo"]

                    # Append data to the gameData list
                    if previous_eval is not None:
                        if turn == 'w':
                            accuracy = calculate_win_percentage_change(previous_eval, current_eval)
                        elif turn == 'b':
                            accuracy = calculate_win_percentage_change(-previous_eval, -current_eval)
                        

                        gameData.append({
                            "FEN": fen_encoder(board.fen()),
                            "Move": move_san,
                            "ScoreBefore": previous_eval,
                            "ScoreAfter": current_eval,
                            "Accuracy": accuracy,  # TODO: CHeck that this accuracy calc matches the way the previous training data was impleented
                            "Elo": player_elo
                        })

                    # print('\n\n\n\n')
                    
                    # Update previous_eval for the next move
                    previous_eval = current_eval
                    node = next_node  # Move to the next node

                # print(gameData)  # Optional: to see the output for each game
                # print json from gameData
                # print(json.dumps(gameData, indent=4))
            if i % 50000 == 0:
                print('writing to file')
                # save gameData to a pickle, and then to a json as well
                with open('data/lichess_gameData.pkl', 'wb') as f:
                    pickle.dump(gameData, f)
                # with open('data/lichess_gameData.json', 'w') as f:
                #     json.dump(gameData, f, indent=4)
                print("Done: ", i) # Optional: to see the progress


if __name__ == "__main__":
    main()
