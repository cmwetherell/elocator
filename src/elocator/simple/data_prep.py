import chess
import chess.pgn
import io
import re
import json
import pickle
from elocator.utils import calculate_win_percentage_change, fen_to_tensor, analyze_positions
import torch

# Constants
FILE_PATH = 'data/lichess.pgn'
OUTPUT_TENSOR_PATH = 'data/lichess_gameData2.pt'
MIN_ELO = 2300
VALID_EVENTS = {'Rated Rapid game', 'Rated Blitz game'}
MAX_MOVES = 30000  # Define how many moves to parse before stopping
USE_STOCKFISH = False  # Set to True to evaluate positions using Stockfish

class ChessGameParser:
    """
    Parses chess games from a PGN file and extracts evaluation data.
    """

    def __init__(self, file_path, output_tensor_path, min_elo, valid_events, max_moves, use_stockfish):
        """
        Initialize the parser with configuration.

        :param file_path: Path to the PGN file
        :param output_tensor_path: Path to save the parsed game data as a tensor
        :param min_elo: Minimum Elo rating for players to include their games
        :param valid_events: Set of valid event types
        :param max_moves: Maximum number of moves to parse
        :param use_stockfish: Boolean flag to use Stockfish for evaluations
        """
        self.file_path = file_path
        self.output_tensor_path = output_tensor_path
        self.min_elo = min_elo
        self.valid_events = valid_events
        self.max_moves = max_moves
        self.use_stockfish = use_stockfish
        self.game_data = []

    @staticmethod
    def extract_eval(comment):
        """
        Extract the evaluation score from a PGN comment.

        :param comment: Comment containing evaluation data
        :return: Evaluation score as centipawns or None if not found
        """
        if comment is None:
            return None
        match = re.search(r'\[%eval ([^]]+)\]', comment)
        if not match:
            return None

        eval_str = match.group(1)
        if eval_str.startswith('#'):
            try:
                return 100000  # Representing a mate-in-x situation
            except ValueError:
                return None
        else:
            try:
                return round(float(eval_str) * 100)
            except ValueError:
                return None

    def is_valid_game(self, game):
        """
        Check if the game meets the criteria for parsing.

        :param game: PGN game object
        :return: True if the game is valid, otherwise False
        """
        try:
            white_elo = int(game.headers['WhiteElo'])
            black_elo = int(game.headers['BlackElo'])
            event = game.headers['Event']
        except (KeyError, ValueError):
            return False

        return (
            white_elo > self.min_elo
            and black_elo > self.min_elo
            # and event in self.valid_events
        )

    def parse_game(self, game):
        """
        Parse a single chess game and extract data.

        :param game: PGN game object
        """
        node = game
        previous_eval = None

        while node.variations:
            next_node = node.variation(0)
            board = node.board()
            move_san = board.san(next_node.move)
            current_eval = self.extract_eval(next_node.comment)

            # Use Stockfish for evaluation if no annotation exists
            if current_eval is None and self.use_stockfish:
                current_eval = analyze_positions([board.fen()])[0]

            if current_eval is None:
                break

            turn = 'w' if board.turn else 'b'
            player_elo = (
                game.headers['WhiteElo']
                if board.turn == chess.WHITE
                else game.headers['BlackElo']
            )

            if previous_eval is not None:
                accuracy = calculate_win_percentage_change(
                    previous_eval, current_eval
                ) if turn == 'w' else calculate_win_percentage_change(
                    -previous_eval, -current_eval
                )

                self.game_data.append({
                    "FEN": fen_to_tensor(board.fen()),
                    "Move": move_san,
                    "ScoreBefore": previous_eval,
                    "ScoreAfter": current_eval,
                    "Accuracy": accuracy,
                    "Elo": player_elo
                })

                if len(self.game_data) >= self.max_moves:
                    return

            previous_eval = current_eval
            node = next_node

    def parse(self):
        """
        Parse games from the PGN file.
        """
        with open(self.file_path, 'rb') as f:
            pgn = io.TextIOWrapper(f, encoding='utf-8', errors='ignore')

            if len(self.game_data) % (self.max_moves // 10) == 0:
                print(f"Processed {len(self.game_data)} moves")

            while len(self.game_data) < self.max_moves:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break

                if self.is_valid_game(game):
                    self.parse_game(game)

            self.save_data()

    def save_data(self):
        """
        Save the parsed game data to a tensor file.
        """
        tensor_data = {
            "FENs": torch.stack([item["FEN"] for item in self.game_data]),
            "Metadata": [
                {
                    "Move": item["Move"],
                    "ScoreBefore": item["ScoreBefore"],
                    "ScoreAfter": item["ScoreAfter"],
                    "Accuracy": item["Accuracy"],
                    "Elo": item["Elo"]
                } for item in self.game_data
            ]
        }

        torch.save(tensor_data, self.output_tensor_path)

        print(f"Saved {len(self.game_data)} moves to {self.output_tensor_path}")

if __name__ == "__main__":
    parser = ChessGameParser(
        file_path=FILE_PATH,
        output_tensor_path=OUTPUT_TENSOR_PATH,
        min_elo=MIN_ELO,
        valid_events=VALID_EVENTS,
        max_moves=MAX_MOVES,
        use_stockfish=USE_STOCKFISH
    )
    parser.parse()
