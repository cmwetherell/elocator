'utility module for Elocator'

import io
import numpy as np
import math
import chess
import chess.pgn
import chess.engine
from typing import Union, List, Tuple, Dict
import torch

default_fen = "rnbqkbnr/p1p1pppp/3p4/Pp6/8/8/1PPPPPPP/RNBQKBNR b KQkq b6 0 3"


def flip_fen(fen: str) -> str:
    # Create a chess board from the FEN string
    board = chess.Board(fen)

    # Flip the board if it's black's turn to make it white's turn
    if board.turn == chess.BLACK:
        board = board.mirror()

    # Get the new FEN string with white to move
    new_fen = board.fen()
    return new_fen


def fen_encoder(fen: str) -> np.array:
    """Encode a FEN string into a 8x8x12 numpy array.

    Args:
        fen (str): FEN string of the position.

    Returns:
        np.array: 8x8x12 numpy array.
    """
    # if black to move, mirror fen with chess library
    board = chess.Board(fen)
    if board.turn == chess.BLACK:
        board = board.mirror()
        fen = board.fen()

    # Initialize the board array
    board = np.zeros((8, 8, 12), dtype=np.uint8)

    # Split FEN string into board position and other info
    fen_split = fen.split(" ")
    for idx, val in enumerate(fen_split):
        if idx == 0:
            position = val
        elif idx == 1:
            side_to_move = val
        elif idx == 2:
            castling_rights = val
        elif idx == 3:
            ep_square = val
        elif idx == 4:
            halfmove = val
        elif idx == 5:
            move_counter = val

    # Split position into ranks
    ranks = position.split("/")

    # Loop through ranks
    for i, rank in enumerate(ranks):
        file = 0
        # Loop through each character in the FEN rank string
        for c in rank:
            # If the character is a number, skip that many files
            if c.isnumeric():
                file += int(c)
            # If the character is a piece, place it on the board
            else:
                # Convert FEN piece character to integer
                piece = piece_to_int(c)
                # Place the piece on the board
                board[i, file, piece] = 1
                # Move to the next file
                file += 1
    encoded = board.flatten()

    # extend encoded to add 8 ep squares and 4 castling rights
    # only need to keep traxck fo 8 squares since own vs opponent
    ep_array = np.zeros(8, dtype=np.uint8)

    if ep_square != "-":
        try:
            ep_array[convert_ep_square_to_int(ep_square)] = 1
        except KeyError:
            pass

    castling_array = np.zeros(4, dtype=np.uint8)
    if "K" in castling_rights:
        castling_array[0] = 1
    if "Q" in castling_rights:
        castling_array[1] = 1
    if "k" in castling_rights:
        castling_array[2] = 1
    if "q" in castling_rights:
        castling_array[3] = 1

    # print each of the three components
    # print(encoded)
    # print(ep_array)
    # print(castling_array)
        

    encoded = np.append(encoded, ep_array)
    encoded = np.append(encoded, castling_array)

    return encoded.tolist()

def piece_to_int(piece: str) -> int:
    """Convert a FEN piece character to an integer.

    Args:
        piece (str): FEN piece character.

    Returns:
        int: Integer representation of the piece.
    """
    # Dictionary mapping piece characters to integers
    piece_map = {
        "p": 0,
        "n": 1,
        "b": 2,
        "r": 3,
        "q": 4,
        "k": 5,
        "P": 6,
        "N": 7,
        "B": 8,
        "R": 9,
        "Q": 10,
        "K": 11
    }

    return piece_map[piece]

def convert_ep_square_to_int(ep_square: str) -> int:
    """Convert a FEN ep square to an integer.

    Args:
        ep_square (str): FEN ep square.

    Returns:
        int: Integer representation of the ep square.
    """
    # Dictionary mapping ep squares to integers (file index 0-7)
    # Rank 6 = white can capture EP, Rank 3 = black can capture EP (mirrored to rank 6)
    ep_square_map = {
        "a6": 0, "b6": 1, "c6": 2, "d6": 3,
        "e6": 4, "f6": 5, "g6": 6, "h6": 7,
        "a3": 0, "b3": 1, "c3": 2, "d3": 3,
        "e3": 4, "f3": 5, "g3": 6, "h3": 7,
    }

    return ep_square_map[ep_square]

# print(fen_encoder("rnbqkbnr/p1p1pppp/3p4/Pp6/8/8/1PPPPPPP/RNBQKBNR w KQkq b6 0 3"))

def get_win_percent(centipawns): # from Lichess
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * centipawns)) - 1)

def calculate_win_percentage_change(centipawns_before, centipawns_after):
    # Calculate Win% using the given formula for before and after values
    win_percent_before = get_win_percent(centipawns_before)
    win_percent_after = get_win_percent(centipawns_after)
    
    # Calculate the change in Win%
    #worse moves have a higher score
    win_percent_loss = -1 * min(0, win_percent_after - win_percent_before)
    
    return win_percent_loss


def fen_decode(encoded):
    '''
    Decode a flattened 8x8x12 + 8 + 4 numpy array into a FEN string.
    
    Args:
        encoded (np.array): Flattened 8x8x12 + 8 + 4 numpy array.
    
    Returns:
        str: FEN string of the position.
    '''

    # split the encoed into the three components
    board = np.array(encoded[:768]).reshape(8, 8, 12)
    ep_array = encoded[768:776]
    castling_array = encoded[776:]

    # convert board back into componernt strings by rank and file and piece
    fen_board = ""
    for rank in board:
        empty = 0
        for square in rank:
            piece = int_to_piece(np.argmax(square) + 1 * any(square))
            if piece == "":
                empty += 1
            else:
                if empty > 0:
                    fen_board += str(empty)
                    empty = 0
                fen_board += piece
        if empty > 0:
            fen_board += str(empty)
        fen_board += "/"

    # remove the last slash
    fen_board = fen_board[:-1]
    
    # ternery operator instead
    ep_square = int_to_ep_square(np.argmax(ep_array)) if any(ep_array) else "-"
        

    # convert castling_array to castling_rights
    castling_rights = ""
    if castling_array[0] == 1:
        castling_rights += "K"
    if castling_array[1] == 1:
        castling_rights += "Q"
    if castling_array[2] == 1:
        castling_rights += "k"
    if castling_array[3] == 1:
        castling_rights += "q"
    
    castling_rights = castling_rights if castling_rights else "-"

    return f"{fen_board} w {castling_rights} {ep_square} 0 1"

def int_to_piece(idx):
    piece_map = {
        1: 'p', 2: 'n', 3: 'b', 4: 'r', 5: 'q', 6: 'k',
        7: 'P', 8: 'N', 9: 'B', 10: 'R', 11: 'Q', 12: 'K'
    }
    return piece_map.get(idx, '')

def int_to_ep_square(idx):
    ep_square_map = {
        0: 'a6', 1: 'b6', 2: 'c6', 3: 'd6',
        4: 'e6', 5: 'f6', 6: 'g6', 7: 'h6'
    }
    return ep_square_map.get(idx, '-')

def parse_pgn(pgn: str) -> Tuple[Dict, List]: # type: ignore
    """Parse a PGN string into a list of moves, extract headers into dict.

    Args:
        pgn (str): PGN string of the game.

    Returns:
        list:tuple of headers and a List of FENs.
    """
    pgn = io.StringIO(pgn)
    game = chess.pgn.read_game(pgn)
    headers = dict(game.headers)
    FENs = []
    board = game.board()
    FENs.append(board.fen()) # push default fen
    for move in game.mainline_moves():
        board.push(move)
        FENs.append(board.fen())
    return headers, FENs

def analyze_positions(fens: Union[str, List[str]]) -> List[float]:
    # TODO: Combine this with the FEN generator so we dont make a billion chess boards.
    """Analyze a position or positions and return the evaluation scores.

    Args:
        fens (Union[str, List[str]]): A single FEN string or a list of FEN strings of the positions.

    Returns:
        List[float]: Evaluation scores of the positions.
    """
    import shutil
    stockfish_path = shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path, timeout=30)

    # Ensure fens is a list even if a single FEN string is provided
    if isinstance(fens, str):
        fens = [fens]

    evaluations = []

    for fen in fens:
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(depth=20))  # Adjust time limit as needed
        evaluation = info["score"].white().score(mate_score=10000)  # Use a large number for mate score
        evaluations.append(evaluation if evaluation is not None else 0)

    engine.quit()
    return evaluations

# ---------------------------------
# 1) FEN to Input Tensor
# ---------------------------------

PIECE_MAP = {
    'P': 0,  # White Pawn
    'N': 1,  # White Knight
    'B': 2,  # White Bishop
    'R': 3,  # White Rook
    'Q': 4,  # White Queen
    'K': 5,  # White King
    'p': 6,  # Black Pawn
    'n': 7,  # Black Knight
    'b': 8,  # Black Bishop
    'r': 9,  # Black Rook
    'q': 10, # Black Queen
    'k': 11, # Black King
}

def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Converts a FEN string into an 18x8x8 tensor:
      [0..11]  -> one-hot planes for each piece type
      [12]     -> side-to-move (all ones if white, zeros if black)
      [13..16] -> castling rights [white_K, white_Q, black_K, black_Q]
      [17]     -> en passant file (encoded across the 8 columns of the rank, else zeros)
    
    If you want a different channel layout (e.g., separate plane for each castling right),
    feel free to adjust. The output is a float tensor with shape (18, 8, 8).
    """
    parts = fen.split()
    board_part = parts[0]
    side_to_move = parts[1]
    castling_part = parts[2]
    en_passant_part = parts[3]

    # Initialize zero tensor [channels, 8, 8]
    # We'll do 18 channels: 12 for piece type, 1 for side-to-move, 4 for castling, 1 for en passant
    tensor = torch.zeros((18, 8, 8), dtype=torch.float32)

    # 1) Fill in piece planes
    rows = board_part.split('/')
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for char in row:
            if char.isdigit():
                # This means we have 'n' empty squares
                col_idx += int(char)
            else:
                # It's a piece
                channel = PIECE_MAP[char]
                tensor[channel, row_idx, col_idx] = 1.0
                col_idx += 1

    # 2) Side to move
    #    If side to move = 'w', set channel 12 to 1s. If 'b', leave it at 0.
    if side_to_move == 'w':
        tensor[12].fill_(1.0)

    # 3) Castling rights
    #    White K-side = channel 13, White Q-side = 14, Black K-side = 15, Black Q-side = 16
    if 'K' in castling_part:  # White can castle short
        tensor[13].fill_(1.0)
    if 'Q' in castling_part:  # White can castle long
        tensor[14].fill_(1.0)
    if 'k' in castling_part:  # Black can castle short
        tensor[15].fill_(1.0)
    if 'q' in castling_part:  # Black can castle long
        tensor[16].fill_(1.0)

    # 4) En passant
    #    If en_passant_part != '-', it indicates a file (a-h) and a rank. E.g., 'e3'.
    #    We'll set channel 17 in the relevant square. Usually it's the 3rd or 6th rank.
    if en_passant_part != '-':
        file_letter = en_passant_part[0]  # e.g., 'e'
        file_idx = ord(file_letter) - ord('a')  # 0..7
        # We find rank (the second char in e3). Convert from chess rank to row index in [0..7].
        rank_char = en_passant_part[1]  # e.g., '3'
        rank_idx = 8 - int(rank_char)   # rank 1 -> row 7, rank 8 -> row 0, etc.
        tensor[17, rank_idx, file_idx] = 1.0
    
    return tensor

if __name__ == "__main__":
    # Example usage:
    encoded = fen_encoder(default_fen)
    # print("Encoded FEN:", encoded)
    decoded_fen = fen_decode(encoded)
    print("Original FEN:", default_fen)
    print("Decoded FEN:", decoded_fen)

    print(flip_fen(default_fen))

    sample_pgn = '''[Event "F/S Return Match"]
[Site "Belgrade, Serbia JUG"]
[Date "1992.11.04"]
[Round "29"]
[White "Fischer, Robert J."]
[Black "Spassky, Boris V."]
[Result "1/2-1/2"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 {This opening is called the Ruy Lopez.} 3... a6
4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7
11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6 16. Bh4 c5 17. dxe5
Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 20. Nbd2 Nxd6 21. Nc4 Nxc4 22. Bxc4 Nb6
23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7 Rxe1+ 26. Qxe1 Kxf7 27. Qe3 Qg5 28. Qxg5
hxg5 29. b3 Ke6 30. a3 Kd6 31. axb4 cxb4 32. Ra5 Nd5 33. f3 Bc8 34. Kf2 Bf5
35. Ra7 g6 36. Ra6+ Kc5 37. Ke1 Nf4 38. g3 Nxh3 39. Kd2 Kb5 40. Rd6 Kc5 41. Ra6
Nf2 42. g4 Bd3 43. Re6 1/2-1/2'''
    headers, FENs = parse_pgn(sample_pgn)
    print("Headers:", headers)
    print("FENs:", FENs)
