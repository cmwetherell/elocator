'utility module for Elocator'

import numpy as np
import math
import chess

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

    try:
        ep_array[convert_ep_square_to_int(ep_square)] = 1
    except:
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
    # Dictionary mapping ep squares to integers
    ep_square_map = {
        "a6": 0,
        "b6": 1,
        "c6": 2,
        "d6": 3,
        "e6": 4,
        "f6": 5,
        "g6": 6,
        "h6": 7
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

if __name__ == "__main__":
    # Example usage:
    encoded = fen_encoder(default_fen)
    # print("Encoded FEN:", encoded)
    decoded_fen = fen_decode(encoded)
    print("Original FEN:", default_fen)
    print("Decoded FEN:", decoded_fen)

    print(flip_fen(default_fen))