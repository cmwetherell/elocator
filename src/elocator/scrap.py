import chess

move = chess.Move.from_uci("e2e5")
board = chess.Board()

board.san(move)
board.push(move)

# print board graphically

print(board)
