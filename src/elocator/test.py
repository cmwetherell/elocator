import chess.engine

def analyze_position(fen: str) -> float:
    """Analyze a position and return the evaluation score.

    Args:
        fen (str): FEN string of the position.

    Returns:
        float: Evaluation score of the position.
    """
    stockfish_path = "/opt/homebrew/bin/stockfish"

    # Create a chess board from the FEN string
    board = chess.Board(fen)

    # Initialize the engine
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    info = engine.analyse(board, chess.engine.Limit(time=0.1))

    # Get the evaluation from the engine
    evaluation = info["score"].white().score()

    # Close the engine
    engine.quit()

    return evaluation

if __name__ == "__main__":
    default_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    eval = analyze_position(default_fen)
    print(eval)