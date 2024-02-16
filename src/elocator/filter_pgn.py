import chess
import chess.engine
import io
import chess.pgn

# engine configuration
depth = 20
time = 0.25
threads = 32

def filter_pgn(path: str) -> None:
    """Filter a PGN file to only include games with players rated 2500+.

    Args:
        path (str): Path to the PGN file.
    """
    # Open pgn file in binary mode and read
    with open(path, 'rb') as file:
        raw_data = file.read()
    
    # Decode the content with UTF-8 encoding, replace errors
    decoded_data = raw_data.decode('utf-8', errors='replace')

    # Use StringIO to simulate a file object from the decoded string
    pgn = io.StringIO(decoded_data)

    # Open new file to write filtered games to
    with open("./data/filtered.pgn", "w") as filtered_pgn:

        i = 0
        while True:
            i += 1
            if i % 1000 == 0:
                print("Done: ", i)

            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            # Skip game if it doesn't have necessary headers
            if not all(key in game.headers for key in ["WhiteElo", "BlackElo", "Result"]):
                continue

            # Skip if WhiteElo and BlackElo not >2500
            if int(game.headers["WhiteElo"]) < 2500 or int(game.headers["BlackElo"]) < 2500:
                continue

            # Skip Internet games
            if "Site" in game.headers and "INT" in game.headers["Site"]:
                continue
            
            # add newline to end of game
            filtered_pgn.write(game.__str__())
            filtered_pgn.write("\n\n")


def main():

    pgn_file = "./data/scidFilter.pgn"

    pgn = open(pgn_file)

    filter_pgn(pgn_file)
    

if __name__ == "__main__":
    main()
