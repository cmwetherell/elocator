"""Filter caissabase.pgn to keep only classical OTB games with strong players."""

import chess.pgn
import argparse
import sys
import re
from collections import Counter

# Online/rapid/blitz indicators in Event or Site headers
ONLINE_SITE_KEYWORDS = {
    "int",          # chess.com INT, chess24.com INT, etc.
    "chess.com",
    "chess24",
    "lichess",
    "playchess",
    "icc",
    "fics",
    "internet",
    "online",
}

FAST_EVENT_KEYWORDS = {
    "rapid",
    "blitz",
    "bullet",
    "speed",
    "titled tue",
    "titled tuesday",
    "armageddon",
    "lightning",
}

# Compile into a single regex for efficiency
_online_site_re = re.compile(
    "|".join(re.escape(kw) for kw in ONLINE_SITE_KEYWORDS), re.IGNORECASE
)
_fast_event_re = re.compile(
    "|".join(re.escape(kw) for kw in FAST_EVENT_KEYWORDS), re.IGNORECASE
)


def is_classical_otb(game, min_elo=2000):
    """Return True if the game is a classical over-the-board game with strong players.

    Filters:
      1. Both players must have Elo >= min_elo
      2. Game must have a decisive or drawn result (not *)
      3. Site must not contain online platform indicators
      4. Event must not contain rapid/blitz/bullet keywords
      5. Game must have at least 10 moves (skip miniatures / walkovers)
    """
    headers = game.headers

    # Must have Elo headers
    white_elo = headers.get("WhiteElo", "")
    black_elo = headers.get("BlackElo", "")
    if not white_elo or not black_elo or white_elo == "?" or black_elo == "?":
        return False, "missing_elo"

    try:
        if int(white_elo) < min_elo or int(black_elo) < min_elo:
            return False, "low_elo"
    except ValueError:
        return False, "bad_elo"

    # Must have a result
    result = headers.get("Result", "*")
    if result == "*":
        return False, "no_result"

    # Filter online platforms by Site
    site = headers.get("Site", "")
    if _online_site_re.search(site):
        return False, "online_site"

    # Filter non-classical by Event
    event = headers.get("Event", "")
    if _fast_event_re.search(event):
        return False, "fast_event"

    # Skip very short games (walkovers, administrative results)
    moves = list(game.mainline_moves())
    if len(moves) < 20:  # 10 full moves minimum
        return False, "too_short"

    return True, "pass"


def filter_pgn(input_path, output_path, min_elo=2000, max_games=None):
    """Stream-filter a PGN file without loading it all into memory."""
    reject_reasons = Counter()
    accepted = 0
    total = 0

    with open(input_path, errors='replace') as pgn_in, \
         open(output_path, 'w') as pgn_out:

        while True:
            game = chess.pgn.read_game(pgn_in)
            if game is None:
                break

            total += 1

            ok, reason = is_classical_otb(game, min_elo=min_elo)
            if ok:
                pgn_out.write(str(game))
                pgn_out.write("\n\n")
                accepted += 1
            else:
                reject_reasons[reason] += 1

            if total % 10000 == 0:
                print(f"  Scanned {total:,} games, accepted {accepted:,} "
                      f"({accepted/total*100:.1f}%)", flush=True)

            if max_games and accepted >= max_games:
                print(f"Reached max_games limit ({max_games})")
                break

    print(f"\n{'='*50}")
    print(f"FILTER RESULTS")
    print(f"{'='*50}")
    print(f"Total scanned: {total:,}")
    print(f"Accepted: {accepted:,} ({accepted/total*100:.1f}%)")
    print(f"\nRejection reasons:")
    for reason, count in reject_reasons.most_common():
        print(f"  {reason:15s}: {count:,} ({count/total*100:.1f}%)")

    return accepted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter PGN for classical OTB games")
    parser.add_argument("--input", default="./data/caissabase.pgn",
                        help="Input PGN file (default: caissabase.pgn)")
    parser.add_argument("--output", default="./data/filtered.pgn",
                        help="Output PGN file")
    parser.add_argument("--min-elo", type=int, default=2000,
                        help="Minimum Elo for both players (default: 2000)")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Stop after accepting this many games")
    args = parser.parse_args()

    print(f"Filtering {args.input} → {args.output}")
    print(f"Min Elo: {args.min_elo}, Max games: {args.max_games or 'unlimited'}")
    filter_pgn(args.input, args.output, min_elo=args.min_elo, max_games=args.max_games)
