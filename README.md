# Elocator

A project to help identify the Elo of a chess player based on a set of moves or games. Can be used to identify cheating, game throwing, or other anomalies in a players game.

### This project is new and under active development.

### TODO:
- Parse Caissabase PGN's to make dataset and Stockfish derived "Accuracy". Data should be board state, player Elo, move, move accuracy. Move likely not needed for the model.
    - Maybe filter the data to 2600+ and don't use Elo in the model, then we can use the model as an Elocator.
- Build model to predict accuracy given the board state.
- Use model as an input to various game and/or Elo evaluations. E.g., expected accuracy at a given Elo, standard deviation of accuracy at given Elo, etc.