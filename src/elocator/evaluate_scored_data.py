import pandas as pd
import numpy as np
import plotly.express as px

# from utils import fen_encoder, fen_decode

df = pd.read_pickle('./data/scored.pkl')
df.Accuracy /= 100

# Step 1: Bin the 'ScoreBefore' column
bins = [-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf]
labels = ['<-2.5', '-2.5 to -1.5', '-1.5 to -0.5', '-0.5 to 0.5', '0.5 to 1.5', '1.5 to 2.5', '>2.5']
df['ScoreBeforeBins'] = pd.cut(df['ScoreBefore'], bins=bins, labels=labels)

# Step 2: Group by the new bins and calculate mean, then melt for long format
grouped_data = df.groupby('ScoreBeforeBins').agg({'Accuracy': 'mean', 'ModelPredictions': 'mean'}).reset_index()
melted_data = grouped_data.melt(id_vars='ScoreBeforeBins', value_vars=['Accuracy', 'ModelPredictions'],
                                var_name='Metric', value_name='Value')

# Step 3: Plot the data using Plotly Express
fig = px.line(melted_data, x='ScoreBeforeBins', y='Value', color='Metric',
              title='Mean Model Predictions and Accuracy by ScoreBefore Bins',
              labels={'ScoreBeforeBins': 'Score Before Bins', 'Value': 'Mean Value'},
              markers=True)

fig.update_layout(xaxis_title='Score Before Bins',
                  yaxis_title='Mean Value',
                  legend_title_text='Metric')
fig.show()



# df['AccRel'] = df.ModelPredictions / df.Accuracy
# df['AccDiff'] = df.ModelPredictions - df.Accuracy

def split_games(df: pd.DataFrame) -> pd.DataFrame:
    '''Function takes dataframe of training mvoes as input and breaks it back into player-games'''

    default_fen = df.iloc[0].FEN

    df['BeginGame'] = df.FEN.apply(lambda x: x == default_fen)
    df['GameID'] = df.BeginGame.cumsum()
    
    color = [1]
    for i in range(1, df.FEN.shape[0]):
        if (df.BeginGame[i] == True) | (color[-1] == 0):
            # if beginning of game, or if last move was 0 (black) then white (1)
            color.append(1)
        else:
            color.append(0)
    df['Color'] = color

    df = df[[
        'GameID',
        'Color',
        'Elo',
        'AccDiff'
    ]]
    df = df.groupby(['GameID', 'Color', 'Elo']).sum().reset_index()
    
    return df

class ChessPosition():
    def __init__(self, fen_encoded):
         self.fen_encoded = fen_encoded
    
    def get_material_count(self):
        # Convert the encoded FEN back to 8x8x12, ignoring en passant and castling info
        board_encoded = np.array(self.fen_encoded[:-12]).reshape((8, 8, 12))
        
        # Piece values
        piece_values = {
            0: -1,  # Black Pawn
            1: -3,  # Black Knight
            2: -3,  # Black Bishop
            3: -5,  # Black Rook
            4: -9,  # Black Queen
            6: 1,  # White Pawn
            7: 3,  # White Knight
            8: 3,  # White Bishop
            9: 5,  # White Rook
            10: 9,  # White Queen
        }
        
        # Initialize material count
        material_count = 0
        
        # Calculate material count
        for piece_index, value in piece_values.items():
            count = np.sum(board_encoded[:, :, piece_index])
            material_count += count * value
        
        return material_count
    
    def queen_on_board(self):
        # Assuming encoded is a flattened list from fen_encoder, convert back to 8x8x12
        # Remove the last 12 elements (8 for en passant and 4 for castling rights) before reshaping
        board_encoded = np.array(self.fen_encoded[:-12]).reshape((8, 8, 12))
        
        # Indexes for queens, adjust these if your encoding differs
        white_queen_index = 10
        black_queen_index = 4
        
        # Check for the presence of a queen
        white_queen_present = np.any(board_encoded[:, :, white_queen_index] == 1)
        black_queen_present = np.any(board_encoded[:, :, black_queen_index] == 1)
        
        return white_queen_present or black_queen_present

print('starting to calculate material count and queen on board...')

df['MaterialCount'] = df.FEN.apply(lambda x: ChessPosition(x).get_material_count())

print('finished calculating material count')

print('starting to calculate queen on board...')

df['QueenOnBoard'] = df.FEN.apply(lambda x: ChessPosition(x).queen_on_board())

print('finished calculating queen on board')

# plot accuracy vs model prediction with material count and queen on board as the x axis in two separate graphs

#  take the mean of accuracy and model predictions for each material count and queen on board
# take the absolute value of material count

df['MaterialCount'] = df['MaterialCount'].abs()

# create bins of exactly 0, 1, 2, 3+ for material count
bins = [-np.inf, 0, 1, 2, np.inf]
labels = ['0', '1', '2', '3+']
df['MaterialCount'] = pd.cut(df['MaterialCount'], bins=bins, labels=labels)

grouped_data = df.groupby('MaterialCount').agg({'Accuracy': 'mean', 'ModelPredictions': 'mean'}).reset_index()
melted_data = grouped_data.melt(id_vars='MaterialCount', value_vars=['Accuracy', 'ModelPredictions'],
                                var_name='Metric', value_name='Value')

fig = px.line(melted_data, x='MaterialCount', y='Value', color='Metric',
                title='Mean Model Predictions and Accuracy by Material Count',
                labels={'MaterialCount': 'Material Count', 'Value': 'Mean Value'},
                markers=True)

fig.update_layout(xaxis_title='Material Count',
                    yaxis_title='Mean Value',
                    legend_title_text='Metric')
fig.show()

# plot simple graph with queen on board or not on the x axis

grouped_data = df.groupby('QueenOnBoard').agg({'Accuracy': 'mean', 'ModelPredictions': 'mean'}).reset_index()
melted_data = grouped_data.melt(id_vars='QueenOnBoard', value_vars=['Accuracy', 'ModelPredictions'],
                                var_name='Metric', value_name='Value')

fig = px.line(melted_data, x='QueenOnBoard', y='Value', color='Metric',
                title='Mean Model Predictions and Accuracy by Queen On Board',
                labels={'QueenOnBoard': 'Queen On Board', 'Value': 'Mean Value'},
                markers=True)

fig.update_layout(xaxis_title='Queen On Board',
                    yaxis_title='Mean Value',
                    legend_title_text='Metric')
fig.show()




# df = split_games(df=df)

# # print(df.head(68))
# print(df)

# # create a scatter plot of Elo and AccDiff

# fig = px.scatter(df, x = 'AccDiff', y = 'Elo', )
# fig.show()