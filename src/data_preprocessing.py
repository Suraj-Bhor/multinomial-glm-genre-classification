import pandas as pd
import numpy as np

def preprocess(data):
    # Remove unnecessary columns
    data = data.drop(columns=['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name'])
    
    # Handle missing values
    # data = data.dropna()
    
    # Convert 'explicit' to boolean
    data['explicit'] = data['explicit'].astype(bool)
    
    # Log transform 'duration_ms'
    data['duration_ms'] = np.log(data['duration_ms'] + 1)
    
    # Inverse transform 'loudness'
    data['loudness'] = 1 / (data['loudness'].max() - data['loudness'] + 1)
    
    # Power transform 'speechiness'
    data['speechiness'] = data['speechiness'] ** (1/5)
    
    # Log transform 'acousticness'
    data['acousticness'] = np.log(data['acousticness'] + 1)
    
    return data