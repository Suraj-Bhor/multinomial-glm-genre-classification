import pandas as pd
from sklearn.preprocessing import LabelEncoder

def engineer_features(data):
    # Create 'spoken_words' feature
    data['spoken_words'] = (data['speechiness'] > 0.66).astype(int)
    
    # Bin 'popularity' into categories
    data['popularity_category'] = pd.cut(data['popularity'], bins=3, labels=['Low', 'Medium', 'High'])
    
    # One-hot encode categorical variables
    data = pd.get_dummies(data, columns=['mode', 'key', 'time_signature'])
    
    # Label encode 'track_genre'
    le = LabelEncoder()
    data['track_genre_encoded'] = le.fit_transform(data['track_genre'])
    
    return data