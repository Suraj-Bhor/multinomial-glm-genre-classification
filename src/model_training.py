from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def train_models(data):
    # Prepare features and target
    X = data.drop(['popularity', 'track_genre', 'track_genre_encoded', 'popularity_category'], axis=1)
    y = data['track_genre_encoded']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression model
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    return {
        'model': model,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test
    }