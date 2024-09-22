import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import src.model_training as model_training

def visualize_results(evaluation_results, data):
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(evaluation_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('outputs/figures/confusion_matrix.png')
    plt.close()
    
    # Visualize feature importance
    feature_importance = pd.DataFrame({
        'feature': data.drop(['popularity', 'track_genre', 'track_genre_encoded', 'popularity_category'], axis=1).columns,
        'importance': abs(evaluation_results['model'].coef_[0])
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Feature Importance')
    plt.savefig('outputs/figures/feature_importance.png')
    plt.close()