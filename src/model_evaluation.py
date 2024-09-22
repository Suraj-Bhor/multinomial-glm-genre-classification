from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def evaluate_models(models, data):
    model = models['model']
    X_test = models['X_test']
    y_test = models['y_test']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('outputs/tables/classification_report.csv')
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=data['track_genre'].unique(), columns=data['track_genre'].unique())
    cm_df.to_csv('outputs/tables/confusion_matrix.csv')
    
    return {
        'model': model,
        'classification_report': report_df,
        'confusion_matrix': cm_df
    }