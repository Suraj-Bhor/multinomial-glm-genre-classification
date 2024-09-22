import pandas as pd
from tqdm import tqdm
from src import data_preprocessing, exploratory_data_analysis, feature_engineering, model_training, model_evaluation, visualization, glm_processing
from src.logger import setup_logger

def main():
    # Set up logger
    logger = setup_logger()
    logger.info("Starting Spotify Tracks Analysis")

    # Load the dataset
    logger.info("Loading dataset")
    data = pd.read_csv('data/dataset.csv')
    logger.info(f"Dataset loaded with {len(data)} rows and {len(data.columns)} columns")

    # Create a tqdm object for progress tracking
    steps = ['Preprocessing', 'EDA', 'Feature Engineering', 'Model Training', 'Model Evaluation', 'Visualization']
    progress_bar = tqdm(steps, desc="Analysis Progress")

    # Preprocess the data
    logger.info("Preprocessing data")
    preprocessed_data = data_preprocessing.preprocess(data)
    progress_bar.update(1)

    # Perform exploratory data analysis
    logger.info("Performing exploratory data analysis")
    exploratory_data_analysis.perform_eda(preprocessed_data)
    progress_bar.update(1)

    # Engineer features
    logger.info("Engineering features")
    featured_data = feature_engineering.engineer_features(preprocessed_data)
    progress_bar.update(1)
    
    response_var = 'popularity'
    output_dir = 'outputs'

    final_model, vif_data = glm_processing.glm_processing_pipeline(preprocessed_data, response_var, output_dir)

    # Train models
    logger.info("Training models")
    models = model_training.train_models(featured_data)
    progress_bar.update(1)

    # Evaluate models
    logger.info("Evaluating models")
    evaluation_results = model_evaluation.evaluate_models(models, featured_data)
    progress_bar.update(1)

    # Visualize results
    logger.info("Visualizing results")
    visualization.visualize_results(evaluation_results, featured_data)
    progress_bar.update(1)

    logger.info("Analysis completed successfully")
    progress_bar.close()

if __name__ == "__main__":
    main()