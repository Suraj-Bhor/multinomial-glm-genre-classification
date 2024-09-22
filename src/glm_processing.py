import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt


def one_hot_encode(data, columns):
    """One-hot encode the specified categorical columns."""
    return pd.get_dummies(data, columns=columns, drop_first=True)

def ensure_numeric(data):
    """Convert columns in the dataframe to numeric, if possible."""
    return data.apply(pd.to_numeric, errors='coerce')

def fit_glm(y, X, family=sm.families.Gaussian()):
    """Fit a GLM model with the specified response variable and predictors."""
    # Ensure all data is numeric
    X = ensure_numeric(X)
    y = pd.to_numeric(y, errors='coerce')
    
    # Check for any missing values or invalid data
    if X.isnull().values.any() or y.isnull().values.any():
        #print which column has missing values
        print(X.columns[X.isnull().any()])
        raise ValueError("X or y contains NaN values. Ensure all data is clean.")

    # Fit the GLM model
    return sm.GLM(y, X, family=family).fit()

def remove_non_significant_predictors(glm_model, X, p_value_threshold=0.05):
    """Remove predictors from X that have p-values above the specified threshold."""
    non_significant_predictors = glm_model.pvalues[glm_model.pvalues > p_value_threshold].index.tolist()
    return X.drop(columns=non_significant_predictors)

def calculate_vif(X):
    """Calculate Variance Inflation Factor (VIF) for each predictor."""
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values(by="VIF", ascending=False)

def save_residual_plots(glm_model, residuals, output_dir):
    """Save residuals, histogram, and Q-Q plot as files instead of showing them."""
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # Residual plot
    sns.residplot(x=glm_model.fittedvalues, y=residuals, lowess=True, ax=axes[0], line_kws={'color': 'red', 'lw': 1})
    axes[0].set_title('Residual Plot')
    axes[0].set_xlabel('Fitted values')
    axes[0].set_ylabel('Residuals')

    # Histogram
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set_title('Histogram of Residuals')
    axes[1].set_xlabel('Residuals')

    # Q-Q plot
    sm.qqplot(residuals, fit=True, line='45', ax=axes[2])
    axes[2].set_title('Q-Q Plot')

    plt.tight_layout()
    # Save the plots instead of showing them
    plt.savefig(f'{output_dir}/residual_plots.png', dpi=300)
    plt.close()

def handle_missing_values(X, strategy='mean'):
    """
    Handle missing values in the dataset.
    You can choose to fill NaN values with 'mean', 'median', or 'drop' the rows.
    """
    if strategy == 'mean':
        X = X.fillna(X.mean())
    elif strategy == 'median':
        X = X.fillna(X.median())
    elif strategy == 'drop':
        X = X.dropna()
    return X

def add_polynomial_terms(X, degree=3):
    """
    Add polynomial terms (cubed and quartic) for continuous predictors in X.
    Only adds terms if the column exists and is numeric.
    """
    continuous_columns = ['danceability', 'energy', 'loudness', 'valence']
    
    for col in continuous_columns:
        if col in X.columns:  # Check if the column exists
            # Ensure the column is numeric and handle missing values
            X[col] = pd.to_numeric(X[col], errors='coerce')  # Convert to numeric (if any string values)
            if X[col].notnull().all():  # Check that there are no NaN values
                try:
                    X[f'{col}_cubed'] = X[col] ** degree
                    X[f'{col}_quartic'] = X[col] ** (degree + 1)
                except Exception as e:
                    print(f"Error while adding polynomial terms for '{col}': {e}")
            else:
                print(f"Column '{col}' contains NaN values, handling them before polynomial terms.")
                # Handle NaN values by filling or dropping
                X = handle_missing_values(X, strategy='mean')
                X[f'{col}_cubed'] = X[col] ** degree
                X[f'{col}_quartic'] = X[col] ** (degree + 1)
        else:
            print(f"Column '{col}' not found in the DataFrame, skipping polynomial terms.")
    
    return X


def print_important_glm_info(glm_model):
    """Print important parts of the GLM summary to the CLI."""
    print("\nImportant GLM Summary Information:")
    print(f"Deviance: {glm_model.deviance}")
    print(f"Pearson Chi2: {glm_model.pearson_chi2}")
    print(f"Null Deviance: {glm_model.null_deviance}")
    print(f"AIC: {glm_model.aic}")
    print(f"BIC: {glm_model.bic}")
    print("\nCoefficients with p-values:")
    print(glm_model.params.head())  # Print only the top part of the coefficients
    print(glm_model.pvalues.head())  # Print only the top part of the p-values

def glm_processing_pipeline(dataset, response_var, output_dir):
    """
    Pipeline for processing a GLM model:
    - One-hot encodes categorical variables
    - Fits a GLM model
    - Removes non-significant predictors
    - Checks for multicollinearity
    - Adds polynomial terms and refits the model
    - Saves important outputs to files
    """
    # Define predictors and response
    predictors = ['danceability', 'energy', 'loudness', 'valence', 'explicit', 'acousticness', 'instrumentalness', 'liveness', 'tempo']
    dataset['explicit'] = dataset['explicit'].astype(int)
    X = dataset[predictors]
    # One-hot encode categorical variables
    categorical_cols = one_hot_encode(dataset, columns=['key', 'mode', 'time_signature', 'track_genre', 'popularity_category'])
    # Prepare predictors and response
    X = pd.concat([X, categorical_cols], axis=1)
    
    # Adding a constant for the intercept
    X = sm.add_constant(X)
    
    #drop popularity from predictors
    X = X.drop(columns=['popularity'])
    
    # Define the response variable
    y = dataset[response_var]
    # Fit the initial GLM
    glm_initial = fit_glm(y, X)
    
    # Print important GLM information on the CLI
    print_important_glm_info(glm_initial)
    
    # Save the full GLM summary to a file and create a file if it doesn't exist
    with open(f'{output_dir}/initial_glm_summary.txt', 'w') as f:
        f.write(glm_initial.summary().as_text())

    # Remove non-significant predictors
    X_simplified = remove_non_significant_predictors(glm_initial, X)
    glm_simplified = fit_glm(y, X_simplified)
    
    # Print and save simplified GLM model info
    print("\nSimplified Model Summary:")
    print_important_glm_info(glm_simplified)
    with open(f'{output_dir}/simplified_glm_summary.txt', 'w') as f:
        f.write(glm_simplified.summary().as_text())

    # Calculate VIF for the simplified model
    vif_data = calculate_vif(X_simplified)
    vif_data.to_csv(f'{output_dir}/tables/vif_values.csv', index=False)

    # Save residual plots
    residuals = glm_simplified.resid_response
    save_residual_plots(glm_simplified, residuals, output_dir)

    # Add polynomial terms
    # X_simplified = add_polynomial_terms(X_simplified)
    # glm_higher_order = fit_glm(y, X_simplified)

    # # Print important info for the higher-order model
    # # print("\nHigher Order Model Summary:")
    # # print_important_glm_info(glm_higher_order)
    
    # # # Save higher-order model summary to file
    # # with open(f'{output_dir}/higher_order_glm_summary.txt', 'w') as f:
    # #     f.write(glm_higher_order.summary().as_text())

    return glm_simplified, vif_data