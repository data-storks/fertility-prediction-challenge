"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera

"""
def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")

    # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df['new_child'].isna()]  
    
    selected_columns = ["cf20m004","cf20m024","cf20m128","cf20m130"]

    categorical_preprocessor = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown="ignore")
    )

    preprocessor = ColumnTransformer([
        ('one-hot-encoder', categorical_preprocessor, selected_columns)])

    logistic_pipeline = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
    
    ## Append  oversample

    X_oversampled, y_oversampled = resample(model_df[selected_columns][model_df['new_child'] == 1],
                                        model_df['new_child'][model_df['new_child'] == 1],
                                        replace = True,
                                        n_samples = model_df[selected_columns][model_df['new_child'] == 0].shape[0],
                                        random_state = 2024)
    X_balanced = pd.concat((model_df[selected_columns][model_df['new_child'] == 0], X_oversampled))
    y_balanced = pd.concat((model_df['new_child'][model_df['new_child'] == 0], y_oversampled))
        
    
    # fit the model
    logistic_pipeline.fit(X_balanced,y_balanced)
    
    # Save the model
    joblib.dump(logistic_pipeline, "model.joblib")
