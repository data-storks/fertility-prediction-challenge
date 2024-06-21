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
    
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_df, outcome_df['new_child'], test_size=0.30, random_state=42)

    # make pipeline
    model = make_pipeline(categorical_preprocessor, LogisticRegression(max_iter=500))
    
    # fit the model
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, "model.joblib")
