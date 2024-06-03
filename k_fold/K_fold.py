def k_fold (models, X, y):

    """
    Find MAE and MSE scores for each model using K-fold cross validation.

    Parameters
    ----------
    models : model that has completed the learning you want to evaluate.
    X : feature except target.
    y : target feature.

    
    Returns
    -------
    float
        The average MAE score and MSE score for each model.

    Examples
    --------
    >>> k_fold(ridge, X, y)
    983, 6288443


    """

    kf = KFold(n_splits=10, shuffle = True, random_state= 42)
    mae_scores = {}
    mse_scores = {}

    # Iterate over each model
    for name, model in models.items():
        # Initialize lists to store evaluation metrics for each fold
        mae_scores[name] = []
        mse_scores[name] = []
        
        # Iterate over each fold
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate evaluation metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Append scores to lists
            mae_scores[name].append(mae)
            mse_scores[name].append(mse)
    
    # Calculate average scores over all folds for each model
    avg_mae = {name: sum(scores) / len(scores) for name, scores in mae_scores.items()}
    avg_mse = {name: sum(scores) / len(scores) for name, scores in mse_scores.items()}
    
    return avg_mae, avg_mse