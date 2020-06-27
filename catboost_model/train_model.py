from catboost import Pool, CatBoostClassifier, cv
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np


FORCE_REBUILD = False


cwd = os.getcwd()

def save_prediction(predicted_values):
    # Saving submission file
    submission_df = pd.read_csv(os.path.join(cwd, "dataset", "submission.csv"))
    subm = submission_df
    subm['Survived'] = predicted_values
    subm.to_csv(os.path.join(cwd, "dataset", "submission_predicted.csv"), index=False)

def fit_catboost_cv(model, params, X, y):
    # Fit model using catboost grid
    best_params = model.grid_search(params, X=X, y=y, verbose=False)
    return best_params

def fit_sklearn_gridsearchcv(model, params, X, y):
    # Fit model using sklearn gridsearch and cross validation
    gcv = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1)
    gcv.fit(X=X, y=y)
    return gcv.best_params_

def get_datasets():
    # Getting training data
    train = pd.read_csv(os.path.join(cwd, "dataset", "train_processed.csv"))
    X_train = train[['Pclass', 'Sex', 'Fare', 'Embarked']]
    # Training labels 
    y_train = train[['Survived']]

    # Getting test data
    test = pd.read_csv(os.path.join(cwd, "dataset", "test_processed.csv"))
    X_test = test[['Pclass', 'Sex', 'Fare', 'Embarked']]

    return X_train, y_train, X_test

def fit_model(X_train, y_train):
    # Setting cat features
    cat_f = [0, 1, 3]

    # setting pool
    train_pool = Pool(X_train, y_train, cat_features=cat_f)

    # Setting parameters for grid search
    params = {
        "iterations" : [410],
        "learning_rate" : [0.06],
        "depth" : [7],
        "l2_leaf_reg" : [0.1, 0.5, 0.8],
        "bagging_temperature" : [0.75, 1, 1.25],
        "leaf_estimation_method" : ["Newton"]
    }

    # model setting
    model = CatBoostClassifier(loss_function="Logloss", cat_features=cat_f, boosting_type="Ordered")

    # Sklearn gridsearch
    sklearn_params = fit_sklearn_gridsearchcv(model, params, X_train, y_train)

    # Catboost sv
    #cv_params = fit_catboost_cv(model, params, X_train, y_train)

    #print('Best cv params: {0}'.format(cv_params['params']))
    print('Best sklearn params: {0}'.format(sklearn_params))
    result_model = CatBoostClassifier(iterations=sklearn_params['iterations'], 
                                        learning_rate=sklearn_params['learning_rate'],
                                        depth=sklearn_params['depth'],
                                        loss_function='Logloss',
                                        od_type="Iter",
                                        od_wait=30)

    result_model.fit(X_train, y_train, cat_features=cat_f)
    result_model.save_model(os.path.join(cwd, "models", "catboost_classifier.cbm"))   
    return result_model

if __name__ == '__main__':
    # Getting dataset for training
    X_train, y_train, X_test = get_datasets()

    # Fitting model if FORCE_REBUILD, else loading from folder
    if FORCE_REBUILD:
        # Fitting model
        model = fit_model(X_train, y_train)
    else:
        # loading model
        path_model = os.path.join(cwd, "models", "catboost_classifier.cbm")
        if os.path.exists(path_model):
            model = CatBoostClassifier().load_model(fname = os.path.join(cwd, "models", "catboost_classifier.cbm"))
        else:
            model = fit_model(X_train, y_train)

    # Making prediction and saving
    predicted = model.predict(X_test, prediction_type="Class")
    save_prediction(predicted)
