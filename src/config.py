CONFIG = {

    "data_path": "data/creditcard.csv",

    "target_column": "Class",

    "test_size": 0.2,

    "random_state": 42,

    "threshold": 0.5,

    "xgb_params": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 300
    }
}