import xgboost as xgb


def train_xgboost(X_train, y_train, X_val, y_val, params):

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def compute_scale_pos_weight(y_train):

    neg = sum(y_train == 0)
    pos = sum(y_train == 1)

    return neg / pos