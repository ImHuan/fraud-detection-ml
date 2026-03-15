import pandas as pd
from src.config import CONFIG
from src.preprocess import preprocess_data, split_data, apply_smote
from src.train import train_xgboost, compute_scale_pos_weight
from src.evaluate import evaluate


def main():
    df = pd.read_csv(CONFIG["data_path"])

    df_clean = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(
        df_clean,
        CONFIG["target_column"],
        CONFIG["test_size"],
        CONFIG["random_state"]
    )

    # ========================
    # Model 1: Baseline
    # ========================
    print("===== Baseline (No SMOTE, No scale_pos_weight) =====")

    model_baseline = train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        CONFIG["xgb_params"]
    )

    evaluate(model_baseline, X_test, y_test, CONFIG["threshold"])


    # ========================
    # Model 2: SMOTE + scale_pos_weight
    # ========================
    print("\n===== SMOTE + scale_pos_weight =====")

    X_smote, y_smote = apply_smote(
        X_train,
        y_train,
        CONFIG["random_state"]
    )

    weight = compute_scale_pos_weight(y_smote)

    params = CONFIG["xgb_params"].copy()
    params["scale_pos_weight"] = weight

    model_improved = train_xgboost(
        X_smote,
        y_smote,
        X_test,
        y_test,
        params
    )

    evaluate(model_improved, X_test, y_test, CONFIG["threshold"])


if __name__ == "__main__":
    main()