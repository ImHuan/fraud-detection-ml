from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def preprocess_data(df):
    df_clean = df.copy()

    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna()
    return df_clean

def split_data(df, target, test_size, random_state):

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state):

    smote = SMOTE(random_state=random_state)

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    return X_resampled, y_resampled