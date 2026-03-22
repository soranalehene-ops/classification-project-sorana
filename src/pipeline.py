import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "Depression"

NUMERIC_FEATURES = [
    "Academic Pressure",
    "Financial Stress",
    "Age",
    "Work/Study Hours",
    "Study Satisfaction",
    "CGPA",
]

CATEGORICAL_FEATURES = [
    "Dietary Habits",
    "Sleep Duration",
    "Have you ever had suicidal thoughts ?",
]

SELECTED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataframe without modifying the original input.
    """
    df_clean = df.copy()

    # Standardize column names
    df_clean.columns = df_clean.columns.str.strip()

    # Replace placeholder missing values
    df_clean = df_clean.replace(["?", " ?", "", "NA", "N/A", "na", "null"], np.nan)

    # Drop non-predictive identifier if present
    if "id" in df_clean.columns:
        df_clean = df_clean.drop(columns=["id"])

    # Convert target
    if TARGET_COL in df_clean.columns:
        df_clean[TARGET_COL] = pd.to_numeric(df_clean[TARGET_COL], errors="coerce").astype("Int64")

    # Convert numeric predictors
    numeric_candidates = [
        "Age",
        "Academic Pressure",
        "Work Pressure",
        "CGPA",
        "Study Satisfaction",
        "Job Satisfaction",
        "Work/Study Hours",
        "Financial Stress",
    ]

    for col in numeric_candidates:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    return df_clean


def get_feature_list(include_sensitive: bool = True) -> list:
    """
    Return selected feature list.
    Set include_sensitive=False to exclude suicidal-thoughts feature.
    """
    if include_sensitive:
        return SELECTED_FEATURES.copy()

    return [
        "Academic Pressure",
        "Financial Stress",
        "Age",
        "Work/Study Hours",
        "Study Satisfaction",
        "CGPA",
        "Dietary Habits",
        "Sleep Duration",
    ]


def build_preprocessor(include_sensitive: bool = True) -> ColumnTransformer:
    """
    Build preprocessing pipeline for selected columns.
    """
    feature_list = get_feature_list(include_sensitive=include_sensitive)

    num_cols = [col for col in NUMERIC_FEATURES if col in feature_list]
    cat_cols = [col for col in CATEGORICAL_FEATURES if col in feature_list]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    return preprocessor


def split_data(
    df: pd.DataFrame,
    include_sensitive: bool = True,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Clean data, select features, and create a stratified train-test split.
    """
    df_clean = clean_data(df)
    feature_list = get_feature_list(include_sensitive=include_sensitive)

    X = df_clean[feature_list].copy()
    y = df_clean[TARGET_COL].copy()

    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def fit_transform_splits(
    df: pd.DataFrame,
    include_sensitive: bool = True,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Split data, fit preprocessor on training data only, and transform train/test.
    """
    X_train, X_test, y_train, y_test = split_data(
        df,
        include_sensitive=include_sensitive,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(include_sensitive=include_sensitive)

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Log shapes
    print("X_train raw shape:", X_train.shape)
    print("X_test raw shape:", X_test.shape)
    print("X_train transformed shape:", X_train_t.shape)
    print("X_test transformed shape:", X_test_t.shape)

    # Quick assertion: same transformed column count across splits
    assert X_train_t.shape[1] == X_test_t.shape[1], "Mismatch in transformed feature count across splits."

    return preprocessor, X_train_t, X_test_t, y_train, y_test
