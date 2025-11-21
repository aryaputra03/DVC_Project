import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
import yaml
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def handle_missing_values(df: pd.DataFrame, params: dict)-> pd.DataFrame:
    df_clean = df.copy()
    numerical_col = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_col = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    if 'target' in numerical_col:
        numerical_col.remove('target')
    
    num_strategy = params['preprocessing']['missing_value_strategy_numerical']
    print(f"   - Numerical imputation strategy: {num_strategy}")

    num_imputer = SimpleImputer(strategy=num_strategy)
    df_clean[numerical_col] = num_imputer.fit_transform(df_clean[numerical_col])

    cat_strategy = params['preprocessing']['missing_value_strategy_categorical']
    print(f"   - Categorical imputation strategy: {cat_strategy}")

    cat_imputer = SimpleImputer(strategy=cat_strategy)
    df_clean[categorical_col] = cat_imputer.fit_transform(df_clean[categorical_col])

    return df_clean

def trasnform_skewed_features(df: pd.DataFrame, params: dict)-> pd.Dataframe:
    df_transformed = df.copy()
    skewed_features = params['preprocessing']['skewed_features']
    skew_threshold = params['preprocessing']['skew_threshold']

    for col in skewed_features:
        if col in df_transformed.columns:
            skewness = df_transformed[col].skew()
            print(f"   - {col}: skewness = {skewness:.3f}", end="")
            if abs(skewness) > skew_threshold:
                pt = PowerTransformer(method='yeo-johnson')
                df_transformed[col] = pt.fit_transform(df_transformed[[col]])
                new_skewness = df_transformed[col].skew()
                print(f" → transformed to {new_skewness:.3f}")
            else:
                print(" → no transformation needed")
    return df_transformed

def encode_categorical_feature(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.to_list()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"   - {col}: {len(le.classes_)} categories encoded")
    return df_encoded, label_encoders

def scale_numerical_features(df: pd.DataFrame, params: dict)->pd.DataFrame:
    df_scaled = df.copy()
    if params['preprocessing']['scale_features']:
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numerical_cols:
            numerical_cols.remove('target')

        scaler = StandardScaler()
        df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])

        print(f"   - Scaled {len(numerical_cols)} numerical features")
    else:
        print("   - Scaling disabled in params")

def remove_outliers(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_clean = df.copy()

    if params["preprocessing"]['remove_outliers']:
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numerical_cols:
            numerical_cols.remove('target')
        initial_rows = len(df_clean)

        for col in numerical_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

        removed = initial_rows - len(df_clean)
        print(f"   - Removed {removed} outlier rows ({removed/initial_rows*100:.1f}%)")
    else:
        print("   - Outlier removal disabled in params")
    
    return df_clean

def preprocess_data(input_path: str, output_path: str):
    params = load_params()
    df = pd.read_csv(input_path)
    print(f"   - Shape: {df.shape}")
    print(f"   - Missing values: {df.isnull().sum().sum()}")

    df = handle_missing_values(df, params)
    df = remove_outliers(df, params)
    df = trasnform_skewed_features(df, params)
    df, encoders = encode_categorical_feature(df)
    df = scale_numerical_features(df, params)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE")
    print("="*60)
    print(f"📂 Output saved to: {output_path}")
    print(f"   - Final shape: {df.shape}")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    print(f"   - File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('--input', type=str, default='data/raw/dataset.csv')
    parser.add_argument('--output', type=str, default='data/processed/dataset_cleaned.csv')
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output)


if __name__ == "__main__":
    main()