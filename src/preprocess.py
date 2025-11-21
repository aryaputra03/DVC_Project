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
