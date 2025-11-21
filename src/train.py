import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump
import yaml
import os
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_data(input_path: str):
    df = pd.read_csv(input_path)
    X = df.drop('target', axis=1)
    y = df['target']

    print(f"   - Features: {X.shape[1]}")
    print(f"   - Samples: {X.shape[0]}")
    print(f"   - Target distribution: {dict(y.value_counts())}")

    return X, y

def handle_imbalance(X, y, params):
    if params['training']['handle_imbalance']:
        strategy = params['training']['imbalance_strategy']
        
        if strategy == 'smote':
            smote = SMOTE(random_state=params['training']['random_state'])
            X_resampled, y_resampled = smote.fit_resample(X,y)
            print(f"   - SMOTE applied")
            print(f"   - Before: {dict(pd.Series(y).value_counts())}")
            print(f"   - After: {dict(pd.Series(y_resampled).value_counts())}")
            return X_resampled, y_resampled
        print(" - No resampling applied")
        return X, y
    
def get_model(params):
    model_type = params['training']['model_type']
    model_params = params['training']['model_params'].get(model_type, {})
    random_state = params['training']['random_state']

    print(f"🤖 Initializing model: {model_type}")
    print(f"   - Parameters: {model_params}")

    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 10),
            min_samples_split=model_params.get('min_samples_split', 2),
            min_samples_leaf=model_params.get('min_samples_leaf', 1),
            random_state=random_state,
        )
    elif model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 6),
            learning_rate=model_params.get('learning_rate', 0.1),
            subsample=model_params.get('subsample', 0.8),
            colsample_bytree=model_params.get('colsample_bytree', 0.8),
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            C=model_params.get('C', 1.0),
            max_iter=model_params.get('max_iter', 1000),
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model

def train_model(X, y, params):
    X_balanced, y_balanced = handle_imbalance(X, y, params)
    test_size = params['training']['test_size']
    random_state = params['training']['random_state']

    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced,
        test_size=test_size,
        random_state=random_state,
        stratify=y_balanced
    )
    print(f"\n📊 Data split:")
    print(f"   - Train size: {len(X_train)}")
    print(f"   - Test size: {len(X_test)}")

    model = get_model(params)
    cv_folds = params['training']['cv_folds']
    print(f"\n🔄 Running {cv_folds}-fold cross-validation...")

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    print(f"   - CV Scores: {cv_scores}")
    print(f"   - Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    print("\n🏋️ Training final model on full training set...")
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"   - Training accuracy: {train_score:.4f}")
    print(f"   - Test accuracy: {test_score:.4f}")

    if hasattr(model, 'feature_importances_'):
        importaces = pd.DataFrame({
            'feature':X.columns,
            'importance':model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n📊 Top 5 Feature Importances:")
        for _, row in importaces.head().iterrows():
            print(f"   - {row['feature']}: {row['importance']:.4f}")
            
    training_info = {
        'model_type': params['training']['model_type'],
        'cv_mean_score': float(cv_scores.mean()),
        'cv_std_score': float(cv_scores.std()),
        'train_accuracy' : float(train_score),
        'test_accuracy' : float(test_score),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X.shape[1]
    }
    return model, X_test, y_test, training_info

def save_model(model, output_path: str, training_info:dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dump(model, output_path)
    print(f"\n💾 Model saved to: {output_path}")

    info_path = output_path.replace('.pkl', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"💾 Training info saved to: {info_path}")

def main():
    parser = argparse.ArgumentParser(description='Train classification model')
    parser.add_argument('--input', type=str, default='data/processed/dataset_cleaned.csv')
    parser.add_argument('--output', type=str, default='models/model.pkl')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 MODEL TRAINING PIPELINE")
    print("="*60)

    params = load_params()

    X,y = load_data(args.input)

    model, X_test, y_test, training_info = train_model(X, y, params)

    save_model(model, args.output, training_info)

    test_data = pd.concat([X_test.reset_index(drop=True),
                           y_test.reset_index(drop=True)],
                           axis=1)
    test_data.to_csv('data/processed/test_data.csv', index=False)

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()