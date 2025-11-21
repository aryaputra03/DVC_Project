import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from joblib import load
import yaml
import os
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_model(model_path: str):
    print(f"📂 Loading model from: {model_path}")
    model = load(model_path)
    return model

def load_test_data(data_path: str):
    df = pd.read_csv(data_path)
    X_test = df.drop('target', axis=1)
    y_test = df['target']
    print(f"   - Test samples: {len(X_test)}")
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted')),
        'recall': float(recall_score(y_test, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
        'precision_binary': float(precision_score(y_test, y_pred)),
        'recall_binary': float(recall_score(y_test, y_pred)),
        'f1_binary': float(f1_score(y_test, y_pred))
    }

    if y_prob is not None:
        metrics['auc_roc'] = float(roc_auc_score(y_test, y_prob))
    
    cm = confusion_matrix(y_test, y_pred)
    metrics['true_negatives'] = int(cm[0][0])
    metrics['false_positives'] = int(cm[0][1])
    metrics['false_negatives'] = int(cm[1][0])
    metrics['true_positives'] = int(cm[1][1])

    report = classification_report(y_test, y_pred)

    return metrics, report, cm

def print_evaluation_result(metrics: dict, report: str, cm: np.ndarray):
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    print("\n🎯 Main Metrics:")
    print(f"   - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   - Precision: {metrics['precision']:.4f}")
    print(f"   - Recall:    {metrics['recall']:.4f}")
    print(f"   - F1 Score:  {metrics['f1_score']:.4f}")

    if 'auc_roc' in metrics:
        print(f"    -AUC-ROC: {metrics['auc_roc']:.4f}")
        print("\n📈 Confusion Matrix:")
        print(f"                    Predicted")
        print(f"                  Neg     Pos")
        print(f"   Actual Neg   {cm[0][0]:5d}   {cm[0][1]:5d}")
        print(f"   Actual Pos   {cm[1][0]:5d}   {cm[1][1]:5d}")
        
        print("\n📋 Classification Report:")
        print(report)

def save_metrics(metrics: dict, report: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved to: {metrics_path}")

    report_path = os.path.join('report', 'classification_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {metrics['f1_score']:.4f}\n")
        if 'auc_roc' in metrics:
            f.write(f"AUC-ROC:   {metrics['auc_roc']:.4f}\n")
        f.write("\n")
        f.write(report)
    print(f"💾 Report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, default='models/model.pkl')
    parser.add_argument('--data', type=str, default='data/processed/test_data.csv')
    parser.add_argument('--output', type=str, default='metrics')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🔍 MODEL EVALUATION PIPELINE")
    print("="*60)

    model = load_model(args.model)
    X_test, y_test = load_test_data(args.data)
    metrics, report, cm = evaluate_model(model, X_test, y_test)
    print_evaluation_result(metrics, report, cm)
    save_metrics(metrics, report, args.output)

    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()