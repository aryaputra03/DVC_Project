# 🚀 DVC Classification Project

**Final Project - Data Version Control untuk Machine Learning Pipeline**

[![DVC](https://img.shields.io/badge/-DVC-945DD6?style=flat&logo=dataversioncontrol&logoColor=white)](https://dvc.org)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)](https://scikit-learn.org)

---

## 📋 Deskripsi Project

Project ini mendemonstrasikan penggunaan **DVC (Data Version Control)** untuk mengelola pipeline Machine Learning secara reproducible. Dataset yang digunakan adalah **synthetic classification dataset** dengan karakteristik kompleks:

### 📊 Spesifikasi Dataset
| Karakteristik | Detail |
|--------------|--------|
| **Total Rows** | 10,000 |
| **Total Features** | 10 (+ 1 target) |
| **Numerical Features** | 6 |
| **Categorical Features** | 4 |
| **Target** | Binary Classification (0, 1) |
| **Missing Values** | Yes (1 fitur dengan ~30% missing) |
| **Skewed Features** | 2 fitur dengan distribusi sangat skewed |

### 🔧 Features
| Feature | Type | Missing % | Note |
|---------|------|-----------|------|
| `age` | Numerical | ~5% | Normal distribution |
| `income` | Numerical | ~15% | **Highly skewed** (log-normal) |
| `transaction_amount` | Numerical | ~5% | **Highly skewed** (exponential) |
| `credit_score` | Numerical | ~12% | Normal distribution |
| `account_balance` | Numerical | ~3% | Uniform distribution |
| `years_customer` | Numerical | ~2% | Poisson distribution |
| `education` | Categorical | ~8% | 5 categories |
| `employment_status` | Categorical | **~30%** | **Missing parah** |
| `region` | Categorical | ~2% | 5 categories |
| `account_type` | Categorical | ~1% | 4 categories |

---

## 🏗️ Pipeline Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PREPROCESS    │────▶│     TRAIN       │────▶│    EVALUATE     │
│                 │     │                 │     │                 │
│ • Missing Value │     │ • Model Select  │     │ • Accuracy      │
│ • Skew Transform│     │ • Cross-Val     │     │ • Precision     │
│ • Encoding      │     │ • SMOTE         │     │ • Recall        │
│ • Scaling       │     │ • Training      │     │ • F1/AUC-ROC    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 🚀 Quick Start

### 1️⃣ Clone Repository
```bash
git clone https://github.com/username/dvc-classification-project.git
cd dvc-classification-project
```

### 2️⃣ Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
.\venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 3️⃣ Generate Dataset
```bash
python src/generate_data.py
```

### 4️⃣ Run Pipeline
```bash
dvc repro
```

### 5️⃣ View Results
```bash
# Lihat metrics
cat metrics/metrics.json

# Lihat classification report
cat reports/classification_report.txt
```

---

## 📁 Project Structure

```
dvc-classification-project/
│
├── 📂 data/
│   ├── raw/
│   │   └── dataset.csv          # Raw dataset (DVC tracked)
│   └── processed/
│       ├── dataset_cleaned.csv  # Preprocessed dataset
│       └── test_data.csv        # Test split for evaluation
│
├── 📂 src/
│   ├── generate_data.py         # Dataset generation
│   ├── preprocess.py            # Data preprocessing
│   ├── train.py                 # Model training
│   └── evaluate.py              # Model evaluation
│
├── 📂 models/
│   ├── model.pkl                # Trained model (DVC tracked)
│   └── model_info.json          # Training metadata
│
├── 📂 metrics/
│   └── metrics.json             # Evaluation metrics
│
├── 📂 reports/
│   └── classification_report.txt
│
├── 📂 .github/workflows/
│   └── ci-pipeline.yaml         # GitHub Actions CI/CD
│
├── 📄 dvc.yaml                  # Pipeline definition
├── 📄 dvc.lock                  # Pipeline state
├── 📄 params.yaml               # Hyperparameters
├── 📄 requirements.txt          # Dependencies
├── 📄 .gitignore
└── 📄 README.md
```

---

## ⚙️ DVC Commands Reference

| Command | Description |
|---------|-------------|
| `dvc init` | Inisialisasi DVC di repository |
| `dvc add <file>` | Track file dengan DVC |
| `dvc repro` | Jalankan/reproduksi pipeline |
| `dvc push` | Push data ke remote storage |
| `dvc pull` | Pull data dari remote storage |
| `dvc checkout` | Checkout versi data tertentu |
| `dvc diff` | Lihat perubahan pipeline |
| `dvc metrics show` | Tampilkan metrics |
| `dvc dag` | Visualisasi DAG pipeline |

---

## 🔄 Workflow Versioning

### Update Dataset
```bash
# Generate dataset baru
python src/generate_data.py --rows 15000

# Track perubahan
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc
git commit -m "Update dataset to 15k rows"

# Push ke remote
dvc push
git push
```

### Rollback ke Versi Sebelumnya
```bash
# Checkout versi sebelumnya
git checkout HEAD~1 data/raw/dataset.csv.dvc
dvc checkout

# Jalankan ulang pipeline
dvc repro
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~0.85 |
| Precision | ~0.84 |
| Recall | ~0.85 |
| F1 Score | ~0.84 |
| AUC-ROC | ~0.92 |

*Note: Hasil dapat bervariasi tergantung random seed dan parameter*

---

## 🔐 Remote Storage Setup

### Google Drive
```bash
dvc remote add -d gdrive gdrive://<FOLDER_ID>
dvc remote modify gdrive gdrive_acknowledge_abuse true
```

### AWS S3
```bash
dvc remote add -d s3remote s3://mybucket/dvc-storage
```

### Local
```bash
dvc remote add -d local /path/to/storage
```

---

## 🧪 CI/CD Pipeline

GitHub Actions workflow includes:
1. ✅ **Lint & Test** - Code quality checks
2. ✅ **DVC Pipeline** - Reproduce ML pipeline
3. ✅ **Artifact Upload** - Save model & metrics

---

## 📝 License

MIT License - Feel free to use for learning purposes!

---

## 👤 Author

**DVC Final Project**  
Machine Learning Operations Course