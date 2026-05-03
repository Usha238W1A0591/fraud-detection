# 🛡️ Banking Fraud Detection System

> An end-to-end machine learning system that detects fraudulent financial transactions in real time — built with Python, XGBoost, Scikit-learn, and a live Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.976-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| 🏆 ROC-AUC Score | **0.976** |
| 🎯 Fraud Recall | **87%** |
| ✅ Alert Precision | **87.2%** |
| 📦 Total Transactions Analysed | **56,962** |
| 🚨 High Risk Alerts Fired | **78** |
| 🔍 Actual Fraud Caught | **68 out of 98** |
| ❌ Missed Fraud | **Only 13** |

---

## 🏗️ System Architecture

```
Raw Transactions (284,807 rows)
         ↓
 Data Exploration and EDA
         ↓
 Preprocessing  (Scaling + SMOTE)
    Normal: 227,451  →  227,451
    Fraud:      394  →  227,451   balanced!
         ↓
 ML Model Training
    ├── Random Forest  (AUC: 0.969)
    └── XGBoost        (AUC: 0.976)  winner
         ↓
 Risk Score Engine (0 to 100 score)
    LOW RISK     →  Allow transaction
    MEDIUM RISK  →  Flag for review
    HIGH RISK    →  Block and Alert
         ↓
 Live Streamlit Dashboard
```

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| ML Models | XGBoost, Random Forest |
| Data Processing | Pandas, NumPy |
| Model Evaluation | Scikit-learn |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Model Saving | Joblib |

---

## 📁 Project Structure

```
fraud-detection/
│
├── notebooks/
│   ├── 01_explore.ipynb          # Data loading, EDA, visualisations
│   ├── 02_preprocessing.ipynb    # Scaling, train/test split, SMOTE
│   ├── 03_model.ipynb            # Train XGBoost + Random Forest, evaluate
│   └── 04_risk_scoring.ipynb     # Risk engine, alert generator, save results
│
├── dashboard/
│   ├── app.py                    # Streamlit live dashboard
│   ├── results.csv               # All 56,962 scored transactions
│   └── alerts.csv                # 78 high-risk flagged transactions
│
├── data/
│   └── creditcard.csv            # Download from Kaggle (link below)
│
├── models/
│   ├── best_model.pkl            # Saved XGBoost model
│   ├── scaler.pkl                # Saved scaler
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   └── y_test.pkl
│
├── .gitignore
└── README.md
```

---

## 🚀 How to Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/Usha238W1A0591/fraud-detection.git
cd fraud-detection
```

### 2. Install all dependencies

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn streamlit plotly joblib
```

### 3. Download the dataset

Get the dataset from Kaggle:
👉 [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place `creditcard.csv` inside the `data/` folder.

### 4. Run the notebooks in order

Open VS Code or Jupyter and run each notebook one by one:

```
01_explore.ipynb        →  Understand and visualise the data
02_preprocessing.ipynb  →  Scale, split, and balance with SMOTE
03_model.ipynb          →  Train XGBoost and Random Forest models
04_risk_scoring.ipynb   →  Generate risk scores and fraud alerts
```

### 5. Launch the live dashboard

```bash
cd dashboard
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📈 Key Findings

### V14 is the strongest fraud indicator

XGBoost feature importance shows V14 dominates with a score of **0.536** — more than 10x the next feature (V4 at 0.049).

| Feature | Importance Score |
|---|---|
| V14 | 0.536 |
| V4 | 0.049 |
| V12 | 0.039 |
| V8 | 0.033 |
| V13 | 0.022 |

### Class imbalance was the biggest challenge

The raw dataset had only **0.173% fraud** — just 492 fraud cases out of 284,807 transactions. Without fixing this, the model would predict "normal" for everything and still appear 99.8% accurate — but be completely useless at catching fraud.

**SMOTE fixed this** by generating realistic synthetic fraud samples:

| | Normal | Fraud |
|---|---|---|
| Before SMOTE | 227,451 | 394 |
| After SMOTE | 227,451 | 227,451 |

### XGBoost beat Random Forest on every metric

| Model | ROC-AUC | Fraud Recall | Training Time |
|---|---|---|---|
| Random Forest | 0.969 | 82% | 88 seconds |
| **XGBoost** | **0.976** | **87%** | **7 seconds** |

XGBoost was more accurate AND 12x faster to train.

---

## 🖥️ Dashboard Features

The live Streamlit dashboard includes:

- **4 KPI cards** — Total transactions, high risk flagged, actual fraud found, alert precision
- **Transaction Risk Distribution** — Bar chart of LOW / MEDIUM / HIGH risk counts
- **Fraud Probability Distribution** — Histogram of model confidence scores
- **Risk Score: Fraud vs Normal** — Overlapping histograms showing model separation
- **Top 10 Fraud Indicators** — Horizontal bar chart of XGBoost feature importances
- **Live Alerts Table** — Filterable by risk score slider, shows all flagged transactions
- **Transaction Explorer** — Look up any individual transaction by index number

---

## 📌 Dataset

**[Credit Card Fraud Detection — Kaggle MLG-ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**

- 284,807 credit card transactions from September 2013 (European cardholders)
- 492 fraud cases — only 0.173% of all transactions
- Features V1 to V28 are PCA-transformed to protect cardholder privacy
- `Amount` and `Time` are the only raw unmodified features
- `Class` is the target column: 0 = Normal, 1 = Fraud

---

## 💡 What Makes This Project Stand Out

This project goes beyond a basic ML notebook by including:

1. **Custom Risk Score Engine** — Combines model fraud probability with transaction signals (V14 magnitude, amount size, transaction time) into a 0–100 composite risk score
2. **Dual Model Comparison** — XGBoost and Random Forest both trained, evaluated, and compared with full classification metrics and confusion matrices
3. **SMOTE Oversampling** — Properly handles extreme class imbalance rather than ignoring it
4. **Structured Alert Generator** — Outputs formatted alerts with transaction ID, timestamp, fraud probability, risk score, and recommended action (BLOCK / FLAG)
5. **Interactive Streamlit Dashboard** — Live visual app with Plotly charts, filterable alerts table, and transaction-level explorer

---

## 📝 Resume Description

```
Fraud Detection System | Python, XGBoost, Streamlit, Scikit-learn

- Built an end-to-end ML fraud detection system trained on 284,807 real
  credit card transactions (Kaggle MLG-ULB dataset)
- Achieved ROC-AUC of 0.976 using XGBoost, outperforming Random Forest (0.969)
- Handled severe class imbalance (0.17% fraud) using SMOTE oversampling,
  balancing training data to 454,902 samples
- Built a custom risk scoring engine (0-100) combining model probability with
  transaction signals to classify transactions as LOW / MEDIUM / HIGH risk
- Flagged 78 high-risk transactions with 87.2% precision and 87% fraud recall
- Developed a live Streamlit dashboard with Plotly charts, real-time alert
  table, and transaction explorer across 56,962 test transactions
```

---

## 👩‍💻 Author

**Usha Kiranmai Kowluri** — Built as a complete machine learning portfolio project, covering the full pipeline from raw data exploration to a live interactive dashboard.

- GitHub: [Usha238W1A0591](https://github.com/Usha238W1A0591)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
