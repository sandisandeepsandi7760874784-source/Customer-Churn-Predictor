# 📊 Customer Churn Prediction & Retention Intelligence Platform

An intermediate-to-advanced, production-grade Data Science and Machine Learning project designed to predict customer churn, identify revenue at risk, and prescribe actionable retention strategies.

Built with **Python**, **Pandas**, **NumPy**, **Scikit-Learn**, **Matplotlib**, **Seaborn**, and **Streamlit**.

---

## 📌 Project Overview

Customer churn (when customers cancel or stop doing business with a company) is one of the highest expenses in subscription-based industries like telecommunications, SaaS, and financial services. Acquiring a new customer can cost up to **5x more** than retaining an existing one.

This project delivers:
1. **Automated Data Ingestion & Preprocessing**: Cleans raw customer data, handles missing/whitespace anomalies, and engineers a reusable transformation pipeline.
2. **Exploratory Data Analysis (EDA)**: Statistical and visual analysis uncovering key drivers of customer attrition.
3. **Machine Learning Model Comparison**: Trains and evaluates **Logistic Regression**, **Random Forest**, and **Gradient Boosting** with genuine, un-faked Scikit-learn metrics.
4. **Interactive Streamlit Web Dashboard**: Real-time customer churn risk scoring, visual risk gauges, batch risk explorer with CSV export, and prescriptive retention recommendations.

---

## 📁 Project Structure

```text
Customer Churn Prediction/
│
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.csv      # Raw downloaded dataset (7,043 rows, 21 columns)
│   └── processed/
│       └── cleaned_churn_data.csv        # Cleaned dataset ready for modeling
│
├── models/
│   ├── best_model.joblib                 # Serialized Scikit-learn Pipeline (Preprocessor + Champion Model)
│   └── model_metadata.json               # Serialized model hyperparameters, metrics & features
│
├── reports/
│   └── figures/                          # Exported high-resolution visualization plots
│       ├── churn_distribution.png        # Donut & bar breakdown of churned vs retained customers
│       ├── tenure_churn_distribution.png # Tenure curve showing high 1st-year dropoff
│       ├── contract_churn_analysis.png   # Churn rate across contract types
│       ├── monthly_charges_distribution.png # Monthly bill distribution by churn status
│       ├── correlation_heatmap.png       # Correlation matrix of numerical variables
│       ├── confusion_matrix.png          # Annotated confusion matrix heatmap on test set
│       ├── roc_curve.png                 # Receiver Operating Characteristic (ROC) curve
│       └── feature_importance.png        # Top feature importance ranking
│
├── src/
│   ├── __init__.py                       # Package initializer
│   ├── data_loader.py                    # Dataset auto-downloader and loader
│   ├── data_preprocessing.py             # Data cleaning and Scikit-learn ColumnTransformer
│   ├── eda.py                            # Exploratory Data Analysis & visual generation
│   ├── evaluate.py                       # Accuracy, Precision, Recall, F1, ROC-AUC, and plots
│   └── train.py                          # Multi-model training, ranking, and artifact serialization
│
├── app.py                                # Interactive multi-page Streamlit Dashboard
├── requirements.txt                      # Project dependencies
└── README.md                             # Comprehensive project documentation
```

---

## 📊 Dataset Description

The project uses the **Telco Customer Churn dataset** (7,043 customers, 21 attributes), covering:
- **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Subscribed Services**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Account & Billing Details**: `tenure` (months with company), `Contract` (Month-to-month, One year, Two year), `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Target Variable**: `Churn` (`1` = Customer left, `0` = Customer retained)

---

## 🛠️ Data Preprocessing & Pipeline Architecture

A common pitfall in data science projects is **data leakage** (e.g. fitting scalers or encoders on the entire dataset before splitting). This project follows senior engineering best practices:

1. **Whitespace Cleaning**: In the raw dataset, 11 customers with `tenure = 0` had `" "` (blank string) in `TotalCharges`. We convert these to numeric and impute `0.0` (as they have not yet completed a billing cycle).
2. **ColumnTransformer Pipeline**:
   - **Numerical Features** (`tenure`, `MonthlyCharges`, `TotalCharges`): Standardized with `StandardScaler()`.
   - **Categorical Features** (16 columns): Encoded with `OneHotEncoder(drop='first', handle_unknown='ignore')`.
3. **Zero Data Leakage**: The preprocessor and classifier are bundled in an end-to-end Scikit-learn `Pipeline`. The transformer is fitted strictly on the training fold (`X_train`) and applied cleanly to `X_test` and incoming inference requests.

---

## 🏆 Model Training & Genuine Performance Results

The models were evaluated on an unseen **20% stratified test set** (1,409 customers) with class balance maintained.

### Model Comparison Table:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Balanced) | 73.81% | 50.43% | 78.34% | 0.6136 | 0.8417 | Candidate |
| **Random Forest** (Balanced) | 75.59% | 52.72% | 77.81% | 0.6285 | 0.8401 | Candidate |
| **Gradient Boosting** | **80.06%** | **65.45%** | **52.67%** | **0.5837** | **0.8433** | 🏆 **Champion** |

### Champion Model: Gradient Boosting Classifier
- **Overall Accuracy**: **80.06%**
- **ROC-AUC**: **0.8433** (Strong discriminatory power)
- **Confusion Matrix Breakdown (1,409 Test Cases)**:
  - **True Negatives (Retained correctly)**: 931
  - **False Positives (False churn alarm)**: 104
  - **False Negatives (Missed churners)**: 177
  - **True Positives (Churn identified)**: 197

### Key Feature Drivers (Why Customers Churn):
1. **Contract Type (`Month-to-month`)**: The single strongest predictor of churn. Month-to-month customers churn at **42.7%**, compared to **2.8%** for two-year contract holders.
2. **Tenure (`tenure`)**: Customers in their first 12 months have over 50% of all churn events.
3. **Monthly Charges (`MonthlyCharges`)**: Customers paying high fees without value perception (e.g. Fiber optic without security/tech support) churn at significantly higher rates.
4. **Lack of Tech Support & Online Security**: Customers without tech support churn at over **41%**.

---

## 🚀 How to Run the Project

### 1. Clone or Open the Workspace
```bash
cd "d:/Customer Churn Prediction"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Data Pipeline & Model Training
You can run individual modules or the full end-to-end pipeline:

- **Download & Inspect Data**:
  ```bash
  python -m src.data_loader
  ```
- **Clean & Preprocess Data**:
  ```bash
  python -m src.data_preprocessing
  ```
- **Run Exploratory Data Analysis (Generates Figures in `reports/figures/`)**:
  ```bash
  python -m src.eda
  ```
- **Train Models, Evaluate & Save Artifacts**:
  ```bash
  python -m src.train
  ```

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Dashboard Modules

The application includes 5 comprehensive interactive pages:

1. **📊 Executive Overview**:
   - Executive KPIs: Total Customers, Churn Rate %, Avg Monthly Revenue, Monthly Revenue at Risk ($139K+).
   - High-level composition donut chart and customer tenure histogram.
2. **🔍 Churn Insights & EDA**:
   - Interactive breakdown of churn by contract, internet type, tech support, payment method, etc.
   - Interactive scatter plot of Monthly Charges vs Total Lifetime Value.
3. **🎯 Single Customer Churn Predictor**:
   - Interactive form to input customer demographics, services, contract, and billing details.
   - Outputs Churn Probability, Risk Category (Low / Medium / High Risk), and a Visual Risk Meter.
   - **Prescriptive Retention Actions**: Automatically generates personalized retention interventions (e.g., offering annual contract incentives, complementary tech support, or plan reviews).
4. **📂 Batch Risk Explorer**:
   - Ranks the entire customer base by model-predicted churn risk.
   - Filter by risk level, contract type, and minimum monthly bill.
   - **One-click CSV Download** to send target lists directly to retention and customer success teams.
5. **📈 Model Performance**:
   - Side-by-side performance comparison matrix across all candidate models.
   - Renders the authentic Confusion Matrix, ROC-AUC curve, and Feature Importance ranking.

---

## 🛡️ Production & Error Handling Highlights

- **Pandas 3.0 & Python 3.14 Compatibility**: Strict type handling that works with both PyArrow string dtypes and legacy object columns.
- **Resilient Web Inference**: Input validation ensures default fallbacks if individual fields are omitted.
- **Modular Codebase**: Each component (`data_loader`, `data_preprocessing`, `eda`, `train`, `evaluate`) is self-contained with docstrings and clean functions, adhering to clean architecture principles.
