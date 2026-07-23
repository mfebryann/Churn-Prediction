# 📉 Customer Churn Prediction & Retention Analysis

A Machine Learning and Data Analytics project designed to predict customer churn in the telecommunications industry. The system performs Exploratory Data Analysis (EDA) to uncover churn drivers, handles data preprocessing and class imbalance using SMOTE, and builds predictive models to help optimize customer retention strategies.

---

## 🌟 Key Features

* **Exploratory Data Analysis (EDA):** Deep dive into demographic, service, and account features to identify key churn drivers.
* **Statistical Hypothesis Testing:** Conducts Chi-Square tests for categorical variables and T-Tests for continuous variables to validate feature significance.
* **Missing Value Imputation:** Smart handling of missing values on new customers (`tenure = 0`) using their monthly charge baseline.
* **Advanced Preprocessing:** Features tailored encoding (One-Hot & Ordinal) and `StandardScaler` feature scaling.
* **Class Imbalance Resolution:** Balances target classes using **SMOTE** (Synthetic Minority Over-sampling Technique).
* **Predictive Modeling:** Machine Learning models trained to classify at-risk customers with target accuracy exceeding **80%**.

---

## 🏗️ Project Pipeline

1. **Data Understanding & Cleaning:**
   * Converts `TotalCharges` to numeric (`float64`) and `SeniorCitizen` to categorical type.
   * Imputes 11 missing values in `TotalCharges` based on `MonthlyCharges` for customers with 0-month tenure.
   * Verifies no duplicated records and absence of severe numerical outliers.
2. **Feature Selection & Engineering:**
   * Drops non-informative attributes (`customerID`, `gender`, `PhoneService`).
   * Encodes nominal categorical variables using **One-Hot Encoding**.
   * Encodes contract duration using **Ordinal Encoding** (`Month-to-month`: 1, `One year`: 12, `Two year`: 24).
3. **Resampling & Scaling:**
   * Resamples dataset using **SMOTE** to achieve balanced target distribution (5,174 samples per class).
   * Normalizes continuous numerical attributes using `StandardScaler`.
4. **Model Training & Evaluation:**
   * Evaluates machine learning algorithms to achieve robust predictive performance (> 80% accuracy).

---

## 📂 Outputs & Key Insights

The pipeline produces key statistical insights and data artifacts:

* **Demographics:**
  * Customers without partners or dependents show a statistically significant higher propensity to churn.
  * Gender shows no statistically significant impact on churn rates (`p-value = 0.971`).
* **Service Types:**
  * Fiber Optic subscribers (43.96% of users) experienced higher churn rates compared to DSL users, primarily driven by higher `MonthlyCharges`.
* **Account Contracts:**
  * Month-to-month contracts dominate the subscriber base (56.02%) and show high churn rates, particularly during early tenure (average churn tenure is ~14 months).
* **Statistical Hypothesis Testing Summary:**

| Feature | Test Type | Metric / P-Value | Conclusion |
| :--- | :--- | :--- | :--- |
| **Partner / Dependents** | Chi-Square | p < 0.05 | Statistically Significant |
| **Gender** | Chi-Square | p = 0.971 | Not Significant |
| **Monthly / Total Charges** | T-Test | p < 0.05 | Statistically Significant |

---

## 📊 Preprocessing & Model Highlights

### Code Highlight: Data Pipeline & Resampling

`from sklearn.preprocessing import StandardScaler`
`from imblearn.over_sampling import SMOTE`

`# Ordinal Encoding for Contract Type`
`contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}`
`df['Contract'] = df['Contract'].map(contract_mapping)`

`# Balancing Target Class with SMOTE`
`smote = SMOTE(random_state=42)`
`X_resampled, y_resampled = smote.fit_resample(X, y)`

`# Feature Scaling`
`scaler = StandardScaler()`
`X_scaled = scaler.fit_transform(X_resampled)`

### Performance Target

Target model accuracy performance across testing sets:

* **Target Accuracy:** > 80%
* **Target Metric:** Balanced Precision & Recall for Churn class

---

## 🛠️ Built With

* [Python 3.x](https://www.python.org/) - Programming Language
* [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) - Data Manipulation & Preprocessing
* [Scikit-learn](https://scikit-learn.org/) - Feature Scaling, Encoding & Modeling
* [Imbalanced-learn](https://imbalanced-learn.org/) - SMOTE Resampling Technique
* [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - Data Visualization

---

## 📌 Business Recommendations

* **Incentivize Long-Term Contracts:** Offer targeted discounts or promotions to transition high-risk Month-to-month customers into 1-year or 2-year contracts.
* **Fiber Optic Value Review:** Re-evaluate pricing models and bundle services for Fiber Optic plans to reduce price sensitivity.
* **Early Tenure Intervention:** Focus retention and onboarding campaigns specifically on customers during their first 14 months of service.
