
# 📊 Customer Churn Prediction

A machine learning solution to predict **customer churn** based on service usage and demographic data in the telecom industry. This enables businesses to **retain customers** through proactive interventions.

---

## 💼 Business Objective

Churn refers to when a customer stops using a company’s services. Predicting churn early helps telecom companies:

- Reduce revenue loss  
- Improve customer satisfaction  
- Target high-risk customers with personalized offers  

This project builds a machine learning pipeline that classifies whether a customer is likely to churn or not.

---

## 🧾 Dataset Overview

The dataset contains information such as:

- **Demographics** (e.g., gender, senior citizen status)
- **Service details** (e.g., internet, phone, contract)
- **Billing information** (e.g., monthly & total charges)

The target variable is **Churn** (`Yes` or `No`), converted to binary values (`1` for churn, `0` for retention).

---

## 🔍 Features Used

| Category         | Features                                                                 |
|------------------|--------------------------------------------------------------------------|
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents`                      |
| **Services**     | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, etc. |
| **Billing**      | `MonthlyCharges`, `TotalCharges`, `tenure`                              |

---

## 🧠 Models Implemented

Three classification models are trained using a unified preprocessing pipeline:

- 🎯 **Random Forest**
- 🔢 **Logistic Regression**
- ⚡ **XGBoost**

Each model is trained using a **scikit-learn `Pipeline`** to ensure standardized preprocessing and reproducibility.

---

## ⚙️ Project Structure

```
├── churn_features.pkl                 # List of input features used during training
├── customer_churn.csv                # Input dataset
├── logisticregression_churn_model.pkl # Trained Logistic Regression model
├── randomforest_churn_model.pkl      # Trained Random Forest model
├── xgboost_churn_model.pkl           # Trained XGBoost model
├── model_training.py                 # Python script to train and save models
├── app.py                            # Streamlit app (optional for deployment)
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🏗️ Pipeline Steps

### 1. **Data Cleaning**

- Convert `TotalCharges` to numeric
- Impute missing `TotalCharges` values using median

### 2. **Feature Engineering**

- Separate into **numerical** and **categorical** features
- Scale numerical features using `StandardScaler`
- Encode categorical features using `OneHotEncoder`

### 3. **Model Training**

Each model is wrapped in a scikit-learn `Pipeline` to include preprocessing and training:

```
Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])),
    ("classifier", model)
])
```

### 4. **Model Saving**

All models are saved as `.pkl` files using `joblib`.

---

## 🚀 How to Run the Project

### 🔧 Step 1: Install Dependencies

```
pip install -r requirements.txt
```

### 📈 Step 2: Train Models

```
python model_training.py
```

This will generate:
- `randomforest_churn_model.pkl`
- `logisticregression_churn_model.pkl`
- `xgboost_churn_model.pkl`
- `churn_features.pkl`

### 🖥️ Step 3: Run Streamlit App (Optional)

```
streamlit run app.py
```

This will launch a browser-based UI for making churn predictions interactively.

---

## 🛠️ Requirements

Below are the main dependencies:

```text
pandas
scikit-learn
xgboost
joblib
streamlit
```

*(Full list in \`requirements.txt\`)*

---

## 📊 Future Enhancements

- Add SHAP-based model explainability
- Model performance comparison dashboard
- Integration with real-time data sources via FastAPI or Flask
- Add evaluation metrics like ROC AUC, precision, recall

---

## 🙌 Acknowledgements

- [IBM Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- Built using Python, scikit-learn, and XGBoost

---

## 📬 Contact

If you have questions or want to collaborate, feel free to connect with me on
- [LinkedIn](https://www.linkedin.com/in/amit-kharche)  
- [Medium](https://medium.com/@amitkharche14)  
- [GitHub](https://github.com/amitkharche)

