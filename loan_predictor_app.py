
import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def save_to_mysql(user_data):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="your_username",
            password="your_password",
            database="loan_db"
        )
        cursor = conn.cursor()
        query = """
            INSERT INTO loan_history (income, credit_score, employment_type, prediction)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, tuple(user_data.values()))
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        st.error(f"Database error: {err}")

@st.cache_data
def load_dataset():
    return pd.read_csv("loan_eligibility_dataset.csv")

@st.cache_resource
def train_models(df):
    df['Employment_Type'] = df['Employment_Type'].map({'Salaried': 0, 'Self-employed': 1})
    X = df[['Income', 'Credit_Score', 'Employment_Type']]
    y = df['Loan_Status']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    return log_model, nb_model, scaler, X.columns

st.set_page_config(page_title="Loan Eligibility Predictor")
st.title("🏦 Loan Eligibility Predictor")
st.markdown("This app predicts the likelihood of loan approval based on income, credit score, and employment type.")

df = load_dataset()
log_model, nb_model, scaler, feature_names = train_models(df)

income = st.number_input("Enter your Annual Income ($)", min_value=10000, step=1000)
credit_score = st.number_input("Enter your Credit Score", min_value=300, max_value=850)
employment_type = st.selectbox("Employment Type", ["Salaried", "Self-employed"])
model_choice = st.radio("Choose a Prediction Model", ["Logistic Regression", "Naive Bayes"])

if st.button("Predict Loan Eligibility"):
    emp_val = 0 if employment_type == "Salaried" else 1
    input_data = pd.DataFrame([[income, credit_score, emp_val]], columns=["Income", "Credit_Score", "Employment_Type"])
    input_scaled = scaler.transform(input_data)

    if model_choice == "Logistic Regression":
        prediction = log_model.predict(input_scaled)[0]
        probs = log_model.predict_proba(input_scaled)[0]
        importances = np.abs(log_model.coef_[0])
    else:
        prediction = nb_model.predict(input_scaled)[0]
        probs = nb_model.predict_proba(input_scaled)[0]
        importances = np.abs(nb_model.theta_[1] - nb_model.theta_[0])

    result = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.subheader(f"Loan Status: {result}")
    st.write(f"Approval Probability: {probs[1]*100:.2f}%")

    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=feature_names, ax=ax)
    ax.set_title("🔍 Feature Importance")
    st.pyplot(fig)

    user_data = {
        "income": income,
        "credit_score": credit_score,
        "employment_type": employment_type,
        "prediction": int(prediction)
    }
    save_to_mysql(user_data)
    st.success("Prediction saved to database.")
