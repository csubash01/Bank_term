import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("bank_best_model.joblib")

st.set_page_config(page_title="Bank Term Deposit Predictor", page_icon="💳")
st.title("💳 Bank Term Deposit Subscription Predictor")

# Dropdown options
jobs = ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur",
        "student", "blue-collar", "self-employed", "retired", "technician", "services"]
maritals = ["married", "divorced", "single"]
educations = ["unknown", "secondary", "primary", "tertiary"]
binary = ["no", "yes"]
contacts = ["unknown", "telephone", "cellular"]
months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
poutcomes = ["unknown","other","failure","success"]

with st.form("form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    job = st.selectbox("Job", jobs)
    marital = st.selectbox("Marital status", maritals)
    education = st.selectbox("Education", educations)
    default = st.selectbox("Has default?", binary)
    balance = st.number_input("Average yearly balance (EUR)", value=1000)
    housing = st.selectbox("Housing loan?", binary)
    loan = st.selectbox("Personal loan?", binary)
    contact = st.selectbox("Contact communication type", contacts)
    day = st.number_input("Last contact day", min_value=1, max_value=31, value=15)
    month = st.selectbox("Month", months)
    campaign = st.number_input("Number of contacts in this campaign", min_value=1, value=1)
    pdays = st.number_input("Days since last contact (-1 = never)", min_value=-1, value=999)
    previous = st.number_input("Previous contacts", min_value=0, value=0)
    poutcome = st.selectbox("Previous outcome", poutcomes)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build dataframe for prediction
    df = pd.DataFrame([{
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "balance": balance, "housing": housing, "loan": loan,
        "contact": contact, "day": day, "month": month, "campaign": campaign,
        "pdays": pdays, "previous": previous, "poutcome": poutcome
    }])

    proba = model.predict_proba(df)[:,1][0]
    pred = model.predict(df)[0]

    label_map = {1:"YES", 0:"NO"}
    st.subheader("Prediction Result")
    st.write(f"Will the client subscribe? **{label_map[pred]}**")
    st.write(f"Probability of YES: **{proba:.3f}**")
