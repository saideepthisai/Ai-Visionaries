import streamlit as st
import google.generativeai as genai
import os
import json
import hashlib
import catboost as cb
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

# ---------- Page config ----------
st.set_page_config(page_title="HealthAI", page_icon="🧬")

# ---------- User Management ----------
def load_users():
    return json.load(open("users.json", "r")) if os.path.exists("users.json") else {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_user(email, password):
    users = load_users()
    hashed = hash_password(password)
    return users.get(email, {}).get("password") == hashed

# ---------- Load Models ----------
@st.cache_resource
def load_sepsis_model():
    model = cb.CatBoostClassifier()
    model.load_model("catboost_sepsis_model.cbm")
    return model

@st.cache_resource
def load_drug_model():
    model = joblib.load("knn_diagnosis_model.pkl")
    label_encoder = joblib.load("label_encoder_diagnosis.pkl")
    df = pd.read_csv("cleaned_dataset.csv")
    return model, label_encoder, df

@st.cache_resource
def initialize_chatbot():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Gemini API key not found.")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.start_chat(history=[])
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        return None

# ---------- Feature Pages ----------
def sepsis_prediction_page(model):
    st.subheader("🔬 Sepsis Prediction")
    features = [
        "Heart Rate", "Respiratory Rate", "Temperature",
        "Diastolic Blood Pressure", "Systolic Blood Pressure",
        "Mean Arterial Pressure", "Oxygen Saturation", "Age", "Gender"
    ]
    values = [st.number_input(f"{feat}", 0.0, format="%.2f") for feat in features]
    if st.button("Predict Sepsis"):
        input_array = np.array(values).reshape(1, -1)
        prob = model.predict_proba(input_array)[0][1]
        result = "⚠ Sepsis Detected" if prob > 0.5 else "✅ No Sepsis"
        st.markdown(f"Result: {result}  \n*Probability:* {prob:.2f}")

def drug_prediction_page(model, label_encoder, df):
    st.subheader("💊 Drug Prediction")

    st.markdown(
        "<h4 style='text-align: center; color: white;'>⚕ SympTrack: Precision Diagnosis & Drug Forecasting</h4>",
        unsafe_allow_html=True
    )

    symptoms = st.text_area("Symptoms", placeholder="e.g., fever, cough, sore throat")

    if st.button("Go!"):
        if symptoms.strip() == "":
            st.warning("⚠ Please enter symptoms.")
        else:
            sample = pd.DataFrame([{
                "symptoms": symptoms,
                "route": "unknown",
                "outcome": "unknown",
                "drug_prescribed": "dummy"
            }])

            try:
                pred_idx = model.predict(sample)
                predicted_diagnosis = label_encoder.inverse_transform(pred_idx)[0]

                drug_mode = df[df["diagnosis"] == predicted_diagnosis]["drug_prescribed"].mode()
                recommended_drug = drug_mode.iloc[0] if not drug_mode.empty else "Unknown"

                st.success(f"✅ Predicted Diagnosis: {predicted_diagnosis}")
                st.info(f"💊 Recommended Drug: {recommended_drug}")
            except Exception as e:
                st.error(f"⚠ An error occurred during prediction: {e}")

def chatbot_page():
    st.subheader("🧠 MediBot")

    st.markdown("""
        <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-top: 20px;
            font-family: sans-serif;
            color: white;
            max-height: 400px;
            overflow-y: auto;
            padding-right: 10px;
        }
        .user-message {
            align-self: flex-end;
            background-color: #007bff;
            color: white;
            padding: 10px 15px;
            border-radius: 16px 16px 0 16px;
            max-width: 75%;
            word-wrap: break-word;
        }
        .bot-message {
            align-self: flex-start;
            background-color: #444;
            color: white;
            padding: 10px 15px;
            border-radius: 16px 16px 16px 0;
            max-width: 75%;
            word-wrap: break-word;
        }
        </style>
    """, unsafe_allow_html=True)

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = initialize_chatbot()
        st.session_state.chat_history = []
        st.session_state.temp_input = ""

    chat = st.session_state.chat_session
    history = st.session_state.chat_history

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in history:
        role_class = "user-message" if msg["role"] == "user" else "bot-message"
        st.markdown(f'<div class="{role_class}">{msg["text"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    def send_message():
        user_input = st.session_state.temp_input.strip()
        if user_input:
            st.session_state.chat_history.append({"role": "user", "text": user_input})
            try:
                response = chat.send_message(user_input)
                reply = response.text if hasattr(response, 'text') else response.candidates[0]['content']
                st.session_state.chat_history.append({"role": "bot", "text": reply})
            except Exception as e:
                st.session_state.chat_history.append({"role": "bot", "text": f"Error: {e}"})
            st.session_state.temp_input = ""

    st.text_input("Type your message here", key="temp_input", on_change=send_message)

# ---------- Main App ----------
def main():
    st.title("🧬 HealthAI Suite")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        st.sidebar.success(f"Logged in as: {st.session_state.email}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.email = ""
            st.session_state.chat_session = None
            st.session_state.chat_history = []
            st.session_state.show_reset = False

        choice = st.radio("Choose a Service", [
            "Sepsis Prediction",
            "Drug Prediction",
            "AI Chatbot Assistant"
        ])

        if choice == "Sepsis Prediction":
            model = load_sepsis_model()
            sepsis_prediction_page(model)
        elif choice == "Drug Prediction":
            drug_model, label_encoder, df = load_drug_model()
            drug_prediction_page(drug_model, label_encoder, df)
        elif choice == "AI Chatbot Assistant":
            chatbot_page()

    else:
        menu = st.radio("Choose Action", ["Login", "Register"])
        if menu == "Login":
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if login_user(email, password):
                    st.success("Login successful!")
                    st.session_state.logged_in = True
                    st.session_state.email = email
                    st.session_state.show_reset = False
                else:
                    st.error("Invalid credentials.")

            if st.button("Forgot Password?"):
                st.session_state.show_reset = True

            if st.session_state.get("show_reset", False):
                st.subheader("Reset Password")
                reset_email = st.text_input("Registered Email")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")

                if st.button("Reset Password"):
                    users = load_users()
                    if reset_email not in users:
                        st.error("Email not found.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        users[reset_email]["password"] = hash_password(new_password)
                        save_users(users)
                        st.success("Password reset successful. Please login.")
                        st.session_state.show_reset = False

        elif menu == "Register":
            fname = st.text_input("First Name")
            lname = st.text_input("Last Name")
            email = st.text_input("Email")
            phone = st.text_input("Phone")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            if st.button("Register"):
                users = load_users()
                if email in users:
                    st.error("Email already registered.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                else:
                    users[email] = {
                        "password": hash_password(password),
                        "first_name": fname,
                        "last_name": lname,
                        "phone": phone,
                        "gender": gender,
                    }
                    save_users(users)
                    st.success("Registered successfully! Please login.")
                    st.session_state.show_reset = False

if __name__ == "__main__":
    main()