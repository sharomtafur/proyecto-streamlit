import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predicción de Churn", layout="centered")
st.title("Predicción de Abandono de Clientes (Churn)")
st.write("Ingresa los datos del cliente para calcular la probabilidad de que cierre su cuenta.")

try:
    modelo = joblib.load("modelo_rfchurn_tunning.pkl")
except FileNotFoundError:
    st.warning("⚠️ No se encontró el archivo del modelo (.pkl). Asegúrate de guardarlo en esta carpeta.")
    modelo = None

def reset_inputs():
    st.session_state["credit_score"] = 600
    st.session_state["age"] = 35
    st.session_state["tenure"] = 5
    st.session_state["balance"] = 50000.0
    st.session_state["num_products"] = 1
    st.session_state["has_cr_card"] = 1
    st.session_state["is_active"] = 1
    st.session_state["salary"] = 60000.0

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Puntaje Crediticio (CreditScore)", min_value=300, max_value=850, value=600, key="credit_score")
    age = st.number_input("Edad (Age)", min_value=18, max_value=100, value=35, key="age")
    tenure = st.number_input("Antigüedad en años (Tenure)", min_value=0, max_value=10, value=5, key="tenure")
    balance = st.number_input("Saldo en cuenta (Balance)", min_value=0.0, value=50000.0, key="balance")

with col2:
    num_products = st.selectbox("Número de productos", [1, 2, 3, 4], key="num_products")
    has_cr_card = st.selectbox("¿Tiene tarjeta de crédito? (1=Sí, 0=No)", [1, 0], key="has_cr_card")
    is_active = st.selectbox("¿Es miembro activo? (1=Sí, 0=No)", [1, 0], key="is_active")
    salary = st.number_input("Salario estimado (EstimatedSalary)", min_value=0.0, value=60000.0, key="salary")

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    predecir = st.button("Predecir", type="primary")

with col_btn2:
    st.button("Resetear", on_click=reset_inputs)

if predecir:
    if modelo is not None:
        datos_cliente = pd.DataFrame({
            "CreditScore": [credit_score],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active],
            "EstimatedSalary": [salary]
        })

        prediction = modelo.predict(datos_cliente)[0]
        probabilities_classes = modelo.predict_proba(datos_cliente)[0]

        if prediction == 0:
            outcome = "No Churn (Cliente se mantiene)"
            probability_churn = probabilities_classes[0]
            style_result = "background-color: lightgreen; font-size: larger; padding: 15px; border-radius: 5px; color: black;"
        else:
            outcome = "Churn (Cliente cerró cuenta)"
            probability_churn = probabilities_classes[1]
            style_result = "background-color: lightcoral; font-size: larger; padding: 15px; border-radius: 5px; color: black;"

        result_html = f"<div style='{style_result}'>La predicción fue de clase <b>'{outcome}'</b> con una probabilidad de <b>{round(probability_churn * 100, 2)}%</b></div>"
        st.markdown(result_html, unsafe_allow_html=True)
    else:
        st.error("No se puede predecir porque falta cargar el archivo del modelo.")