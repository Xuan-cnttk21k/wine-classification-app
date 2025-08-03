import streamlit as st
import pickle
import numpy as np

# Load mô hình đã huấn luyện
with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Wine Classification App 🍷")
st.write("Nhập các chỉ số hóa học của rượu để dự đoán loại rượu.")

# Form nhập dữ liệu
alcohol = st.number_input("Alcohol", 0.0, 20.0, 13.0)
malic_acid = st.number_input("Malic Acid", 0.0, 10.0, 2.0)
ash = st.number_input("Ash", 0.0, 5.0, 2.0)
alcalinity = st.number_input("Alcalinity of ash", 0.0, 30.0, 15.0)
magnesium = st.number_input("Magnesium", 0.0, 200.0, 100.0)
total_phenols = st.number_input("Total phenols", 0.0, 5.0, 2.0)
flavanoids = st.number_input("Flavanoids", 0.0, 5.0, 2.0)
nonflavanoid_phenols = st.number_input("Nonflavanoid phenols", 0.0, 1.0, 0.3)
proanthocyanins = st.number_input("Proanthocyanins", 0.0, 5.0, 1.0)
color_intensity = st.number_input("Color intensity", 0.0, 15.0, 5.0)
hue = st.number_input("Hue", 0.0, 2.0, 1.0)
od280 = st.number_input("OD280/OD315", 0.0, 5.0, 3.0)
proline = st.number_input("Proline", 0.0, 2000.0, 1000.0)

if st.button("Dự đoán"):
    features = np.array([[alcohol, malic_acid, ash, alcalinity, magnesium, total_phenols, flavanoids,
                          nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280, proline]])
    prediction = model.predict(features)[0]
    st.success(f"Loại rượu dự đoán: Class {prediction}")
