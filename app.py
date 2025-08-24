
import streamlit as st
import pickle
import numpy as np

# Load mô hình đã huấn luyện
with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Wine Classifier", page_icon="🍷", layout="wide")

st.title("🍷 Wine Classification App")
st.markdown("Nhập các **chỉ số hóa học** của rượu để dự đoán loại rượu.")

# Layout 3 cột nhập dữ liệu
col1, col2, col3 = st.columns(3)

with col1:
    alcohol = st.number_input("Alcohol", 0.0, 20.0, 13.0)
    malic_acid = st.number_input("Malic Acid", 0.0, 10.0, 2.0)
    ash = st.number_input("Ash", 0.0, 5.0, 2.0)
    alcalinity = st.number_input("Alcalinity of ash", 0.0, 30.0, 15.0)
    magnesium = st.number_input("Magnesium", 0.0, 200.0, 100.0)

with col2:
    total_phenols = st.number_input("Total phenols", 0.0, 5.0, 2.0)
    flavanoids = st.number_input("Flavanoids", 0.0, 5.0, 2.0)
    nonflavanoid_phenols = st.number_input("Nonflavanoid phenols", 0.0, 1.0, 0.3)
    proanthocyanins = st.number_input("Proanthocyanins", 0.0, 5.0, 1.0)
    color_intensity = st.number_input("Color intensity", 0.0, 15.0, 5.0)

with col3:
    hue = st.number_input("Hue", 0.0, 2.0, 1.0)
    od280 = st.number_input("OD280/OD315", 0.0, 5.0, 3.0)
    proline = st.number_input("Proline", 0.0, 2000.0, 1000.0)

# Nút dự đoán
st.markdown("---")
center = st.columns([3,1,3])[1]

with center:
    if st.button("🔍 Dự đoán", use_container_width=True):
        features = np.array([[alcohol, malic_acid, ash, alcalinity, magnesium,
                              total_phenols, flavanoids, nonflavanoid_phenols,
                              proanthocyanins, color_intensity, hue, od280, proline]])
        prediction = model.predict(features)[0]

        # Map class -> tên + icon + màu
        wine_info = {
            0: {"name": "Rượu vang đỏ", "icon": "🍷", "color": "#FF4B4B"},
            1: {"name": "Rượu vang trắng", "icon": "🥂", "color": "#FFD700"},
            2: {"name": "Rượu vang hồng", "icon": "🍾", "color": "#FF69B4"},
        }

        info = wine_info.get(prediction, {"name": "Không xác định", "icon": "❓", "color": "#CCCCCC"})

        st.markdown(
            f"""
            <div style="padding:20px; border-radius:15px; text-align:center;
                        background-color:{info['color']}; color:white; font-size:22px; font-weight:bold;">
                {info['icon']} Kết quả dự đoán: {info['name']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Hiệu ứng vui mắt
        st.balloons()
