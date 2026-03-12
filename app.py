import streamlit as st
from PIL import Image
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import VGG16
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Q-NeuroStroke | Quantum-AI Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom UI Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

h1, h2, h3 {
    color: #58a6ff;
    font-weight: 700 !important;
}

.css-1d391kg, [data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

.main-ribbon {
    background: linear-gradient(135deg, #1f6feb 0%, #8957e5 100%);
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}

.metric-title {
    font-size: 1rem;
    color: #8b949e;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #3fb950;
}

.result-stroke {
    background: rgba(248, 81, 73, 0.1);
    border: 2px solid #f85149;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
}

.result-normal {
    background: rgba(46, 160, 67, 0.1);
    border: 2px solid #2ea043;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Ribbon Title
# -----------------------------
st.markdown("""
<div class="main-ribbon">
    <h1>Q-NeuroStroke</h1>
    <p>Hybrid Quantum-AI Framework for Brain Stroke Detection</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar Menu
# -----------------------------
with st.sidebar:
    st.title("Navigation")
    st.markdown("---")
    page = st.radio(
        "Modules",
        ["🧠 Stroke Detection", "📊 Model Comparison"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Powered by:** VGG16 + NMF + Quantum Encoding + Gaussian NB")

# -----------------------------
# Load Models (Only Once)
# -----------------------------
@st.cache_resource
def load_models():
    vgg = VGG16(weights="imagenet", include_top=False, pooling="avg")

    model_path = os.path.join(os.path.dirname(__file__), "quantum_gnb_model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    nmf_path = os.path.join(os.path.dirname(__file__), "nmf_model.pkl")

    classifier = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    nmf = joblib.load(nmf_path)

    return vgg, classifier, scaler, nmf

# load once
vgg, classifier, scaler, nmf = load_models()

# ==================================================
# Stroke Detection Page
# ==================================================
if page == "🧠 Stroke Detection":

    st.markdown("### 🔬 Diagnostic Tool")
    st.markdown("Upload a Brain CT Scan image to run inference using the Hybrid Quantum-AI model.")

    col_uploader, col_result = st.columns([1, 1], gap="large")

    with col_uploader:
        uploaded_file = st.file_uploader(
            "Drag & Drop CT Scan",
            type=["jpg","png","jpeg"],
            help="Supported formats: JPG, PNG, JPEG"
        )

    if uploaded_file:

        img = Image.open(uploaded_file).convert("RGB")

        with col_uploader:
            st.image(img, caption="Uploaded CT Scan", use_container_width=True)
            run_btn = st.button("🚀 Analyze Scan", use_container_width=True, type="primary")

        if run_btn:

            with col_result:

                with st.spinner("Running Quantum-AI Pipeline..."):

                    img_resized = img.resize((224,224))

                    img_array = np.asarray(img_resized, dtype=np.float32)
                    img_array /= 255.0
                    img_array = img_array[None, ...]

                    # Faster inference
                    features = vgg(img_array, training=False).numpy()

                    reduced_features = nmf.transform(features)

                    quantum_features = np.concatenate(
                        [np.sin(reduced_features), np.cos(reduced_features)],
                        axis=1
                    )

                    scaled = scaler.transform(quantum_features)

                    prediction = classifier.predict(scaled)
                    probability = classifier.predict_proba(scaled)[0]

                st.markdown("---")

                if prediction[0] == 1:
                    st.markdown("""
                    <div class="result-stroke">
                        <h2>⚠️ STROKE DETECTED</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"**Confidence Score:** `{probability[1]*100:.2f}%`")

                else:
                    st.markdown("""
                    <div class="result-normal">
                        <h2>✅ NORMAL BRAIN</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"**Confidence Score:** `{probability[0]*100:.2f}%`")

                st.toast("Inference completed successfully!", icon="✅")

# ==================================================
# Model Comparison Page (UNCHANGED)
# ==================================================
elif page == "📊 Model Comparison":

    st.markdown("### 📈 Classical vs. Quantum Performance")

    classical_metrics = {"Accuracy": 76.47, "Precision": 0.79, "Recall": 0.84, "F1 Score": 0.81}
    quantum_metrics = {"Accuracy": 80.39, "Precision": 0.86, "Recall": 0.81, "F1 Score": 0.83}

    col_cards1, col_cards2, col_cards3, col_cards4 = st.columns(4)

    with col_cards1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Quantum Accuracy</div>
            <div class="metric-value">{quantum_metrics["Accuracy"]:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_cards2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Quantum Precision</div>
            <div class="metric-value">{quantum_metrics["Precision"]:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_cards3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Quantum Recall</div>
            <div class="metric-value">{quantum_metrics["Recall"]:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_cards4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Quantum F1 Score</div>
            <div class="metric-value">{quantum_metrics["F1 Score"]:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"] * 2,
        "Value": [
            classical_metrics["Accuracy"]/100,
            classical_metrics["Precision"],
            classical_metrics["Recall"],
            classical_metrics["F1 Score"],
            quantum_metrics["Accuracy"]/100,
            quantum_metrics["Precision"],
            quantum_metrics["Recall"],
            quantum_metrics["F1 Score"]
        ],
        "Model": ["Classical"] * 4 + ["Quantum"] * 4
    })

    fig = px.bar(df, x="Metric", y="Value", color="Model", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Confusion Matrix")

    cm = [[15,5],[1,30]]

    fig_cm, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=["Normal","Stroke"],
        yticklabels=["Normal","Stroke"]
    )

    st.pyplot(fig_cm)