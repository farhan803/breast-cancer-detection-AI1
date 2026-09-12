# ==========================================
# Breast Cancer Detection AI
# app.py (Part-1)
# ==========================================
import sys
sys.path = [p for p in sys.path if 'AppData\\Roaming' not in p and 'AppData/Roaming' not in p]
import streamlit as st
st.set_page_config(page_title="Breast Cancer AI", layout="wide", initial_sidebar_state="expanded")
from pathlib import Path
from utils.data_loader import load_dataset, get_dataset_summary
# ==========================================
# CACHE
# ==========================================

@st.cache_data(show_spinner=False)
def get_data():
    return load_dataset()


@st.cache_data(show_spinner=False)
def get_summary():
    return get_dataset_summary()


# ==========================================
# LOAD DATA
# ==========================================

try:
    with st.spinner("Loading dataset..."):
        df = get_data()
        summary = get_summary()

except Exception as e:
    st.error("❌ Failed to load dataset.")
    st.exception(e)
    st.stop()


# ==========================================
# LOAD CSS
# ==========================================

BASE_DIR = Path(__file__).parent

css_file = BASE_DIR / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ==========================================
# SIDEBAR LOGO
# ==========================================

logo_path = BASE_DIR / "assets" / "logo.png"

with st.sidebar:
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image("assets/logo.png", width=110)

    st.markdown(
        """<h3 style='text-align:center;color:#00E676;'>Breast Cancer AI</h3>
          <p style="text-align:center; color:#A0AEC0;">
            AI Powered Diagnosis
        </p>""",
        unsafe_allow_html=True
    )
    
    

    st.divider()
# ==========================================

st.markdown("""
<style>
/* ---- Typing animation for subtitle ---- */
.typing-subtitle {
    font-size: 2.1rem;
    font-weight: 700;
    font-family: 'Plus Jakarta Sans', 'Georgia', serif;
    font-style: italic;
    color: rgba(148, 163, 184, 0.85);
    letter-spacing: 0.04em;
    overflow: hidden;
    white-space: nowrap;
    border-right: 2.5px solid #00e676;
    width: 0;
    animation:
        typing 2.8s steps(44, end) 0.6s forwards,
        blink-caret 0.75s step-end infinite;
    margin-top: 2px;
    margin-bottom: 28px;
    padding-bottom: 2px;
}

@keyframes typing {
    from { width: 0; }
    to   { width: 44ch; }
}

@keyframes blink-caret {
    0%, 100% { border-color: #00e676; }
    50%       { border-color: transparent; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🩺 Breast Cancer Detection AI")
st.markdown(
    '<p class="typing-subtitle">Artificial Intelligence Powered Medical Diagnosis</p>',
    unsafe_allow_html=True)
st.write(
    """
Welcome to the **Breast Cancer Detection System**.

This application uses Machine Learning algorithms
to predict whether a tumour is:

- 🟢 Benign
- 🔴 Malignant

Navigate using the sidebar to explore the
prediction system, dashboard and model comparison.
"""
)
st.divider()


# ==========================================
# QUICK STATS
# ==========================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "📄 Samples",
    summary.get("samples", 0)
)
c2.metric(
    "📊 Features",
    summary.get("features", 0)
)
c3.metric(
    "🟢 Benign",
    summary.get("benign", 0)
)
c4.metric(
    "🔴 Malignant",
    summary.get("malignant", 0)
)
# ==========================================
# FEATURE CARDS
# ==========================================

st.header("🚀 System Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 AI Prediction</h3>
        <p>
        Predict whether a tumour is
        <b>Benign</b> or <b>Malignant</b>
        using trained Machine Learning models.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Interactive Dashboard</h3>
        <p>
        Explore the dataset using
        interactive charts,
        statistics and visual analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>📈 Model Comparison</h3>
        <p>
        Compare the performance of
        Logistic Regression,
        Random Forest,
        SVM,
        KNN,
        Decision Tree
        and Naive Bayes.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==========================================
# DATASET PREVIEW
# ==========================================

st.header("📂 Dataset Preview")

prev_left, prev_right = st.columns([2.5, 1], gap="large")

with prev_left:
    st.dataframe(
        df.head(10),
        width='stretch',
        height=385)
with prev_right:
    rows_val    = df.shape[0]
    cols_val    = df.shape[1]
    missing_val = int(df.isnull().sum().sum())
    dup_val     = int(df.duplicated().sum())

    # 2x2 grid layout
    r1a, r1b = st.columns(2)
    with r1a:
        st.metric("📋 Rows", rows_val)
    with r1b:
        st.metric("🔢 Columns", cols_val)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    r2a, r2b = st.columns(2)
    with r2a:
        st.metric("❓ Missing", missing_val)
    with r2b:
        st.metric("♻️ Duplicates", dup_val)

st.divider()

# ==========================================
# AVAILABLE MODELS
# ==========================================

st.header("🤖 Machine Learning Models")

st.markdown("""
<style>
.model-card {
    background: rgba(20, 29, 47, 0.6);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 20px 22px 16px;
    margin-bottom: 16px;
    backdrop-filter: blur(14px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.model-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.3);
}
.model-card h4 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    color: #00e676 !important;
    margin: 0 0 10px 0 !important;
    letter-spacing: 0.01em;
}
.model-card p {
    color: #94a3b8 !important;
    font-size: 0.93rem !important;
    line-height: 1.8 !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)

model_col1, model_col2 = st.columns(2)

with model_col1:
    st.markdown("""
    <div class="model-card">
        <h4>⚡ Logistic Regression</h4>
        <p>✔ Fast &amp; Efficient<br>✔ High Accuracy<br>✔ Linear Classification</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="model-card">
        <h4>🌿 Decision Tree</h4>
        <p>✔ Easy Interpretation<br>✔ Non Linear<br>✔ Fast Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="model-card">
        <h4>🌲 Random Forest</h4>
        <p>✔ Ensemble Learning<br>✔ Robust<br>✔ Very Accurate</p>
    </div>
    """, unsafe_allow_html=True)

with model_col2:
    st.markdown("""
    <div class="model-card">
        <h4>🎯 Support Vector Machine</h4>
        <p>✔ Excellent Classification<br>✔ High Performance<br>✔ Margin Based</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="model-card">
        <h4>🔍 K-Nearest Neighbors</h4>
        <p>✔ Instance Based<br>✔ Simple<br>✔ Distance Metric</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="model-card">
        <h4>📊 Naive Bayes</h4>
        <p>✔ Probability Based<br>✔ Extremely Fast<br>✔ Lightweight</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==========================================
# QUICK INFORMATION
# ==========================================

st.header("📋 Project Information")

info1, info2 = st.columns(2)

with info1:

    st.markdown("""
### 🧠 Technologies

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Plotly
- Matplotlib
""")

with info2:

    st.markdown("""
### 📌 Features

- Real Time Prediction
- Data Analytics
- Model Comparison
- Dataset Exploration
- Professional Dashboard
- AI Based Diagnosis
""")

st.divider()

# ==========================================
# PROJECT WORKFLOW
# ==========================================

st.header("⚙️ Prediction Workflow")

funnel_html = """
<style>
.funnel-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    margin: 2rem 0;
}
.funnel-step {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 12px 20px;
    text-align: center;
    font-weight: 600;
    color: #e2e8f0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
.funnel-step:hover {
    transform: translateY(-2px);
    background: rgba(30, 41, 59, 0.8);
    border-color: rgba(66, 153, 225, 0.5);
    color: #fff;
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
}
.funnel-step.step-1 { width: 90%; background: linear-gradient(90deg, rgba(56, 178, 172, 0.15), rgba(66, 153, 225, 0.15)); border-color: rgba(56, 178, 172, 0.3); }
.funnel-step.step-2 { width: 80%; background: linear-gradient(90deg, rgba(66, 153, 225, 0.15), rgba(102, 126, 234, 0.15)); border-color: rgba(66, 153, 225, 0.3); }
.funnel-step.step-3 { width: 70%; background: linear-gradient(90deg, rgba(102, 126, 234, 0.15), rgba(159, 122, 234, 0.15)); border-color: rgba(102, 126, 234, 0.3); }
.funnel-step.step-4 { width: 60%; background: linear-gradient(90deg, rgba(159, 122, 234, 0.15), rgba(237, 100, 166, 0.15)); border-color: rgba(159, 122, 234, 0.3); }
.funnel-step.step-5 { width: 50%; background: linear-gradient(90deg, rgba(237, 100, 166, 0.15), rgba(245, 101, 101, 0.15)); border-color: rgba(237, 100, 166, 0.3); }
.funnel-step.step-6 { width: 40%; background: linear-gradient(90deg, rgba(245, 101, 101, 0.15), rgba(236, 201, 75, 0.15)); border-color: rgba(245, 101, 101, 0.3); }
.funnel-step.step-7 { 
    width: 30%; 
    background: linear-gradient(90deg, rgba(72, 187, 120, 0.25), rgba(56, 178, 172, 0.25)); 
    border-color: rgba(72, 187, 120, 0.5);
    font-size: 1.1em;
    color: #fff;
    box-shadow: 0 0 15px rgba(72, 187, 120, 0.2);
}
.funnel-step.step-7:hover {
    background: linear-gradient(90deg, rgba(72, 187, 120, 0.4), rgba(56, 178, 172, 0.4)); 
    box-shadow: 0 0 20px rgba(72, 187, 120, 0.4);
}
.arrow-down {
    width: 0;
    height: 0;
    border-left: 8px solid transparent;
    border-right: 8px solid transparent;
    border-top: 8px solid rgba(255, 255, 255, 0.2);
    margin: 2px 0;
}
</style>

<div class="funnel-container">
    <div class="funnel-step step-1">Patient Data</div>
    <div class="arrow-down"></div>
    <div class="funnel-step step-2">Data Preprocessing</div>
    <div class="arrow-down"></div>
    <div class="funnel-step step-3">Feature Scaling</div>
    <div class="arrow-down"></div>
    <div class="funnel-step step-4">Machine Learning Model</div>
    <div class="arrow-down"></div>
    <div class="funnel-step step-5">Prediction</div>
    <div class="arrow-down"></div>
    <div class="funnel-step step-6">Confidence Score</div>
    <div class="arrow-down"></div>
    <div class="funnel-step step-7">Final Diagnosis 🩺</div>
</div>
"""
st.markdown(funnel_html, unsafe_allow_html=True)

# ==========================================
# VISUAL ANALYTICS
# ==========================================

import plotly.express as px

st.header("📊 Dataset Visual Analytics")

chart_col1, chart_col2 = st.columns(2)

# ------------------------------------------
# Target Distribution
# ------------------------------------------

with chart_col1:

    st.subheader("Diagnosis Distribution")

    target_col = None

    for col in ["diagnosis", "target"]:
        if col in df.columns:
            target_col = col
            break

    if target_col is not None:

        counts = df[target_col].value_counts()

        fig = px.pie(
            values=counts.values,
            names=counts.index.astype(str),
            hole=0.45,
            title="Benign vs Malignant",
            template="plotly_dark",
            color_discrete_sequence=["#00e676", "#ff4d4d"])

        fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)")
        
        fig.update_traces(
            textfont_color="black",
            textfont_size=16
        )
        st.plotly_chart(
            fig,
            use_container_width=True)
    else:
        st.warning("Target column not found.")


# ------------------------------------------
# Feature Histogram
# ------------------------------------------

with chart_col2:

    st.subheader("Feature Distribution")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) > 0:

        feature = st.selectbox(
            "Select Feature",
            numeric_cols,
            key="feature_hist"
        )

        fig = px.histogram(
            df,
            x=feature,
            nbins=30,
            title=feature,
            template="plotly_dark",
            color_discrete_sequence=["#06b6d4"])

        fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(
            fig,
            use_container_width=True)
st.divider()

# ==========================================
# CORRELATION HEATMAP
# ==========================================

st.header("🔥 Correlation Heatmap")

corr = df.select_dtypes(include="number").corr()

fig = px.imshow(
    corr,
    aspect="auto",
    color_continuous_scale="Viridis",
    title="Feature Correlation",
    template="plotly_dark"
)

fig.update_layout(
    height=700,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(
    fig,
    use_container_width=True)
st.divider()

# ==========================================
# TOP FEATURES
# ==========================================

st.header("🏆 Top Numerical Features")

variance = (
    df.select_dtypes(include="number")
      .var()
      .sort_values(ascending=False)
      .head(10)
)
fig = px.bar(
    x=variance.values,
    y=variance.index,
    orientation="h",
    title="Top 10 Features (Variance)",
    template="plotly_dark",
    color_discrete_sequence=["#00e676"])
fig.update_layout(
    height=500,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(
    fig,
    use_container_width=True)
st.divider()

# ==========================================
# DATA SUMMARY
# ==========================================

st.header("📋 Dataset Statistics")

st.dataframe(
    df.describe().T,
    use_container_width=True)
st.divider()

# ==========================================
# DOWNLOAD DATASET
# ==========================================

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Dataset",
    csv,
    file_name="breast_cancer_dataset.csv",
    mime="text/csv")
st.divider()

# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <hr>

    <div style="text-align:center;padding:20px">

    <h3>🩺 Breast Cancer Detection AI</h3>

    <p>
    Machine Learning Based Diagnosis System
    </p>

    <p>
    Built with using
    Streamlit • Scikit-Learn • Pandas • Plotly
    </p>

    <p>
    © 2026 All Rights Reserved
    </p>

    </div>
    """,
    unsafe_allow_html=True)

