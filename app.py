import streamlit as st
import joblib
import numpy as np
import base64

# ---------------- LOAD MODEL ----------------
model = joblib.load("weather_model.pkl")
le = joblib.load("label_encoder.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Weather Forecast App",
    page_icon="🌤️",
    layout="wide"
)

# ---------------- LOAD BACKGROUND IMAGE ----------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("weather_bg.jpg")

# ---------------- FULL UI STYLING ----------------
st.markdown(f"""
<style>

/* ===== BACKGROUND IMAGE ===== */
.stApp {{
    background: url("data:image/jpg;base64,{img}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* ===== REMOVE STREAMLIT HEADER BACKGROUND ===== */
header {{
    background: transparent !important;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0) !important;
    backdrop-filter: blur(10px);
}}

[data-testid="stDecoration"] {{
    display: none;
}}

/* ===== DARK OVERLAY FOR READABILITY ===== */
.stApp::before {{
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 20, 50, 0.40);
    z-index: 0;
}}

/* KEEP CONTENT ABOVE BACKGROUND */
.block-container {{
    position: relative;
    z-index: 1;
}}

/* ===== TITLE ===== */
.title {{
    font-size: 46px;
    font-weight: 700;
    color: white;
    text-align: center;
    margin-bottom: 10px;
    text-shadow: 0px 5px 20px rgba(0,0,0,0.6);
}}

/* ===== GLASS CARD ===== */
.glass {{
    background: rgba(255, 255, 255, 0.12);
    border-radius: 20px;

    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);

    border: 1px solid rgba(255, 255, 255, 0.25);

    box-shadow: 0 8px 32px rgba(0,0,0,0.3);

    padding: 25px;
    margin-top: 15px;
}}

/* ===== BUTTON ===== */
.stButton > button {{
    width: 100%;
    background: rgba(255,255,255,0.15);
    color: white;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.25);
    font-weight: bold;
    backdrop-filter: blur(10px);
}}

.stButton > button:hover {{
    background: rgba(255,255,255,0.30);
    transform: scale(1.02);
    transition: 0.2s;
}}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<div class='title'>🌤️ Weather Forecast Dashboard</div>", unsafe_allow_html=True)

st.write("Predict temperature using your ML model")

# ---------------- INPUT SECTION ----------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("🏙️ City", le.classes_)
    month = st.slider("📅 Month", 1, 12, 6)
    prcp = st.slider("🌧️ Rainfall", 0.0, 100.0, 10.0)

with col2:
    pres = st.slider("🌡️ Pressure", 900.0, 1100.0, 1013.0)
    tsun = st.slider("☀️ Sunshine", 0.0, 12.0, 6.0)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)

if st.button("🔮 Predict Weather"):

    city_encoded = le.transform([city])[0]

    features = np.array([[month, city_encoded, prcp, pres, tsun]])

    prediction = model.predict(features)[0]

    st.markdown(
        f"""
        <h1 style='color:white;text-align:center;'>
        🌡️ {prediction:.2f} °C
        </h1>
        """,
        unsafe_allow_html=True
    )

    if prediction > 30:
        st.success("🔥 Hot Weather Expected")
    elif prediction > 20:
        st.info("🌤️ Pleasant Weather")
    else:
        st.warning("❄️ Cold Weather Expected")

st.markdown("</div>", unsafe_allow_html=True)