import pandas as pd
import streamlit as st
from predict import DiseasePredictor


st.set_page_config(page_title="Disease Prediction System", page_icon=None, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family:'Inter', sans-serif;
    }

    .stApp{
        background-color:#FFFDF7;
    }

    #MainMenu, footer, header{
        visibility:hidden;
    }

    .block-container{
        padding-top:2rem;
        padding-bottom:2rem;
        max-width:1200px;
    }

    h1,h2,h3,h4,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stHeading"],
    [data-testid="stHeading"] *{
        color:#000000 !important;
        font-weight:700;
    }

    p,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stText"],
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p{
        color:#000000 !important;
    }

    /* ---- Sidebar ---- */
    div[data-testid="stSidebar"]{
        background-color:#FFFDF7;
        border-right:1px solid #E9E2D3;
    }

    .sidebar-brand{
        display:flex;
        align-items:center;
        gap:10px;
        padding:6px 0 2px 0;
    }

    .sidebar-brand-text{
        font-size:19px;
        font-weight:800;
        color:#000000;
        line-height:1.15;
    }

    .sidebar-brand-sub{
        font-size:12px;
        color:#000000;
        margin-top:2px;
    }

    .stat-grid{
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:10px;
        margin-top:6px;
    }

    .stat-box{
        background:#FFFFFF;
        border:1px solid #EDE6D6;
        border-radius:10px;
        padding:12px 10px;
    }

    .stat-label{
        font-size:11px;
        font-weight:600;
        color:#000000;
        text-transform:uppercase;
        letter-spacing:.04em;
    }

    .stat-value{
        font-size:18px;
        font-weight:700;
        color:#000000;
        margin-top:2px;
    }

    .algo-pill{
        display:inline-flex;
        align-items:center;
        gap:6px;
        background:#EFF6FF;
        color:#2563EB;
        font-size:12px;
        font-weight:600;
        padding:6px 12px;
        border-radius:999px;
        margin-top:4px;
    }

    /* ---- Section card ---- */
    .section-heading{
        font-size:16px;
        font-weight:700;
        color:#000000;
        display:flex;
        align-items:center;
        gap:8px;
        margin-bottom:4px;
    }

    .section-caption{
        font-size:13px;
        color:#000000;
        margin-bottom:16px;
    }

    .selected-badge{
        display:inline-block;
        background:#EEF2FF;
        color:#4F46E5;
        font-weight:700;
        font-size:13px;
        padding:5px 14px;
        border-radius:999px;
    }

    /* search field - normal light style */
    div[data-testid="stTextInput"] input{
        background-color:#FFFFFF;
        color:#000000;
    }

    div[data-testid="stTextInput"] input::placeholder{
        color:#6B7280;
    }

    div[data-testid="stTextInput"] input:hover,
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextInput"] input:active{
        background-color:#FFFFFF !important;
        color:#000000 !important;
        box-shadow:none !important;
        outline:none !important;
    }

    div[data-testid="stTextInput"]:hover,
    div[data-testid="stTextInput"]:focus-within{
        border:none !important;
        box-shadow:none !important;
    }

    div[data-testid="stTextInputRootElement"]{
        background-color:#FFFFFF !important;
        border:1px solid #E5E7EB !important;
        border-radius:8px !important;
        box-shadow:none !important;
    }

    div[data-testid="stTextInputRootElement"]:hover,
    div[data-testid="stTextInputRootElement"]:focus-within{
        background-color:#FFFFFF !important;
        border:1px solid #E5E7EB !important;
        box-shadow:none !important;
    }

    /* checkboxes - normal light style */
    div[data-testid="stCheckbox"]{
        background:#FFFFFF;
        border:1px solid #E5E7EB;
        border-radius:8px;
        padding:6px 10px;
        margin-bottom:6px;
        transition:all 0.15s ease;
    }

    div[data-testid="stCheckbox"]:hover{
        background:#F9FAFB;
        border-color:#D1D5DB;
    }

    div[data-testid="stCheckbox"] label p{
        font-size:13.5px;
        color:#000000;
    }

    div[data-testid="stCheckbox"] span[data-testid="stTickIndicator"]{
        background-color:#FFFFFF !important;
        border-color:#000000 !important;
    }

    div[data-testid="stCheckbox"],
    div[data-testid="stCheckbox"] *{
        color-scheme:light !important;
    }

    div[data-testid="stCheckbox"] input[type="checkbox"]{
        accent-color:#4F46E5 !important;
        background-color:#FFFFFF !important;
    }

    div[data-testid="stCheckbox"] [data-baseweb="checkbox"] > span:first-child{
        background-color:#FFFFFF !important;
        border-color:#000000 !important;
    }

    div[data-testid="stCheckbox"] input[aria-checked="false"] + span,
    div[data-testid="stCheckbox"] input:not(:checked) + span{
        background-color:#FFFFFF !important;
        border-color:#000000 !important;
    }

    /* buttons */
    .stButton button{
        border-radius:10px;
        font-weight:700;
        font-size:15px;
        padding:0.65rem 1rem;
        border:none;
        box-shadow:0 6px 16px -6px rgba(37,99,235,0.55);
    }

    /* ---- Prediction result ---- */
    .prediction-card{
        background:linear-gradient(135deg, #111827 0%, #1F2A44 100%);
        border-radius:18px;
        padding:32px 34px;
        margin-top:8px;
        margin-bottom:22px;
    }

    .prediction-title{
        color:#93A5C9 !important;
        font-size:13px;
        font-weight:600;
        letter-spacing:.05em;
        text-transform:uppercase;
        margin-bottom:6px;
    }

    .prediction-value{
        color:#FFFFFF !important;
        font-size:32px;
        font-weight:800;
    }

    /* top predictions rows */
    .pred-row{
        display:flex;
        align-items:center;
        gap:14px;
        padding:10px 0;
        border-bottom:1px solid #F1F3F6;
    }

    .pred-row:last-child{
        border-bottom:none;
    }

    .pred-rank{
        width:26px;
        height:26px;
        border-radius:50%;
        background:#EEF2FF;
        color:#4F46E5;
        font-size:12px;
        font-weight:800;
        display:flex;
        align-items:center;
        justify-content:center;
        flex-shrink:0;
    }

    .pred-name{
        flex:1;
        font-size:14px;
        font-weight:600;
        color:#000000;
    }

    .footer-box{
        text-align:center;
        color:#000000;
        padding:18px 0 4px 0;
        font-size:13px;
        border-top:1px solid #E5E7EB;
        margin-top:10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> DiseasePredictor:
    return DiseasePredictor()


predictor = load_predictor()

symptoms = predictor.feature_names()


with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div>
                <div class="sidebar-brand-text">Disease Predictor</div>
                <div class="sidebar-brand-sub">Clinical decision support</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="algo-pill">Random Forest Model</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-label">Diseases</div>
                <div class="stat-value">500+</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Symptoms</div>
                <div class="stat-value">{len(symptoms)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Dataset</div>
                <div class="stat-value">100K</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Accuracy</div>
                <div class="stat-value">84.3%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("For informational purposes only. Not a substitute for professional medical advice.")


st.title("Disease Prediction System")
st.write("Select a patient's symptoms below and get an instant, data-driven diagnosis estimate.")
st.write("")

if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = set()


st.markdown('<div class="section-card">', unsafe_allow_html=True)

top_row_left, top_row_right = st.columns([3, 1])

with top_row_left:
    st.markdown('<div class="section-heading">Select Symptoms</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Search and check every symptom the patient is experiencing.</div>', unsafe_allow_html=True)

with top_row_right:
    st.markdown(
        f"""
        <div style="text-align:right; margin-top:6px;">
            <span class="selected-badge">{len(st.session_state.selected_symptoms)} selected</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

search = st.text_input("Search Symptoms", placeholder="Type to search symptoms...", label_visibility="collapsed")

if search:
    filtered_symptoms = [symptom for symptom in symptoms if search.lower() in symptom.lower()]
else:
    filtered_symptoms = symptoms

st.write("")

if len(filtered_symptoms) == 0:
    st.info("No symptoms match your search.")
else:
    columns = st.columns(3)
    for index, symptom in enumerate(filtered_symptoms):
        column = columns[index % 3]
        with column:
            checked = symptom in st.session_state.selected_symptoms
            value = st.checkbox(symptom.replace("_", " ").title(), value=checked, key=f"checkbox_{symptom}")
            if value:
                st.session_state.selected_symptoms.add(symptom)
            else:
                st.session_state.selected_symptoms.discard(symptom)

st.markdown('</div>', unsafe_allow_html=True)


predict_col, clear_col = st.columns([4, 1])

with predict_col:
    predict = st.button("Predict Disease", type="primary", use_container_width=True)

with clear_col:
    if st.button("Clear All", use_container_width=True):
        st.session_state.selected_symptoms = set()
        st.rerun()


if predict:

    if len(st.session_state.selected_symptoms) == 0:

        st.error("Please select at least one symptom.")

    else:

        patient = pd.DataFrame([[0] * len(symptoms)], columns=symptoms)

        for symptom in st.session_state.selected_symptoms:
            if symptom in patient.columns:
                patient.loc[0, symptom] = 1

        with st.spinner("Analyzing symptoms..."):
            disease = predictor.predict(patient)
            top_predictions = predictor.predict_top_predictions(patient)

        col1 = st.columns(1)[0]

        with col1:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-title">
                        Predicted Disease
                    </div>
                    <div class="prediction-value">
                        {disease}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Top Predictions</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Other likely diagnoses considered by the model.</div>', unsafe_allow_html=True)

        top_prediction_df = pd.DataFrame(
            {
                "Disease": [item[0] for item in top_predictions],
            }
        )

        rows_html = ""
        for rank, name in enumerate(top_prediction_df["Disease"], start=1):
            rows_html += f"""
            <div class="pred-row">
                <div class="pred-rank">{rank}</div>
                <div class="pred-name">{name}</div>
            </div>
            """

        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


st.markdown(
    """
    <div class="footer-box">
        Disease Prediction System &nbsp;•&nbsp; Built with Python, Scikit-learn &amp; Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)