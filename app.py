
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mental Health Risk Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 17px;
        color: #666;
        margin-bottom: 25px;
    }

    .card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #ddd;
        background-color: #ffffff;
        margin-bottom: 15px;
    }

    .risk-high {
        padding: 20px;
        border-radius: 15px;
        background-color: #ffe5e5;
        border: 2px solid #ff4b4b;
        text-align: center;
    }

    .risk-moderate {
        padding: 20px;
        border-radius: 15px;
        background-color: #fff4cc;
        border: 2px solid #f0ad00;
        text-align: center;
    }

    .risk-low {
        padding: 20px;
        border-radius: 15px;
        background-color: #e5f7e5;
        border: 2px solid #28a745;
        text-align: center;
    }

    .metric-title {
        font-size: 14px;
        color: #666;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 Mental Health Risk Prediction & Analytics System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Cross-Domain Machine Learning Assessment for Workplace & Academic Environments'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

@st.cache_resource
def load_artifacts():

    possible_model_paths = [
        "models/unified_mental_health_model.pkl",
        "../models/unified_mental_health_model.pkl"
    ]

    possible_preprocessor_paths = [
        "models/unified_preprocessor.pkl",
        "../models/unified_preprocessor.pkl"
    ]

    model_path = None
    prep_path = None

    # Find model
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            break

    # Find preprocessor
    for path in possible_preprocessor_paths:
        if os.path.exists(path):
            prep_path = path
            break

    if model_path is None:
        raise FileNotFoundError(
            "Could not find unified_mental_health_model.pkl"
        )

    if prep_path is None:
        raise FileNotFoundError(
            "Could not find unified_preprocessor.pkl"
        )

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)

    return model, preprocessor


# ============================================================
# MODEL LOADING
# ============================================================

try:

    model, preprocessor = load_artifacts()

    st.sidebar.success("✅ ML Pipeline Loaded")

except Exception as e:

    st.sidebar.error(f"❌ Model Loading Error: {e}")

    st.error(
        "The ML model or preprocessor could not be loaded. "
        "Please check your models folder."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Assessment Controls")

st.sidebar.markdown("---")

domain = st.sidebar.selectbox(
    "Domain Assessment Type",
    ["Workplace", "Student"]
)

age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=70,
    value=25
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

family_history = st.sidebar.selectbox(
    "Family History of Mental Illness",
    ["No", "Yes", "Don't know"]
)

interference = st.sidebar.selectbox(
    "Work/Academic Interference Level",
    [
        "Never",
        "Rarely",
        "Sometimes",
        "Often",
        "Don't know"
    ]
)

anxiety = st.sidebar.selectbox(
    "Diagnosed / Experiencing Anxiety",
    ["No", "Yes", "Don't know"]
)

remote_work = st.sidebar.selectbox(
    "Remote Work / Study Setup",
    ["No", "Yes"]
)

benefits = st.sidebar.selectbox(
    "Mental Health Care Benefits Offered",
    ["No", "Yes", "Don't know"]
)

seek_help = st.sidebar.selectbox(
    "Accessible Mental Health Resources",
    ["No", "Yes", "Don't know"]
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def assign_age_group(age_value):

    if age_value <= 25:
        return "Young_Adult_18_25"

    elif age_value <= 40:
        return "Mid_Career_26_40"

    else:
        return "Senior_41_Plus"


def assess_resource_support(benefit_value, resource_value):

    benefit_value = str(benefit_value).lower()
    resource_value = str(resource_value).lower()

    if benefit_value == "yes" and resource_value == "yes":
        return "High_Support"

    elif benefit_value == "yes" or resource_value == "yes":
        return "Moderate_Support"

    else:
        return "Low_Support_Or_Unknown"


age_group = assign_age_group(age)

resource_support = assess_resource_support(
    benefits,
    seek_help
)


high_interference = (
    "True"
    if interference in ["Often", "Sometimes"]
    else "False"
)


family_x_anxiety = (
    "True"
    if family_history == "Yes" and anxiety == "Yes"
    else "False"
)


support_gap = (
    "True"
    if benefits == "No" and interference == "Often"
    else "False"
)


# ============================================================
# INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame(
    {
        "Age": [age],
        "Gender": [gender],
        "Domain": [domain],
        "Family_History": [family_history],
        "Remote_Work_Study": [remote_work],
        "Interference_Level": [interference],
        "Anxiety_Flag": [anxiety],
        "Age_Group": [age_group],
        "Resource_Support_Index": [resource_support],
        "High_Interference_Flag": [high_interference],
        "Family_x_Anxiety": [family_x_anxiety],
        "Support_Gap": [support_gap]
    }
)


# ============================================================
# PREDICTION
# ============================================================

try:

    X_processed = preprocessor.transform(input_data)

    probabilities = model.predict_proba(X_processed)[0]

    # Probability of positive/high-risk class
    if len(probabilities) > 1:
        risk_probability = float(probabilities[1])
    else:
        risk_probability = float(probabilities[0])

    # Model decision threshold
    risk_prediction = 1 if risk_probability >= 0.40 else 0

except Exception as e:

    st.error(f"Prediction Error: {e}")

    st.write("Input data sent to the model:")
    st.dataframe(input_data)

    st.stop()


# Convert probability to percentage

risk_percentage = risk_probability * 100


# ============================================================
# RISK CATEGORY
# ============================================================

if risk_percentage < 40:

    risk_category = "Low Risk"
    risk_class = "risk-low"
    risk_icon = "🟢"

elif risk_percentage < 70:

    risk_category = "Moderate Risk"
    risk_class = "risk-moderate"
    risk_icon = "🟡"

else:

    risk_category = "High Risk"
    risk_class = "risk-high"
    risk_icon = "🔴"


# ============================================================
# TOP KPI CARDS
# ============================================================

st.markdown("## 📊 Risk Assessment Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="card">
            <div class="metric-title">Risk Probability</div>
            <div class="metric-value">{risk_percentage:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="card">
            <div class="metric-title">Risk Category</div>
            <div class="metric-value">{risk_icon} {risk_category}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="card">
            <div class="metric-title">Assessment Domain</div>
            <div class="metric-value">{domain}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="card">
            <div class="metric-title">Age Group</div>
            <div class="metric-value">{age_group}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RISK STATUS
# ============================================================

st.markdown("## 🎯 Overall Risk Status")

st.markdown(
    f"""
    <div class="{risk_class}">
        <h2>{risk_icon} {risk_category}</h2>
        <h3>Estimated Risk Probability: {risk_percentage:.2f}%</h3>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INTERACTIVE RISK GAUGE
# ============================================================

st.markdown("## 📈 Interactive Risk Gauge")

gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=risk_percentage,
        title={
            "text": "Mental Health Risk Probability"
        },
        number={
            "suffix": "%"
        },
        gauge={
            "axis": {
                "range": [0, 100]
            },
            "steps": [
                {
                    "range": [0, 40],
                    "color": "lightgreen"
                },
                {
                    "range": [40, 70],
                    "color": "lightyellow"
                },
                {
                    "range": [70, 100],
                    "color": "lightcoral"
                }
            ],
            "threshold": {
                "line": {
                    "width": 4
                },
                "thickness": 0.75,
                "value": risk_percentage
            }
        }
    )
)

gauge.update_layout(
    height=350
)

st.plotly_chart(
    gauge,
    use_container_width=True
)


# ============================================================
# TWO-COLUMN ANALYTICS
# ============================================================

left_col, right_col = st.columns(2)


# ============================================================
# RISK FACTORS
# ============================================================

with left_col:

    st.markdown("## ⚠️ Risk Factor Analysis")

    risk_factors = {
        "Family History": 0,
        "Anxiety": 0,
        "Work/Study Interference": 0,
        "Support Gap": 0,
        "Limited Resources": 0
    }

    if family_history == "Yes":
        risk_factors["Family History"] = 1

    if anxiety == "Yes":
        risk_factors["Anxiety"] = 1

    if interference in ["Often", "Sometimes"]:
        risk_factors["Work/Study Interference"] = 1

    if support_gap == "True":
        risk_factors["Support Gap"] = 1

    if benefits != "Yes" or seek_help != "Yes":
        risk_factors["Limited Resources"] = 1

    factor_df = pd.DataFrame(
        {
            "Risk Factor": list(risk_factors.keys()),
            "Present": list(risk_factors.values())
        }
    )

    factor_df["Status"] = factor_df["Present"].map(
        {
            1: "Present",
            0: "Not Present"
        }
    )

    fig_factors = px.bar(
        factor_df,
        x="Present",
        y="Risk Factor",
        orientation="h",
        text="Status",
        title="Identified Risk Factors",
        range_x=[0, 1.2]
    )

    fig_factors.update_layout(
        height=400,
        xaxis_title="Indicator",
        yaxis_title=""
    )

    st.plotly_chart(
        fig_factors,
        use_container_width=True
    )


# ============================================================
# MODEL PROBABILITY
# ============================================================

with right_col:

    st.markdown("## 🤖 Model Prediction Probability")

    if len(probabilities) >= 2:

        probability_df = pd.DataFrame(
            {
                "Class": [
                    "Lower Risk",
                    "Higher Risk"
                ],
                "Probability": [
                    probabilities[0] * 100,
                    probabilities[1] * 100
                ]
            }
        )

    else:

        probability_df = pd.DataFrame(
            {
                "Class": ["Predicted Class"],
                "Probability": [
                    probabilities[0] * 100
                ]
            }
        )

    fig_probability = px.bar(
        probability_df,
        x="Class",
        y="Probability",
        text="Probability",
        title="Prediction Confidence"
    )

    fig_probability.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_probability.update_layout(
        height=400,
        yaxis_title="Probability (%)",
        xaxis_title=""
    )

    st.plotly_chart(
        fig_probability,
        use_container_width=True
    )


# ============================================================
# USER PROFILE SUMMARY
# ============================================================

st.markdown("## 👤 Assessment Profile")

profile_col1, profile_col2, profile_col3 = st.columns(3)


with profile_col1:

    st.markdown("### Personal Information")

    st.write(f"**Age:** {age}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Age Group:** {age_group}")
    st.write(f"**Domain:** {domain}")


with profile_col2:

    st.markdown("### Mental Health Indicators")

    st.write(f"**Family History:** {family_history}")
    st.write(f"**Anxiety:** {anxiety}")
    st.write(f"**Interference:** {interference}")


with profile_col3:

    st.markdown("### Support Environment")

    st.write(f"**Remote Work/Study:** {remote_work}")
    st.write(f"**Mental Health Benefits:** {benefits}")
    st.write(f"**Accessible Resources:** {seek_help}")
    st.write(f"**Support Level:** {resource_support}")


# ============================================================
# RECOMMENDATIONS
# ============================================================

st.markdown("## 💡 Personalized Recommendations")

recommendations = []


if risk_percentage >= 70:

    recommendations.append(
        "Consider speaking with a qualified mental health professional "
        "or an appropriate support service."
    )

elif risk_percentage >= 40:

    recommendations.append(
        "Monitor your stress and emotional wellbeing regularly."
    )

else:

    recommendations.append(
        "Continue maintaining healthy routines and positive support systems."
    )


if anxiety == "Yes":

    recommendations.append(
        "Consider professional guidance or evidence-based stress-management techniques."
    )


if interference in ["Often", "Sometimes"]:

    recommendations.append(
        "Review workload, study pressure, breaks, sleep, and work-life balance."
    )


if benefits != "Yes":

    recommendations.append(
        "Explore available mental health benefits or support programs."
    )


if seek_help != "Yes":

    recommendations.append(
        "Identify accessible counselling, wellness, or mental health resources."
    )


if family_history == "Yes":

    recommendations.append(
        "Be aware of your mental wellbeing and seek professional support if symptoms develop."
    )


for recommendation in recommendations:

    st.info("💡 " + recommendation)


# ============================================================
# RAW INPUT DATA
# ============================================================

with st.expander("🔍 View Model Input Data"):

    st.dataframe(
        input_data,
        use_container_width=True
    )


# ============================================================
# DOWNLOAD REPORT
# ============================================================

st.markdown("## 📥 Download Assessment")

report_data = input_data.copy()

report_data["Risk_Probability"] = risk_percentage
report_data["Risk_Category"] = risk_category
report_data["Model_Prediction"] = risk_prediction

csv_data = report_data.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="⬇️ Download Assessment Report",
    data=csv_data,
    file_name="mental_health_risk_assessment.csv",
    mime="text/csv"
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("---")

st.warning(
    "⚠️ This system is an AI/ML-based risk prediction and analytics tool. "
    "It is not a medical diagnosis. The results should not replace assessment "
    "or advice from a qualified mental health professional."
)


st.caption(
    "Mental Health Risk Prediction & Analytics System | "
    "Machine Learning Dashboard"
)
