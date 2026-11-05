from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_DESC_LEN = 3000

CATEGORICAL = [
    "Size",
    "Type of ownership",
    "Industry",
    "Sector",
    "job_state",
    "job_simp",
    "seniority",
]

SKILL_KEYWORDS = {
    "python_yn": "python",
    "spark": "spark",
    "aws": "aws",
    "excel": "excel",
}

NOT_SPECIFIED = "Not specified"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_random_forest.pkl")
    columns = joblib.load(MODELS_DIR / "columns.pkl")
    return model, columns


def options_for(prefix, columns):
    options = {}
    for col in columns:
        if col.startswith(prefix + "_"):
            label = col[len(prefix) + 1:].strip()
            options[NOT_SPECIFIED if label == "-1" else label] = col
    return options


def build_features(columns, numeric, selected_columns):
    row = dict.fromkeys(columns, 0)
    row.update(numeric)
    for col in selected_columns:
        row[col] = 1
    return pd.DataFrame([row], columns=columns)


model, columns = load_artifacts()

st.set_page_config(page_title="Salary Predictor", page_icon="💼")
st.title("Data Science Salary Predictor")
st.write("Estimate a data science salary from company and job details.")

# --- ورودی‌های عددی و بله/خیر ---
left, right = st.columns(2)

with left:
    rating = st.slider("Company rating", 1.0, 5.0, 3.5, 0.1)
    age = st.number_input("Company age (years)", 0, 300, 20)
    num_comp = st.number_input("Number of competitors", 0, 10, 1)

with right:
    same_state = st.checkbox("Job in same state as HQ")
    hourly = st.checkbox("Hourly wage")
    employer_provided = st.checkbox("Salary provided by employer")

# --- ورودی‌های دسته‌ای (dropdown) ---
choices = {}
dropdown_left, dropdown_right = st.columns(2)
for i, name in enumerate(CATEGORICAL):
    options = options_for(name, columns)
    with dropdown_left if i % 2 == 0 else dropdown_right:
        label = st.selectbox(name, list(options.keys()))
    choices[name] = options[label]

# --- متن آگهی (مهارت‌ها و طول متن از اینجا استخراج می‌شن) ---
job_desc = st.text_area("Paste the job description (optional)", height=200)

text = job_desc.lower()
skills = {col: int(word in text) for col, word in SKILL_KEYWORDS.items()}
desc_len = len(job_desc) if job_desc.strip() else DEFAULT_DESC_LEN

if job_desc.strip():
    detected = [word for col, word in SKILL_KEYWORDS.items() if skills[col]]
    st.caption("Detected skills: " + (", ".join(detected) if detected else "none"))

# --- پیش‌بینی ---
if st.button("Predict", type="primary"):
    numeric = {
        "Rating": rating,
        "age": age,
        "num_comp": num_comp,
        "desc_len": desc_len,
        "same_state": int(same_state),
        "hourly": int(hourly),
        "employer_provided": int(employer_provided),
        **skills,
    }
    X = build_features(columns, numeric, choices.values())
    prediction = model.predict(X)[0]
    st.success(f"Predicted salary: ${prediction:.0f}K per year")