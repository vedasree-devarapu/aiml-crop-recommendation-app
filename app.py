# Crop Recommendation & Soil Analyzer - Streamlit App
# Keep this file in the same folder as:
#   1. crop_model.pkl
#   2. Crop_recommendation.csv

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Crop Recommendation & Soil Analyzer",
    page_icon="🌱",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_model.pkl"
DATA_PATH = BASE_DIR / "Crop_recommendation.csv"

FEATURE_LABELS = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "ph": "Soil pH",
    "rainfall": "Rainfall (mm)",
}

DEFAULT_VALUES = {
    "N": 50.0,
    "P": 50.0,
    "K": 50.0,
    "temperature": 25.0,
    "humidity": 70.0,
    "ph": 6.5,
    "rainfall": 100.0,
}

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(rgba(240, 255, 240, 0.90),
                            rgba(240, 255, 240, 0.90)),
            url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=1920");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main-heading {
        background: linear-gradient(120deg, #07552b, #16834a);
        padding: 26px 18px;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 22px;
        color: white;
    }
    .main-heading h1, .main-heading h3, .main-heading p {
        color: white !important;
    }
    .content-panel {
        background: rgba(255, 255, 255, 0.94);
        padding: 20px;
        border: 1px solid #b7d8b8;
        border-radius: 16px;
        margin-bottom: 15px;
    }
    div.stButton > button {
        background-color: #16834a;
        color: white;
        border-radius: 10px;
        border: 0;
        font-weight: 700;
    }
    div.stButton > button:hover {
        background-color: #07552b;
        color: white;
    }
    /* Make all text clearly visible */
.stMarkdown,
.stMarkdown p,
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown li,
.stMarkdown strong,
.stText,
label {
    color: #183b20 !important;
}

/* Navigation tab text */
button[data-baseweb="tab"] {
    color: #183b20 !important;
    font-weight: bold !important;
}

/* Main page text */
[data-testid="stMarkdownContainer"] {
    color: #183b20 !important;
}

/* Input labels */
[data-testid="stWidgetLabel"] {
    color: #183b20 !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------
@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)


@st.cache_data
def load_dataset(path_string):
    return pd.read_csv(path_string)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Place **crop_model.pkl** in the same folder "
        "as this Streamlit app."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load crop_model.pkl: {exc}")
    st.stop()

features = list(getattr(model, "feature_names_in_", []))
if not features:
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

df = None
data_error = None
if DATA_PATH.exists():
    try:
        df = load_dataset(str(DATA_PATH))
    except Exception as exc:
        data_error = str(exc)
else:
    data_error = "Crop_recommendation.csv was not found in the app folder."

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="main-heading">
        <h1>🌱 CROP RECOMMENDATION &amp; SOIL ANALYZER</h1>
        <h3>Smart Agriculture Using Artificial Intelligence</h3>
        <p>Making smarter farming decisions with Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# NAVIGATION TABS
# --------------------------------------------------
home_tab, recommend_tab, analysis_tab, technology_tab, about_tab = st.tabs(
    ["🏠 Home", "🌾 Recommend Crop", "📊 Model Analysis",
     "💻 Technology Used", "ℹ️ About"]
)

with home_tab:
    st.markdown('<div class="content-panel">', unsafe_allow_html=True)
    st.markdown(
        """
        # 🌿 Welcome, Farmer!

        Welcome to our Crop Recommendation & Soil Analyzer.

        Our application uses Artificial Intelligence and the Random Forest
        algorithm to recommend suitable crops based on soil nutrients and
        environmental conditions.

        ---

        ## 🎯 About Our Project

        Agriculture is an important part of our lives. Choosing a suitable crop
        according to soil and weather conditions can help farmers make
        informed agricultural decisions.

        ---

        ## 🔍 What Does Our App Analyze?

        1. Nitrogen (N)
        2. Phosphorus (P)
        3. Potassium (K)
        4. Temperature
        5. Humidity
        6. Soil pH
        7. Rainfall

        ---

        ## 💡 Our Objective

        To develop an AI-based application that analyzes soil and weather
        information and recommends a crop.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with recommend_tab:
    st.markdown("## 🌾 Crop Recommendation")
    st.write(
        "Enter the soil and weather values below. The trained Random Forest "
        "model will predict suitable crops."
    )

    input_values = {}
    columns = st.columns(2)
    for index, feature in enumerate(features):
        label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
        with columns[index % 2]:
            input_values[feature] = st.number_input(
                label,
                value=float(DEFAULT_VALUES.get(feature, 0.0)),
                key=f"input_{feature}",
                format="%.2f",
            )

    if st.button("🌱 Recommend Crop", type="primary", use_container_width=True):
        try:
            input_data = pd.DataFrame([input_values], columns=features)
            probabilities = model.predict_proba(input_data)[0]
            ranked = sorted(
                zip(model.classes_, probabilities),
                key=lambda item: item[1],
                reverse=True,
            )
            predicted_crop = str(model.predict(input_data)[0]).title()

            st.success(f"Recommended crop: **{predicted_crop}**")
            st.markdown("### Top 3 Crop Recommendations")
            for rank, (crop, probability) in enumerate(ranked[:3], start=1):
                st.write(f"**{rank}. {str(crop).title()}** — {probability * 100:.2f}% model score")
                st.progress(float(probability))
            st.caption(
                "Scores are model probabilities, not a guarantee of crop yield "
                "or farming success. Consider local agricultural guidance."
            )
        except AttributeError:
            # Fallback for models that do not implement predict_proba.
            try:
                input_data = pd.DataFrame([input_values], columns=features)
                prediction = model.predict(input_data)[0]
                st.success(f"Recommended crop: **{str(prediction).title()}**")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with analysis_tab:
    st.markdown("## 📊 Model Analysis")
    st.write("Visual evaluation of the trained Random Forest model.")

    if df is None:
        st.warning(
            "Dataset-based charts need **Crop_recommendation.csv** in the same "
            "folder as this app. " + (data_error or "")
        )
    else:
        missing_columns = [column for column in features if column not in df.columns]
        if "label" not in df.columns or missing_columns:
            st.warning(
                "The dataset does not contain the columns required for analysis. "
                f"Missing feature columns: {missing_columns}; "
                f"'label' column present: {'label' in df.columns}."
            )
        else:
            x = df[features]
            y = df["label"]
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42, stratify=y
            )

            # 1. Confusion matrix (held-out test data)
            st.markdown("### 1. Confusion Matrix")
            y_pred = model.predict(x_test)
            labels = list(model.classes_)
            cm = confusion_matrix(y_test, y_pred, labels=labels)

            fig, ax = plt.subplots(figsize=(11, 8))
            image = ax.imshow(cm, cmap="Reds")
            ax.set_title("Random Forest - Confusion Matrix")
            ax.set_xlabel("Predicted Crop")
            ax.set_ylabel("Actual Crop")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_yticklabels(labels)
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # 2. Feature importance
            st.markdown("### 2. Feature Importance")
            if hasattr(model, "feature_importances_"):
                feature_df = pd.DataFrame(
                    {
                        "Feature": features,
                        "Importance": model.feature_importances_,
                    }
                ).sort_values("Importance", ascending=True)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(
                    feature_df["Feature"],
                    feature_df["Importance"],
                    color="blue",
                    edgecolor="black",
                )
                ax.set_title("Random Forest - Feature Importance")
                ax.set_xlabel("Importance Score")
                ax.set_ylabel("Soil and Weather Features")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("This model does not provide feature importance values.")

            # 3. Crop distribution
            st.markdown("### 3. Crop Distribution")
            crop_counts = df["label"].value_counts()
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(
                crop_counts.index.astype(str),
                crop_counts.values,
                color="lightgreen",
                edgecolor="black",
            )
            ax.set_title("Crop Distribution in Dataset")
            ax.set_xlabel("Crop Names")
            ax.set_ylabel("Number of Records")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

with technology_tab:
    st.markdown(
        """
        ## 💻 Technologies Used

        ### 1. Python
        Used as the main programming language.

        ### 2. Pandas
        Used to load and process the CSV dataset.

        ### 3. NumPy
        Used for numerical operations.

        ### 4. Scikit-learn
        Used for machine learning model development and evaluation.

        ### 5. Random Forest Classifier
        Used to predict suitable crops.

        ### 6. Streamlit
        Used to develop the interactive web application.

        ### 7. Google Colab
        Used as the development environment.

        ### 8. Joblib
        Used to save and load the trained model.

        ---

        ## Project Workflow

        1. Dataset collection
        2. Data cleaning and preprocessing
        3. Model training
        4. Model evaluation
        5. Crop prediction
        6. Application development
        """
    )

with about_tab:
    st.markdown(
        """
        ## ℹ️ About Our Project

        **Project Name:** Crop Recommendation & Soil Analyzer

        **Course:** Introduction to Artificial Intelligence

        **Machine Learning Algorithm:** Random Forest Classifier

        **Dataset:** Crop Recommendation Dataset

        **Application:** AI-based Crop Recommendation System

        ---

        ## Future Improvements

        - Add crop descriptions.
        - Display model accuracy.
        - Add feature importance graphs.
        - Provide additional agricultural information.
        - Improve soil analysis features.

        ---

        ## Our Goal

        To demonstrate how Artificial Intelligence can support
        agricultural decision-making.
        """
    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#183b20;'>"
    "**🌱 Crop Recommendation & Soil Analyzer**<br>"
    "First-Year Artificial Intelligence & Machine Learning Project"
    "</div>",
    unsafe_allow_html=True,
)
