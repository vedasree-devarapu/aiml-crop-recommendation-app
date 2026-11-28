import gradio as gr
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# =========================================
# 1. LOAD TRAINED RANDOM FOREST MODEL
# =========================================

model = joblib.load("crop_model.pkl")

# Automatically detect model features
features = list(model.feature_names_in_)

feature_labels = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "ph": "Soil pH",
    "rainfall": "Rainfall (mm)"
}

default_values = {
    "N": 50,
    "P": 50,
    "K": 50,
    "temperature": 25,
    "humidity": 70,
    "ph": 6.5,
    "rainfall": 100
}

# =========================================
# 2. CROP PREDICTION FUNCTION
# =========================================

def recommend_crop(*values):
    input_data = pd.DataFrame(
        [values],
        columns=features
    )
    prediction = model.predict(input_data)
    crop = str(prediction[0]).upper()
    return f"""
    🌾 YOUR RECOMMENDED CROP

    # {crop}

    Prediction generated using the trained
    Random Forest Machine Learning model.
    """

# GRAPH 1: CONFUSION MATRIX
def create_confusion_matrix():
    y_pred = model.predict(x2)
    cm = confusion_matrix(
        y2,
        y_pred,
        labels=model.classes_
    )
    fig, ax = plt.subplots(figsize=(12, 9))
    image = ax.imshow(cm, cmap="Reds")
    ax.set_title("Random Forest - Confusion Matrix")
    ax.set_xlabel("Predicted Crop")
    ax.set_ylabel("Actual Crop")
    ax.set_xticks(range(len(model.classes_)))
    ax.set_yticks(range(len(model.classes_)))
    ax.set_xticklabels(model.classes_, rotation=90)
    ax.set_yticklabels(model.classes_)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    plt.close(fig)
    return fig

# GRAPH 2: FEATURE IMPORTANCE
def create_feature_importance():
    importance = model.feature_importances_
    feature_names = x.columns
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        feature_names,
        importance,
        color="blue",
        edgecolor="black"
    )
    ax.set_title("Random Forest - Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Soil and Weather Features")
    fig.tight_layout()
    plt.close(fig)
    return fig

# GRAPH 3: CROP DISTRIBUTION
def create_crop_distribution():
    crop_counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        crop_counts.index,
        crop_counts.values,
        color="green",
        edgecolor="black"
    )
    ax.set_title("Crop Distribution in Dataset")
    ax.set_xlabel("Crop Names")
    ax.set_ylabel("Number of Records")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    plt.close(fig)
    return fig

# =========================================
# 3. APPLICATION DESIGN
# =========================================

custom_css = """
.gradio-container {
    background-color: #e8f5e9 !important;
    background-image:
        linear-gradient(
            rgba(240, 255, 240, 0.78),
            rgba(240, 255, 240, 0.78)
        ),
        url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=1920");
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    max-width: 1200px !important;
    margin: auto !important;
}
#main-heading {
    background: linear-gradient(
        120deg,
        #07552b,
        #16834a
    ) !important;
    padding: 30px !important;
    border-radius: 18px !important;
    text-align: center !important;
    margin-bottom: 20px !important;
}
#main-heading h1,
#main-heading h2,
#main-heading p {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.gradio-container .tab-nav {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 5px !important;
    overflow-x: auto !important;
    background: #dceddc !important;
    border-radius: 12px !important;
    padding: 6px !important;
}
.gradio-container .tab-nav button,
.gradio-container .tab-nav button *,
.gradio-container button[role="tab"],
.gradio-container button[role="tab"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    background-color: #e0f2e1 !important;
    white-space: nowrap !important;
}
.gradio-container .tab-nav button.selected,
.gradio-container button[role="tab"][aria-selected="true"] {
    background-color: #a5d6a7 !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}
.gradio-container .tab-nav button.selected *,
.gradio-container button[role="tab"][aria-selected="true"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}
.gradio-container .tab-nav button:hover {
    background-color: #c8e6c9 !important;
    color: #000000 !important;
}
.content-panel {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    border: 1px solid #b7d8b8 !important;
}
.prose,
.prose p,
.prose h1,
.prose h2,
.prose h3,
.prose h4,
.prose li,
.prose strong,
.prose span,
label {
    color: #183b20 !important;
    -webkit-text-fill-color: #183b20 !important;
    opacity: 1 !important;
}
.gradio-container label {
    color: #183b20 !important;
    font-weight: 600 !important;
}
.gradio-container input {
    background-color: #ffffff !important;
    color: #000000 !important;
}
.gradio-container button.primary {
    background-color: #16834a !important;
    color: #ffffff !important;
    font-weight: bold !important;
    border-radius: 10px !important;
}
footer {
    color: #183b20 !important;
}
.gradio-container {
    width: 100% !important;
    max-width: 100% !important;
    min-height: 100vh !important;
    margin: 0 !important;
    padding: 20px !important;
    box-sizing: border-box !important;
}
"""

# =========================================
# 4. CREATE GRADIO APPLICATION
# =========================================

with gr.Blocks(
    title="Crop Recommendation & Soil Analyzer",
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="lime"
    ),
    css=custom_css
) as app:

    # MAIN HEADER
    gr.Markdown("""
    <div id="main-heading">
    <h1>🌱 CROP RECOMMENDATION & SOIL ANALYZER</h1>
    <h2>Smart Agriculture Using Artificial Intelligence</h2>
    <p>Making smarter farming decisions with Machine Learning</p>
    </div>
    """)

    # =====================================
    # HOME TAB
    # =====================================
    with gr.Tab("Home"):
        with gr.Column(elem_classes=["content-panel"]):
            gr.Markdown("""
            # 🌿 Welcome, Farmer!

            Welcome to our Crop Recommendation & Soil Analyzer.

            Our application uses Artificial Intelligence and
            Random Forest to recommend suitable crops based
            on soil nutrients and environmental conditions.

            ---

            ## 🎯 About Our Project

            Agriculture is an important part of our lives.

            Choosing a suitable crop according to soil and
            weather conditions can help farmers make
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

            To develop an AI-based application that analyzes
            soil and weather information and recommends a crop.
            """)

    # =====================================
    # RECOMMEND CROP TAB
    # =====================================
    with gr.Tab("Recommend Crop"):
        with gr.Column(elem_classes=["content-panel"]):
            gr.Markdown("""
            # 🌾 Crop Recommendation

            Enter the soil and weather values below.

            Our trained Random Forest model will predict
            a suitable crop based on your inputs.
            """)

            input_components = []
            with gr.Row():
                for feature in features:
                    component = gr.Number(
                        label=feature_labels.get(feature, feature),
                        value=default_values.get(feature, 0),
                        precision=2
                    )
                    input_components.append(component)

            predict_button = gr.Button(
                "🌱 Recommend Crop",
                variant="primary"
            )

            result = gr.Markdown(
                "Your crop recommendation will appear here."
            )

            predict_button.click(
                fn=recommend_crop,
                inputs=input_components,
                outputs=result
            )

    # =====================================
    # MODEL ANALYSIS TAB
    # =====================================
    with gr.Tab("Model Analysis"):
        with gr.Column(elem_classes=["content-panel"]):
            gr.Markdown("""
            # Model Analysis
            ### Visual evaluation of our Random Forest model
            """)
            gr.Markdown("## 1. Confusion Matrix")
            gr.Plot(value=create_confusion_matrix)

            gr.Markdown("## 2. Feature Importance")
            gr.Plot(value=create_feature_importance)

            gr.Markdown("## 3. Crop Distribution")
            gr.Plot(value=create_crop_distribution)

    # =====================================
    # TECHNOLOGY USED TAB
    # =====================================
    with gr.Tab("Technology Used"):
        with gr.Column(elem_classes=["content-panel"]):
            gr.Markdown("""
            # 💻 Technologies Used

            ### 1. Python
            Used as the main programming language.

            ### 2. Pandas
            Used to load and process the CSV dataset.

            ### 3. NumPy
            Used for numerical operations.

            ### 4. Scikit-learn
            Used for machine learning model development.

            ### 5. Random Forest Classifier
            Used to predict suitable crops.

            ### 6. Gradio
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
            """)

    # =====================================
    # ABOUT TAB
    # =====================================
    with gr.Tab("About"):
        with gr.Column(elem_classes=["content-panel"]):
            gr.Markdown("""
            # ℹ️ About Our Project

            **Project Name:**
            Crop Recommendation & Soil Analyzer

            **Course:**
            Introduction to Artificial Intelligence

            **Machine Learning Algorithm:**
            Random Forest Classifier

            **Dataset:**
            Crop Recommendation Dataset

            **Application:**
            AI-based Crop Recommendation System

            ---

            ## Future Improvements

            - Add crop descriptions.
            - Display model accuracy.
            - Add feature importance graphs.
            - Provide additional agricultural information.
            - Improve soil analysis features.

            ---

            ## Our Goal

            To demonstrate how Artificial Intelligence
            can support agricultural decision-making.
            """)

    # =====================================
    # FOOTER
    # =====================================
    gr.Markdown("""
    ---
    <center>
    **🌱 Crop Recommendation & Soil Analyzer**
    First-Year Artificial Intelligence & Machine Learning Project
    </center>
    """)

# =========================================
# 5. LAUNCH APPLICATION
# =========================================
app.launch(
    share=True,
    inbrowser=False,
    show_error=True
)
