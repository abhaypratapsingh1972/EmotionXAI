import streamlit as st
from PIL import Image
import pandas as pd
import time

from utils.predict import (
    predict_emotion,
    preprocess_image,
    model,
    class_names
)

from utils.explain import EXPLANATIONS
from utils.gradcam_utils import generate_gradcam


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="EmotionXAI",
    page_icon="😊",
    layout="wide"
)

st.title("😊 EmotionXAI")

st.subheader(
    "Explainable Facial Emotion Recognition using EfficientNet-B0 and Grad-CAM"
)

st.markdown("---")


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    # --------------------------------------------------
    # INFERENCE TIMER
    # --------------------------------------------------

    start_time = time.time()

    emotion, confidence, top_probs, top_indices = predict_emotion(image)

    end_time = time.time()

    inference_time = (end_time - start_time) * 1000

    # --------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------

    input_tensor = preprocess_image(image)

    heatmap = generate_gradcam(
        model,
        input_tensor,
        image
    )

    # --------------------------------------------------
    # IMAGE + HEATMAP SIDE BY SIDE
    # --------------------------------------------------

    st.header("Image Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            heatmap,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    st.markdown("---")

    st.header("Prediction")

    st.success(
        f"Emotion: {emotion.upper()}"
    )

    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )

    st.metric(
        "Inference Time",
        f"{inference_time:.2f} ms"
    )

    # --------------------------------------------------
    # TOP 3 PREDICTIONS
    # --------------------------------------------------

    st.markdown("---")

    st.header("Top 3 Predictions")

    top3_data = []

    for prob, idx in zip(top_probs, top_indices):

        label = class_names[idx.item()]
        probability = prob.item()

        top3_data.append(
            {
                "Emotion": label,
                "Probability": probability
            }
        )

        st.write(
            f"**{label.capitalize()}** : {probability * 100:.2f}%"
        )

        st.progress(float(probability))

    # --------------------------------------------------
    # BAR CHART
    # --------------------------------------------------

    st.markdown("---")

    st.header("Probability Distribution")

    prob_df = pd.DataFrame(top3_data)

    st.bar_chart(
        prob_df.set_index("Emotion")
    )

    # --------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------

    st.markdown("---")

    st.header("Explanation")

    st.info(
        EXPLANATIONS[emotion]
    )

    # --------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------

    st.markdown("---")

    report_text = f"""
EmotionXAI Prediction Report

Predicted Emotion: {emotion}
Confidence: {confidence * 100:.2f}%

Top Predictions:
"""

    for prob, idx in zip(top_probs, top_indices):

        label = class_names[idx.item()]

        report_text += (
            f"\n- {label}: {prob.item() * 100:.2f}%"
        )

    report_text += f"""

Explanation:
{EXPLANATIONS[emotion]}

Inference Time:
{inference_time:.2f} ms
"""

    st.download_button(
        label="📄 Download Prediction Report",
        data=report_text,
        file_name="emotion_prediction_report.txt",
        mime="text/plain"
    )