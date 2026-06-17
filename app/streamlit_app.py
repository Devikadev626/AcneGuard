import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import streamlit as st
import pandas as pd

from PIL import Image

from src.predict import predict_image, model
from src.gradcam import generate_gradcam_overlay
from src.report_generator import generate_report

from database.database import (
    create_prediction,
    list_predictions
)


# ===================================
# PAGE CONFIG
# ===================================

st.set_page_config(
    page_title="AcneGuard",
    page_icon="🩺",
    layout="centered"
)


# ===================================
# TITLE
# ===================================

st.title("🩺 AcneGuard")

st.subheader(
    "AI Acne Severity Detection System"
)

st.markdown("---")


# ===================================
# FILE UPLOAD
# ===================================

uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)


# ===================================
# PREDICTION
# ===================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # ===================================
    # SAVE IMAGE
    # ===================================

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    os.makedirs(
        "gradcam/temp",
        exist_ok=True
    )

    image_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    image.save(
        image_path
    )

    # ===================================
    # MODEL PREDICTION
    # ===================================

    severity, confidence = predict_image(
        image_path
    )

    # ===================================
    # GENERATE GRAD-CAM
    # ===================================

    gradcam_path = os.path.join(
        "gradcam",
        "temp",
        f"gradcam_{uploaded_file.name}"
    )

    try:

        generate_gradcam_overlay(
            model=model,
            target_layer=model.layer4,
            image_path=image_path,
            output_path=gradcam_path
        )

    except Exception as e:

        st.error(
            f"Grad-CAM generation failed: {e}"
        )

        gradcam_path = None

    # ===================================
    # PDF REPORT
    # ===================================

    pdf_path = generate_report(
    severity=severity,
    confidence=confidence,
    gradcam_path=gradcam_path
)
    # ===================================
    # SAVE TO SQLITE
    # ===================================

    try:

        create_prediction(
            image_name=uploaded_file.name,
            severity=severity,
            confidence=confidence
        )

    except Exception as e:

        st.error(
            f"Failed to save prediction record: {e}"
        )

    st.markdown("---")

    # ===================================
    # RESULTS
    # ===================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Severity",
            severity
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence}%"
        )

    # ===================================
    # LOW CONFIDENCE WARNING
    # ===================================

    if confidence < 75:

        st.warning(
            "⚠️ Low prediction confidence. "
            "Clinical review recommended."
        )

    # ===================================
    # RECOMMENDATIONS
    # ===================================

    if severity == "Grade 0":

        st.success(
            "Clear / Very Mild Acne"
        )

        recommendation = (
            "Maintain a healthy skincare routine."
        )

        st.info(
            recommendation
        )

    elif severity == "Grade 1":

        st.info(
            "Mild Acne"
        )

        recommendation = (
            "Use mild topical treatments and proper cleansing."
        )

        st.info(
            recommendation
        )

    elif severity == "Grade 2":

        st.warning(
            "Moderate Acne"
        )

        recommendation = (
            "Consider consulting a dermatologist."
        )

        st.warning(
            recommendation
        )

    else:

        st.error(
            "Severe Acne"
        )

        recommendation = (
            "Professional medical treatment is recommended."
        )

        st.error(
            recommendation
        )

    # ===================================
    # GRAD-CAM DISPLAY
    # ===================================

    if gradcam_path and os.path.exists(
        gradcam_path
    ):

        st.markdown("---")

        st.subheader(
            "AI Attention Heatmap (Grad-CAM)"
        )

        show_heatmap = st.checkbox(
            "Show AI Attention Heatmap"
        )

        if show_heatmap:

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    image,
                    caption="Original Image",
                    use_container_width=True
                )

            with col2:

                st.image(
                    gradcam_path,
                    caption="Grad-CAM Overlay",
                    use_container_width=True
                )

    # ===================================
    # DOWNLOAD REPORT
    # ===================================

    st.markdown("---")

    st.subheader(
        "Download Report"
    )

    with open(
        pdf_path,
        "rb"
    ) as pdf_file:

        st.download_button(
            label="📄 Download Acne Report",
            data=pdf_file,
            file_name="acne_report.pdf",
            mime="application/pdf"
        )


# ===================================
# PREDICTION HISTORY
# ===================================

st.markdown("---")

st.subheader(
    "Prediction History"
)

try:

    predictions = list_predictions(
        limit=10
    )

    if predictions:

        history_data = []

        for pred in predictions:

            history_data.append({

                "Timestamp":
                pred["prediction_date"],

                "Severity":
                pred["severity"],

                "Confidence":
                pred["confidence"]

            })

        history = pd.DataFrame(
            history_data
        )

        st.dataframe(
            history,
            use_container_width=True
        )

    else:

        st.info(
            "No prediction history available."
        )

except Exception as e:

    st.warning(
        f"Could not load history: {e}"
    )