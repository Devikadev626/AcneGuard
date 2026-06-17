
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

from datetime import datetime
from PIL import Image

from src.predict import predict_image
from src.report_generator import generate_report


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

    image_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    image.save(
        image_path
    )

    # ===================================
    # PREDICTION
    # ===================================

    severity, confidence = predict_image(
        image_path
    )

    # ===================================
    # GENERATE PDF REPORT
    # ===================================

    pdf_path = generate_report(
        severity=severity,
        confidence=confidence
    )

    # ===================================
    # SAVE HISTORY
    # ===================================

    os.makedirs(
        "reports",
        exist_ok=True
    )

    history_file = "reports/history.csv"

    if not os.path.exists(
        history_file
    ):

        pd.DataFrame(
            columns=[
                "Timestamp",
                "Severity",
                "Confidence"
            ]
        ).to_csv(
            history_file,
            index=False
        )

    new_record = pd.DataFrame({

        "Timestamp": [
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ],

        "Severity": [
            severity
        ],

        "Confidence": [
            confidence
        ]
    })

    new_record.to_csv(
        history_file,
        mode="a",
        header=False,
        index=False
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
    # DOWNLOAD PDF REPORT
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

history_file = "reports/history.csv"

try:

    if os.path.exists(
        history_file
    ):

        history = pd.read_csv(
            history_file,
            encoding="utf-8-sig"
        )

        st.dataframe(
            history.tail(10),
            use_container_width=True
        )

    else:

        st.info(
            "No prediction history available."
        )

except Exception as e:

    st.warning(
        f"Could not load history file: {e}"
    )
