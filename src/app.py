import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="AcneGuard",
    page_icon="🩺"
)

st.title("AcneGuard")
st.subheader(
    "AI Acne Severity Detection System"
)

uploaded_file = st.file_uploader(
    "Upload Acne Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict Severity"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file,
                uploaded_file.type
            )
        }

        response = requests.post(
            API_URL,
            files=files
        )

        result = response.json()

        st.success(
            f"Severity: {result['severity']}"
        )

        st.info(
            f"Confidence: {result['confidence']}%"
        )

        severity = result["severity"]

        if severity == "Grade 0":
            st.write(
                "Minimal acne detected."
            )

        elif severity == "Grade 1":
            st.write(
                "Mild acne detected."
            )

        elif severity == "Grade 2":
            st.write(
                "Moderate acne detected."
            )

        else:
            st.write(
                "Severe acne detected."
            )