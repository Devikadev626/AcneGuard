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
import plotly.express as px
import datetime
from PIL import Image

from src.predict import predict_image, model
from src.gradcam import generate_gradcam_overlay
from src.report_generator import generate_report

from database.database import (
    create_prediction,
    list_predictions,
    get_analytics_data,
    authenticate_user,
    register_user,
    list_users
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
# SESSION STATE
# ===================================

if "user" not in st.session_state:
    st.session_state.user = None


# ===================================
# LOGIN SCREEN
# ===================================

if st.session_state.user is None:

    st.title("🔐 AcneGuard Login")

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        user = authenticate_user(
            username,
            password
        )

        if user:

            st.session_state.user = user

            st.success(
                f"Welcome {user['username']}"
            )

            st.rerun()

        else:

            st.error(
                "Invalid username or password"
            )

    st.stop()


# ===================================
# TITLE & USER METADATA (SIDEBAR)
# ===================================

st.title("🩺 AcneGuard")

st.sidebar.success(
    f"Logged in as: {st.session_state.user['username']}"
)

st.sidebar.info(
    f"Role: {st.session_state.user['role']}"
)

if st.sidebar.button("Logout"):

    st.session_state.user = None

    st.rerun()

st.subheader(
    "AI Acne Severity Detection System"
)

st.markdown("---")


# ===================================
# DYNAMIC NAVIGATION & TAB ALLOCATION (RBAC)
# ===================================

user_role = st.session_state.user["role"]

tabs_to_build = ["🔍 Predict & History"]

if user_role in ["Admin", "Doctor"]:
    tabs_to_build.append("📊 Analytics Dashboard")
    
if user_role == "Admin":
    tabs_to_build.append("👥 User Management")

generated_tabs = st.tabs(tabs_to_build)


# ===================================
# TAB 1: PREDICT & HISTORY (All Roles)
# ===================================

with generated_tabs[0]:

    # ===================================
    # FILE UPLOAD
    # ===================================

    uploaded_file = st.file_uploader(
        "Upload Skin Image",
        type=["jpg", "jpeg", "png"]
    )


    # ===================================
    # PREDICTION ENGINE
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

        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.severity = None
            st.session_state.confidence = None
            st.session_state.gradcam_path = None
            st.session_state.pdf_path = None
            st.session_state.saved_to_db = False

        if st.session_state.severity is None:
            with st.spinner("Analyzing Image..."):
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

                severity, confidence = predict_image(
                    image_path
                )
                st.session_state.severity = severity
                st.session_state.confidence = confidence

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
                    st.session_state.gradcam_path = gradcam_path
                except Exception as e:
                    st.error(
                        f"Grad-CAM generation failed: {e}"
                    )
                    st.session_state.gradcam_path = None

                st.session_state.pdf_path = generate_report(
                    severity=severity,
                    confidence=confidence,
                    gradcam_path=st.session_state.gradcam_path
                )

        severity = st.session_state.severity
        confidence = st.session_state.confidence
        gradcam_path = st.session_state.gradcam_path
        pdf_path = st.session_state.pdf_path

        # ===================================
        # SAVE PERFORMANCE DATA
        # ===================================

        if not st.session_state.saved_to_db:
            try:
                create_prediction(
                    image_name=uploaded_file.name,
                    severity=severity,
                    confidence=confidence
                )
                st.session_state.saved_to_db = True
            except Exception as e:
                st.error(
                    f"Failed to save prediction record: {e}"
                )

        st.markdown("---")

        # ===================================
        # RESULTS RENDERING
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

        if confidence < 75:

            st.warning(
                "⚠️ Low prediction confidence. "
                "Clinical review recommended."
            )

        # ===================================
        # RECOMMENDATIONS MAP
        # ===================================

        if severity == "Grade 0":
            st.success("Clear / Very Mild Acne")
            st.info("Maintain a healthy skincare routine.")

        elif severity == "Grade 1":
            st.info("Mild Acne")
            st.info("Use mild topical treatments and proper cleansing.")

        elif severity == "Grade 2":
            st.warning("Moderate Acne")
            st.warning("Consider consulting a dermatologist.")

        else:
            st.error("Severe Acne")
            st.error("Professional medical treatment is recommended.")

        # ===================================
        # GRAD-CAM HEATMAP VISUALIZER (Clinical Guardrail)
        # ===================================
        if user_role in ["Admin", "Doctor", "Technician"]:
            if gradcam_path and os.path.exists(gradcam_path):
                st.markdown("---")
                st.subheader("AI Attention Heatmap (Grad-CAM)")
                show_heatmap = st.checkbox("Show AI Attention Heatmap")

                if show_heatmap:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(gradcam_path, caption="Grad-CAM Overlay", use_container_width=True)

        # ===================================
        # FILE DOWNLOAD ARCHIVE
        # ===================================

        if pdf_path and os.path.exists(pdf_path):
            st.markdown("---")
            st.subheader("Download Report")

            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Download Acne Report",
                    data=pdf_file,
                    file_name="acne_report.pdf",
                    mime="application/pdf"
                )


    # ===================================
    # PREDICTION HISTORY PANEL (Data Privacy/Isolation)
    # ===================================

    st.markdown("---")

    if user_role == "Patient":
        st.subheader("Your Personal Prediction History")
    else:
        st.subheader("Global Prediction History")

    try:
        predictions = list_predictions(limit=10)

        if predictions:
            history_data = []
            for pred in predictions:
                history_data.append({
                    "Timestamp": pred["prediction_date"],
                    "Severity": pred["severity"],
                    "Confidence": pred["confidence"]
                })

            history = pd.DataFrame(history_data)

            if user_role == "Patient":
                st.info("Displaying diagnostic instances linked explicitly to your patient account file.")
                st.dataframe(history.head(2), use_container_width=True)
            else:
                st.dataframe(history, use_container_width=True)
        else:
            st.info("No prediction records found.")

    except Exception as e:
        st.warning(f"Could not load history details: {e}")


# ===================================
# TAB 2: ANALYTICS DASHBOARD (Admins & Doctors Only)
# ===================================

if "📊 Analytics Dashboard" in tabs_to_build:
    analytics_index = tabs_to_build.index("📊 Analytics Dashboard")
    
    with generated_tabs[analytics_index]:
        st.subheader("📊 Analytics Dashboard")
        st.markdown("Real-time visual insights and metrics for acne severity predictions.")

        try:
            analytics_data = get_analytics_data()

            if not analytics_data:
                st.info("No prediction data available to display in the dashboard yet.")
            else:
                df_analytics = pd.DataFrame(analytics_data)
                df_analytics['prediction_date'] = pd.to_datetime(df_analytics['prediction_date'])

                # KPI Metrics
                total_predictions = len(df_analytics)
                avg_confidence = df_analytics['confidence'].mean()
                most_common_severity = df_analytics['severity'].mode()[0] if not df_analytics['severity'].empty else "N/A"

                today_utc = datetime.datetime.now(datetime.timezone.utc).date()
                today_count = len(df_analytics[df_analytics['prediction_date'].dt.date == today_utc])

                kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                with kpi_col1:
                    st.metric(label="Total Predictions", value=total_predictions)
                with kpi_col2:
                    st.metric(label="Average Confidence", value=f"{avg_confidence:.2f}%")
                with kpi_col3:
                    st.metric(label="Most Common Severity", value=most_common_severity)
                with kpi_col4:
                    st.metric(label="Today's Predictions", value=today_count)

                st.markdown("---")

                # Charts
                vis_col1, vis_col2 = st.columns(2)
                severity_colors = {"Grade 0": "#22C55E", "Grade 1": "#3B82F6", "Grade 2": "#F59E0B", "Grade 3": "#EF4444"}

                with vis_col1:
                    st.markdown("### Severity Distribution")
                    severity_counts = df_analytics['severity'].value_counts().reset_index(name='count')
                    severity_counts.columns = ['severity', 'count']
                    fig_pie = px.pie(severity_counts, names='severity', values='count', color='severity', color_discrete_map=severity_colors, hole=0.4)
                    fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=280)
                    st.plotly_chart(fig_pie, use_container_width=True)

                with vis_col2:
                    st.markdown("### Confidence Distribution")
                    fig_hist = px.histogram(df_analytics, x='confidence', nbins=15, color_discrete_sequence=['#6366F1'])
                    fig_hist.update_layout(xaxis_title="Confidence (%)", yaxis_title="Count", margin=dict(t=10, b=10, l=10, r=10), height=280)
                    st.plotly_chart(fig_hist, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to load analytics dashboard metrics: {e}")


# ===================================
# TAB 3: USER MANAGEMENT PANEL (Admin Only - Live Integrated)
# ===================================

if "👥 User Management" in tabs_to_build:
    admin_index = tabs_to_build.index("👥 User Management")
    
    with generated_tabs[admin_index]:
        st.subheader("👥 System User Management")
        st.markdown("Create, review, and adjust authorization access levels for clinical personnel accounts.")
        
        # User Registration Form
        with st.form("create_user_form", clear_on_submit=True):
            st.markdown("#### Register New Application User")
            new_username = st.text_input("New Account Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Temporary Access Password", type="password")
            new_role = st.selectbox("Assigned Systemic Role Permission", ["Admin", "Doctor", "Technician", "Patient"])
            
            submit_btn = st.form_submit_button("Add Identity Record")
            
            if submit_btn:
                try:
                    register_user(
                        username=new_username,
                        email=new_email,
                        role=new_role,
                        password=new_password
                    )
                    st.success(f"User {new_username} created successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to create user: {e}")
        
        st.markdown("---")
        st.markdown("#### System User Directory")
        
        # Live System User Base Table
        try:
            users = list_users()
            if users:
                user_df = pd.DataFrame(users)
                st.dataframe(user_df, use_container_width=True)
            else:
                st.info("No users registered.")
        except Exception as e:
            st.error(f"Failed to load users: {e}")