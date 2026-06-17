from fpdf import FPDF
from datetime import datetime
import os


def generate_report(
    severity,
    confidence,
    gradcam_path=None,
    output_path="reports/acne_report.pdf"
):

    os.makedirs(
        "reports",
        exist_ok=True
    )

    pdf = FPDF()

    pdf.add_page()

    # ==========================
    # TITLE
    # ==========================

    pdf.set_font(
        "Arial",
        "B",
        16
    )

    pdf.cell(
        0,
        10,
        "AcneGuard AI Report",
        ln=True,
        align="C"
    )

    pdf.ln(10)

    # ==========================
    # REPORT DETAILS
    # ==========================

    pdf.set_font(
        "Arial",
        "",
        12
    )

    pdf.cell(
        0,
        10,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ln=True
    )

    pdf.cell(
        0,
        10,
        f"Severity: {severity}",
        ln=True
    )

    pdf.cell(
        0,
        10,
        f"Confidence: {confidence}%",
        ln=True
    )

    pdf.ln(10)

    # ==========================
    # RECOMMENDATIONS
    # ==========================

    recommendation = {

        "Grade 0":
        "Maintain healthy skincare.",

        "Grade 1":
        "Use mild acne treatment.",

        "Grade 2":
        "Consult dermatologist if needed.",

        "Grade 3":
        "Professional medical treatment recommended."
    }

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        0,
        10,
        "Recommendation",
        ln=True
    )

    pdf.set_font(
        "Arial",
        "",
        12
    )

    pdf.multi_cell(
        0,
        10,
        recommendation[severity]
    )

    pdf.ln(5)

    # ==========================
    # LOW CONFIDENCE WARNING
    # ==========================

    if confidence < 75:

        pdf.set_font(
            "Arial",
            "B",
            12
        )

        pdf.cell(
            0,
            10,
            "WARNING",
            ln=True
        )

        pdf.set_font(
            "Arial",
            "",
            12
        )

        pdf.multi_cell(
            0,
            10,
            f"This prediction has a confidence score of {confidence}%. "
            "Clinical review by a dermatologist is recommended."
        )

        pdf.ln(5)

    # ==========================
    # GRAD-CAM SECTION
    # ==========================

    if gradcam_path and os.path.exists(
        gradcam_path
    ):

        pdf.set_font(
            "Arial",
            "B",
            12
        )

        pdf.cell(
            0,
            10,
            "AI Explainability (Grad-CAM)",
            ln=True
        )

        pdf.image(
            gradcam_path,
            x=20,
            w=160
        )

        pdf.ln(90)

        pdf.set_font(
            "Arial",
            "",
            11
        )

        pdf.multi_cell(
            0,
            8,
            "This Grad-CAM visualization highlights the image regions "
            "that most influenced the AI prediction."
        )

    pdf.output(
        output_path
    )

    return output_path