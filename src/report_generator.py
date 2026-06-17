
from fpdf import FPDF

from datetime import datetime


def generate_report(
    severity,
    confidence,
    output_path="reports/acne_report.pdf"
):

    pdf = FPDF()

    pdf.add_page()

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

    pdf.set_font(
        "Arial",
        "",
        12
    )

    pdf.cell(
        0,
        10,
        f"Date: {datetime.now()}",
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

    pdf.multi_cell(
        0,
        10,
        f"Recommendation:\n\n{recommendation[severity]}"
    )

    pdf.output(
        output_path
    )

    return output_path