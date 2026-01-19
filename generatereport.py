import os
import io
import base64
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, Flowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

# Helper to ensure reports folder exists
def _ensure_reports_folder(folder="reports"):
    os.makedirs(folder, exist_ok=True)
    return folder

# Simple horizontal rule flowable
class HRLine(Flowable):
    def __init__(self, width=450, thickness=1, color=colors.HexColor("#d0d7de")):
        Flowable.__init__(self)
        self.width = width
        self.thickness = thickness
        self.color = color
    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)

def _save_base64_image(base64_str: str, out_path: str) -> str:
    """
    Decode a base64 image string and save to out_path.
    Accepts data URLs (data:image/...) or raw base64.
    Returns the saved path.
    """
    if not base64_str:
        return None
    # strip data URL prefix if present
    if base64_str.startswith("data:"):
        base64_str = base64_str.split(",", 1)[1]
    try:
        img_data = base64.b64decode(base64_str)
    except Exception:
        return None
    with open(out_path, "wb") as f:
        f.write(img_data)
    return out_path

def generate_xray_report(
    hospital_name: str,
    patient_info: dict,
    disease_name: str,
    xray_image_path: str = None,
    xray_image_base64: str = None,
    doctor_info: dict = None,
    report_filename: str = None,
    reports_folder: str = "reports"
) -> str:
    """
    Generate a styled X-ray medical report PDF and return the file path.

    Parameters
    - hospital_name: str
    - patient_info: dict with keys 'name', 'age', 'phone', 'email', optionally 'id', 'gender', 'findings', 'impression'
    - disease_name: str
    - xray_image_path: optional path to an image file
    - xray_image_base64: optional base64-encoded image string (data URL or raw base64)
    - doctor_info: dict with keys 'name' and 'email' (optional)
    - report_filename: optional filename; if None, auto-generated
    - reports_folder: folder to save reports (default 'reports')

    Returns
    - full path to the generated PDF file (str)
    """
    # Prepare folder and filename
    folder = _ensure_reports_folder(reports_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = (patient_info.get("name") or "patient").strip().replace(" ", "_")
    if not report_filename:
        report_filename = f"{safe_name}_xray_report_{timestamp}.pdf"
    pdf_path = os.path.join(folder, report_filename)

    # If base64 provided, decode and save to a temporary image file in reports folder
    temp_image_path = None
    if xray_image_base64:
        ext = ".jpg"
        temp_image_path = os.path.join(folder, f"{safe_name}_xray_{timestamp}{ext}")
        saved = _save_base64_image(xray_image_base64, temp_image_path)
        if not saved:
            temp_image_path = None

    # Prefer explicit file path if exists, else use decoded base64 image
    final_image_path = None
    if xray_image_path and os.path.exists(xray_image_path):
        final_image_path = xray_image_path
    elif temp_image_path and os.path.exists(temp_image_path):
        final_image_path = temp_image_path

    # Register a nicer font if available
    try:
        pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))
        base_font = "DejaVuSans"
    except Exception:
        base_font = "Helvetica"

    # Document setup
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=18*mm,
        bottomMargin=18*mm
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="HospitalHeader", fontName=base_font, fontSize=20, leading=22, alignment=TA_CENTER, textColor=colors.HexColor("#0b3d91")))
    styles.add(ParagraphStyle(name="HospitalSub", fontName=base_font, fontSize=10, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#6b7280")))
    styles.add(ParagraphStyle(name="PatientLabel", fontName=base_font, fontSize=10, leading=12, textColor=colors.HexColor("#374151")))
    styles.add(ParagraphStyle(name="DiseaseHighlight", fontName=base_font, fontSize=14, leading=16, alignment=TA_CENTER, textColor=colors.white))
    styles.add(ParagraphStyle(name="FooterSmall", fontName=base_font, fontSize=9, leading=11, alignment=TA_RIGHT, textColor=colors.HexColor("#6b7280")))

    story = []

    # Header
    story.append(Paragraph(hospital_name, styles["HospitalHeader"]))
    story.append(Paragraph("Radiology Department", styles["HospitalSub"]))
    story.append(Spacer(1, 6))
    story.append(HRLine(width=160*mm, thickness=1.5, color=colors.HexColor("#e6eef8")))
    story.append(Spacer(1, 8))

    # Patient info table
    left_col = [
        ("Patient Name", patient_info.get("name", "")),
        ("Age", str(patient_info.get("age", ""))),
        ("Gender", patient_info.get("gender", "")),
    ]
    right_col = [
        ("Patient ID", patient_info.get("id", "")),
        ("Phone", patient_info.get("phone", "")),
        ("Email", patient_info.get("email", "")),
    ]
    patient_table_data = []
    for (lkey, lval), (rkey, rval) in zip(left_col, right_col):
        patient_table_data.append([
            Paragraph(f"<b>{lkey}</b>", styles["PatientLabel"]),
            Paragraph(lval or "-", styles["PatientLabel"]),
            Paragraph(f"<b>{rkey}</b>", styles["PatientLabel"]),
            Paragraph(rval or "-", styles["PatientLabel"])
        ])
    table = Table(patient_table_data, colWidths=[35*mm, 55*mm, 35*mm, 55*mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1, -1), colors.white),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#e6eef8")),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#f1f5f9")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Disease highlight
    disease_box_width = 160*mm
    disease_bg_color = colors.HexColor("#ef4444")
    disease_table = Table(
        [[Paragraph(f"<b>Diagnosis: {disease_name}</b>", styles["DiseaseHighlight"])]],
        colWidths=[disease_box_width]
    )
    disease_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), disease_bg_color),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(disease_table)
    story.append(Spacer(1, 14))

    # X-ray image insertion
    image_added = False
    if final_image_path:
        try:
            with PILImage.open(final_image_path) as pil_img:
                img_width, img_height = pil_img.size
                aspect = img_height / float(img_width)
                
                # Define constraints
                max_width = 160 * mm
                max_height = 100 * mm  # Prevents image from taking over the whole page
                
                # Calculate display dimensions
                display_width = max_width
                display_height = display_width * aspect
                
                # If height is too large, scale down based on height instead
                if display_height > max_height:
                    display_height = max_height
                    display_width = display_height / aspect

                pil_img_rgb = pil_img.convert("RGB")
                bio = io.BytesIO()
                pil_img_rgb.save(bio, format="JPEG")
                bio.seek(0)
                
                img = Image(bio, width=display_width, height=display_height)
                img.hAlign = "CENTER"
                story.append(img)
                image_added = True
        except Exception as e:
            print(f"Error processing image: {e}")
            image_added = False

    if not image_added:
        story.append(Paragraph("X-ray image: <i>Not available</i>", styles["PatientLabel"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Figure: Frontal chest X-ray", styles["HospitalSub"]))
    story.append(Spacer(1, 18))

    # Findings and Impression
    heading_style = ParagraphStyle("SectionHeading", parent=styles["PatientLabel"], fontSize=12, leading=14, textColor=colors.HexColor("#0b3d91"))
    normal_style = styles["PatientLabel"]
    story.append(Paragraph("<b>Findings</b>", heading_style))
    story.append(Spacer(1, 4))
    findings_text = patient_info.get("findings", "No additional findings provided.")
    story.append(Paragraph(findings_text, normal_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Impression</b>", heading_style))
    story.append(Spacer(1, 4))
    impression_text = patient_info.get("impression", f"Suggest correlation with clinical history for {disease_name}.")
    story.append(Paragraph(impression_text, normal_style))
    story.append(Spacer(1, 24))

    # Footer with doctor info and timestamp
    doc_name = (doctor_info.get("name") if doctor_info else "Dr. __________________")
    doc_email = (doctor_info.get("email") if doctor_info else "")
    footer_table = Table(
        [
            [Paragraph(f"<b>Reporting Physician</b>", styles["PatientLabel"]), "", Paragraph(f"<b>Report generated</b>", styles["PatientLabel"])],
            [Paragraph(doc_name, styles["PatientLabel"]), "", Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["PatientLabel"])],
            [Paragraph(doc_email, styles["PatientLabel"]), "", Paragraph("", styles["PatientLabel"])]
        ],
        colWidths=[80*mm, 10*mm, 60*mm]
    )
    footer_table.setStyle(TableStyle([
        ("LINEBEFORE", (0,0), (0,-1), 0.25, colors.HexColor("#e6eef8")),
        ("LINEAFTER", (2,0), (2,-1), 0.25, colors.HexColor("#e6eef8")),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))
    story.append(footer_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph("This is a computer-generated report and does not replace clinical judgment.", styles["HospitalSub"]))

    # Build PDF
    doc.build(story)

    # Optionally remove temporary decoded image to keep folder tidy
    if temp_image_path and os.path.exists(temp_image_path):
        try:
            os.remove(temp_image_path)
        except Exception:
            pass

    return pdf_path

# Example usage
if __name__ == "__main__":
    # Example base64: small white PNG placeholder (truncated for brevity)
    # In real use, provide a full base64 string (data URL or raw base64)
    sample_base64 = None

    patient = {
        "name": "John Doe",
        "age": 45,
        "gender": "Male",
        "id": "P-2026-0001",
        "phone": "+91-9876543210",
        "email": "johndoe@example.com",
        "findings": "Cardiomediastinal silhouette within normal limits. Patchy airspace consolidation in the right lower zone.",
        "impression": "Right lower lobe consolidation consistent with pneumonia."
    }
    doctor = {"name": "Dr. Anita Menon", "email": "anita.menon@cityhospital.org"}

    pdf = generate_xray_report(
        hospital_name="University College Of Engineering",
        patient_info=patient,
        disease_name="Pneumonia",
        xray_image_path="explanation_output.png",           # or provide a file path
        xray_image_base64=sample_base64, # or provide a base64 string
        doctor_info=doctor
    )
    print("Report saved to:", pdf)
