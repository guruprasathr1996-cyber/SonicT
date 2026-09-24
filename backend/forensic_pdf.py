from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


def _safe(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _paragraph(text, style):
    return Paragraph(
        escape(_safe(text)).replace("\n", "<br/>"),
        style
    )


def _dict_table(data, styles, widths=None):
    rows = []

    if not isinstance(data, dict) or not data:
        return _paragraph("No data available.", styles["BodyText"])

    for key, value in data.items():
        if isinstance(value, (dict, list)):
            continue

        label = str(key).replace("_", " ").title()

        rows.append([
            _paragraph(label, styles["BodyText"]),
            _paragraph(value, styles["BodyText"]),
        ])

    if not rows:
        return _paragraph("No simple fields available.", styles["BodyText"])

    table = Table(
        rows,
        colWidths=widths or [55 * mm, 120 * mm],
        hAlign="LEFT"
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F6F9")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#172033")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DCE5EE")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    return table


def _nested_evidence_table(data, styles):
    if not isinstance(data, dict) or not data:
        return _paragraph("No forensic evidence available.", styles["BodyText"])

    rows = [[
        _paragraph("Evidence", styles["TableHeader"]),
        _paragraph("Value", styles["TableHeader"]),
    ]]

    for key, value in data.items():
        label = str(key).replace("_", " ").title()

        if isinstance(value, dict):
            value_text = ", ".join(
                f"{str(k).replace('_', ' ').title()}: {_safe(v)}"
                for k, v in value.items()
                if not isinstance(v, (dict, list))
            )
        elif isinstance(value, list):
            value_text = f"{len(value)} item(s)"
        else:
            value_text = _safe(value)

        rows.append([
            _paragraph(label, styles["BodyText"]),
            _paragraph(value_text or "-", styles["BodyText"]),
        ])

    table = Table(
        rows,
        colWidths=[55 * mm, 120 * mm],
        hAlign="LEFT"
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7F8F6")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#087C75")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DCE5EE")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    return table


def _section_title(title, styles):
    return Paragraph(title, styles["SectionTitle"])


def _page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawRightString(
        A4[0] - 18 * mm,
        12 * mm,
        f"Page {doc.page}"
    )
    canvas.restoreState()


def generate_forensic_pdf(report):
    """
    Convert a structured SonicT forensic report dictionary
    into a PDF stored in memory.

    Returns:
        BytesIO
    """

    if not isinstance(report, dict):
        raise ValueError("Forensic report must be a dictionary.")

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="SonicT Audio Forensic Analysis Report",
        author="SonicT",
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#172033"),
            spaceAfter=5 * mm,
        )
    )

    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#64748B"),
            spaceAfter=7 * mm,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#087C75"),
            spaceBefore=5 * mm,
            spaceAfter=3 * mm,
        )
    )

    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=colors.HexColor("#087C75"),
        )
    )

    styles.add(
        ParagraphStyle(
            name="HashValue",
            parent=styles["BodyText"],
            fontName="Courier",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#172033"),
            wordWrap="CJK",
        )
    )

    styles["BodyText"].fontName = "Helvetica"
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 13
    styles["BodyText"].textColor = colors.HexColor("#334155")

    story = []

    report_information = report.get("report_information", {})
    evidence_integrity = report.get("evidence_integrity", {})
    privacy_and_retention = report.get("privacy_and_retention", {})
    final_assessment = report.get("final_assessment", {})
    probabilities = report.get("class_probabilities", {})
    forensic_evidence = report.get("forensic_evidence", {})
    suspicious_regions = report.get("suspicious_audio_regions", [])
    speaker_consistency = report.get("speaker_consistency", {})
    security_alert = report.get("security_alert")
    conclusion = report.get("forensic_conclusion", "")
    recommendation = report.get("recommended_action", "")
    disclaimer = report.get("disclaimer", "")

    story.append(
        Paragraph(
            "SonicT Audio Forensic Analysis Report",
            styles["ReportTitle"]
        )
    )

    story.append(
        Paragraph(
            "AI-assisted analysis for deepfake, tampering and replay evidence",
            styles["Subtitle"]
        )
    )

    story.append(_section_title("1. Report Information", styles))
    story.append(_dict_table(report_information, styles))
    story.append(Spacer(1, 3 * mm))

    story.append(_section_title("2. Evidence Integrity", styles))

    if isinstance(evidence_integrity, dict) and evidence_integrity:
        integrity_rows = []

        algorithm = evidence_integrity.get(
            "algorithm",
            "SHA-256"
        )
        status = evidence_integrity.get(
            "status",
            "NOT AVAILABLE"
        )
        sha256_value = evidence_integrity.get(
            "sha256"
        )
        description = evidence_integrity.get(
            "description",
            ""
        )

        integrity_rows.append([
            _paragraph("Algorithm", styles["BodyText"]),
            _paragraph(algorithm, styles["BodyText"]),
        ])

        integrity_rows.append([
            _paragraph("Status", styles["BodyText"]),
            _paragraph(status, styles["BodyText"]),
        ])

        integrity_rows.append([
            _paragraph("SHA-256", styles["BodyText"]),
            Paragraph(
                escape(_safe(sha256_value)),
                styles["HashValue"]
            ),
        ])

        integrity_rows.append([
            _paragraph("Description", styles["BodyText"]),
            _paragraph(description, styles["BodyText"]),
        ])

        integrity_table = Table(
            integrity_rows,
            colWidths=[55 * mm, 120 * mm],
            hAlign="LEFT"
        )

        integrity_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F6F9")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#172033")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DCE5EE")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )

        story.append(integrity_table)
    else:
        story.append(
            _paragraph(
                "No evidence-integrity information available.",
                styles["BodyText"]
            )
        )

    story.append(Spacer(1, 3 * mm))

    story.append(_section_title("3. Privacy & Evidence Retention", styles))

    if isinstance(privacy_and_retention, dict) and privacy_and_retention:
        story.append(
            _dict_table(
                privacy_and_retention,
                styles
            )
        )
    else:
        story.append(
            _paragraph(
                "No privacy or retention information available.",
                styles["BodyText"]
            )
        )

    story.append(Spacer(1, 3 * mm))

    story.append(_section_title("4. Final Assessment", styles))
    story.append(_dict_table(final_assessment, styles))
    story.append(Spacer(1, 3 * mm))

    story.append(_section_title("5. Class Probabilities", styles))

    if isinstance(probabilities, dict) and probabilities:
        probability_rows = [[
            _paragraph("Class", styles["TableHeader"]),
            _paragraph("Probability", styles["TableHeader"]),
        ]]

        for label, probability in sorted(
            probabilities.items(),
            key=lambda item: float(item[1] or 0),
            reverse=True
        ):
            probability_rows.append([
                _paragraph(str(label).upper(), styles["BodyText"]),
                _paragraph(
                    f"{float(probability or 0) * 100:.2f}%",
                    styles["BodyText"]
                ),
            ])

        probability_table = Table(
            probability_rows,
            colWidths=[85 * mm, 90 * mm],
            hAlign="LEFT"
        )

        probability_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7F8F6")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DCE5EE")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )

        story.append(probability_table)
    else:
        story.append(_paragraph("No class probabilities available.", styles["BodyText"]))

    story.append(Spacer(1, 3 * mm))

    story.append(_section_title("6. Forensic Evidence", styles))
    story.append(_nested_evidence_table(forensic_evidence, styles))
    story.append(Spacer(1, 3 * mm))

    story.append(_section_title("7. Suspicious Audio Regions", styles))

    if isinstance(suspicious_regions, list) and suspicious_regions:
        region_rows = [[
            _paragraph("No.", styles["TableHeader"]),
            _paragraph("Start", styles["TableHeader"]),
            _paragraph("End", styles["TableHeader"]),
            _paragraph("Probability", styles["TableHeader"]),
        ]]

        for index, region in enumerate(suspicious_regions, start=1):
            if not isinstance(region, dict):
                continue

            probability = region.get(
                "probability",
                region.get("score", 0)
            )

            region_rows.append([
                _paragraph(index, styles["BodyText"]),
                _paragraph(f"{float(region.get('start', 0)):.2f} s", styles["BodyText"]),
                _paragraph(f"{float(region.get('end', 0)):.2f} s", styles["BodyText"]),
                _paragraph(f"{float(probability or 0) * 100:.2f}%", styles["BodyText"]),
            ])

        region_table = Table(
            region_rows,
            colWidths=[20 * mm, 45 * mm, 45 * mm, 55 * mm],
            hAlign="LEFT"
        )

        region_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7F8F6")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DCE5EE")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )

        story.append(region_table)
    else:
        story.append(_paragraph("No suspicious audio regions were reported.", styles["BodyText"]))

    story.append(Spacer(1, 3 * mm))

    story.append(_section_title("8. Speaker Consistency", styles))
    story.append(_dict_table(speaker_consistency, styles))
    story.append(Spacer(1, 3 * mm))

    story.append(_section_title("9. Security Alert", styles))

    if isinstance(security_alert, dict):
        story.append(_dict_table(security_alert, styles))
    else:
        story.append(_paragraph("No security alert information available.", styles["BodyText"]))

    story.append(Spacer(1, 3 * mm))

    story.append(_section_title("10. Forensic Conclusion", styles))
    story.append(_paragraph(conclusion or "-", styles["BodyText"]))
    story.append(Spacer(1, 3 * mm))

    story.append(_section_title("11. Recommended Action", styles))
    story.append(_paragraph(recommendation or "-", styles["BodyText"]))
    story.append(Spacer(1, 3 * mm))

    story.append(_section_title("12. Forensic Use Notice", styles))
    story.append(_paragraph(disclaimer or "-", styles["BodyText"]))

    story.append(Spacer(1, 5 * mm))
    story.append(
        _paragraph(
            f"Generated by SonicT at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["BodyText"]
        )
    )

    doc.build(
        story,
        onFirstPage=_page_number,
        onLaterPages=_page_number
    )

    buffer.seek(0)
    return buffer
