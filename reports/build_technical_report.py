#!/usr/bin/env python3
"""Build the English competition report with ReportLab.

The Markdown file remains the reviewable source. This renderer intentionally
supports only the constructs used by that source and fails on missing images.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    CondPageBreak,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    LongTable,
    PageBreak,
    Paragraph,
    Preformatted,
    Spacer,
    SimpleDocTemplate,
    Table,
    TableStyle,
)


INK = colors.HexColor("#172235")
MUTED = colors.HexColor("#5B6576")
ACCENT = colors.HexColor("#B91C3B")
ACCENT_DARK = colors.HexColor("#86152E")
PALE = colors.HexColor("#F4F6F9")
RULE = colors.HexColor("#D8DDE6")
WHITE = colors.white


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        (
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            Path("/System/Library/Fonts/Menlo.ttc"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        ),
    ]
    for regular, bold, mono in candidates:
        if regular.is_file() and bold.is_file() and mono.is_file():
            pdfmetrics.registerFont(TTFont("OneLoopSans", str(regular)))
            pdfmetrics.registerFont(TTFont("OneLoopSansBold", str(bold)))
            pdfmetrics.registerFont(TTFont("OneLoopMono", str(mono)))
            pdfmetrics.registerFontFamily(
                "OneLoopSans",
                normal="OneLoopSans",
                bold="OneLoopSansBold",
            )
            return "OneLoopSans", "OneLoopSansBold", "OneLoopMono"
    return "Helvetica", "Helvetica-Bold", "Courier"


FONT, FONT_BOLD, FONT_MONO = register_fonts()


def ascii_dashes(value: str) -> str:
    return (
        value.replace("\N{EM DASH}", " - ")
        .replace("\N{EN DASH}", "-")
        .replace("\N{NON-BREAKING HYPHEN}", "-")
        .replace("\N{MINUS SIGN}", "-")
    )


def inline_markup(value: str) -> str:
    value = ascii_dashes(value.strip())
    protected: list[str] = []

    def stash(markup: str) -> str:
        protected.append(markup)
        return f"@@ONELOOP{len(protected) - 1}@@"

    def link(match: re.Match[str]) -> str:
        label, url = match.groups()
        return stash(
            f'<link href="{html.escape(url, quote=True)}" color="#86152E">'
            f"{html.escape(label)}</link>"
        )

    value = re.sub(r"\[([^]]+)]\(([^)]+)\)", link, value)

    def code(match: re.Match[str]) -> str:
        return stash(
            f'<font name="{FONT_MONO}" color="#86152E">'
            f"{html.escape(match.group(1))}</font>"
        )

    value = re.sub(r"`([^`]+)`", code, value)
    value = html.escape(value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    for index, markup in enumerate(protected):
        value = value.replace(f"@@ONELOOP{index}@@", markup)
    return value


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "OneLoopBody",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.2,
            leading=13.1,
            textColor=INK,
            spaceAfter=5.5,
            splitLongWords=True,
        ),
        "h2": ParagraphStyle(
            "OneLoopH2",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=16,
            leading=19,
            textColor=INK,
            spaceBefore=13,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "OneLoopH3",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=11.5,
            leading=14,
            textColor=ACCENT_DARK,
            spaceBefore=9,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "h4": ParagraphStyle(
            "OneLoopH4",
            parent=base["Heading3"],
            fontName=FONT_BOLD,
            fontSize=9.5,
            leading=12,
            textColor=INK,
            spaceBefore=7,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "table": ParagraphStyle(
            "OneLoopTable",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.4,
            leading=9.2,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "OneLoopTableHead",
            parent=base["BodyText"],
            fontName=FONT_BOLD,
            fontSize=7.5,
            leading=9.2,
            textColor=WHITE,
        ),
        "caption": ParagraphStyle(
            "OneLoopCaption",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.4,
            leading=9.2,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=8,
        ),
        "callout": ParagraphStyle(
            "OneLoopCallout",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8.7,
            leading=12.2,
            textColor=INK,
            leftIndent=9,
            rightIndent=9,
            borderColor=ACCENT,
            borderWidth=0,
            borderPadding=7,
            backColor=colors.HexColor("#FCEEF2"),
            spaceBefore=4,
            spaceAfter=8,
        ),
    }


STYLES = make_styles()


class ReportDoc(SimpleDocTemplate):
    @staticmethod
    def page_chrome(canvas: object, doc: object) -> None:
        canvas.saveState()
        width, height = A4
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.45)
        canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
        canvas.setFont(FONT_BOLD, 6.8)
        canvas.setFillColor(ACCENT_DARK)
        canvas.drawString(18 * mm, height - 11 * mm, "RADEON ONELOOP")
        canvas.setFont(FONT, 6.8)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(
            width - 18 * mm,
            height - 11 * mm,
            "AMD Radeon Hackathon 2026 - Track 3",
        )
        canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
        canvas.drawString(18 * mm, 9 * mm, "Phi Media Lab")
        canvas.drawRightString(width - 18 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()


def title_page(title: str, subtitle: str) -> list[object]:
    title_style = ParagraphStyle(
        "CoverTitle",
        fontName=FONT_BOLD,
        fontSize=30,
        leading=34,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=9,
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        fontName=FONT,
        fontSize=15,
        leading=20,
        textColor=ACCENT_DARK,
        alignment=TA_LEFT,
    )
    meta_style = ParagraphStyle(
        "CoverMeta",
        fontName=FONT,
        fontSize=10,
        leading=16,
        textColor=INK,
        leftIndent=9,
        borderColor=ACCENT,
        borderWidth=0,
        borderPadding=11,
        backColor=PALE,
    )
    return [
        Spacer(1, 28 * mm),
        Table([["ONE RADEON. ONE ROBOT LOOP."]], colWidths=[64 * mm], style=[
            ("BACKGROUND", (0, 0), (-1, -1), ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, -1), WHITE),
            ("FONTNAME", (0, 0), (-1, -1), FONT_BOLD),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]),
        Spacer(1, 9 * mm),
        Paragraph(inline_markup(title), title_style),
        Paragraph(inline_markup(subtitle), subtitle_style),
        Spacer(1, 12 * mm),
        Table([[""]], colWidths=[44 * mm], rowHeights=[1.4 * mm], style=[
            ("BACKGROUND", (0, 0), (-1, -1), ACCENT),
        ]),
        Spacer(1, 15 * mm),
        Paragraph(
            "<b>AMD Radeon Hackathon 2026 - Track 3: Physical AI</b><br/>"
            "Team: Phi Media Lab<br/>"
            "Formal profile: one AMD Radeon gfx1100 / ROCm 7.2.1<br/>"
            "Project repository: github.com/phi-media-lab/radeon-oneloop",
            meta_style,
        ),
        Spacer(1, 20 * mm),
        Paragraph(
            "A reproducible bimanual handover pipeline that runs the Genesis "
            "environment, policy training, and real-time inference on a single "
            "Radeon while learning directly from reviewed human corrections.",
            ParagraphStyle(
                "CoverSummary",
                fontName=FONT,
                fontSize=13,
                leading=19,
                textColor=INK,
            ),
        ),
        PageBreak(),
    ]


def parse_table(lines: list[str], start: int) -> tuple[object, int]:
    rows: list[list[str]] = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        rows.append(cells)
        index += 1
    if len(rows) < 2 or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]):
        raise ValueError(f"malformed Markdown table near line {start + 1}")
    rows.pop(1)
    width = 174 * mm
    count = max(len(row) for row in rows)
    if count == 2:
        col_widths = [0.44 * width, 0.56 * width]
    elif count == 3:
        col_widths = [0.38 * width, 0.31 * width, 0.31 * width]
    else:
        col_widths = [width / count] * count
    rendered = []
    for row_index, row in enumerate(rows):
        style = STYLES["table_head"] if row_index == 0 else STYLES["table"]
        rendered.append([Paragraph(inline_markup(cell), style) for cell in row])
    table = LongTable(rendered, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT_DARK),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
                ("GRID", (0, 0), (-1, -1), 0.35, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return KeepTogether([Spacer(1, 2), table, Spacer(1, 7)]), index


def parse_markdown(source: Path) -> list[object]:
    lines = source.read_text(encoding="utf-8").splitlines()
    title = lines[0].removeprefix("# ").strip()
    subtitle = lines[2].removeprefix("## ").strip()
    story = title_page(title, subtitle)
    index = next(
        position for position, value in enumerate(lines) if value.strip() == "## Abstract"
    )
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            story.append(Paragraph(inline_markup(" ".join(paragraph)), STYLES["body"]))
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush()
            index += 1
            continue
        if index < 12 and stripped.startswith("**"):
            index += 1
            continue
        if stripped.startswith("```"):
            flush()
            language = stripped[3:].strip()
            code: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(ascii_dashes(lines[index]))
                index += 1
            if index == len(lines):
                raise ValueError("unterminated fenced code block")
            code_style = ParagraphStyle(
                "OneLoopCode",
                fontName=FONT_MONO,
                fontSize=6.8,
                leading=9,
                textColor=INK,
                leftIndent=6,
                rightIndent=6,
                borderPadding=7,
                backColor=colors.HexColor("#EEF1F5"),
                spaceBefore=3,
                spaceAfter=7,
            )
            story.append(Preformatted("\n".join(code), code_style, maxLineLength=102))
            index += 1
            continue
        image_match = re.fullmatch(r"!\[([^]]*)]\(([^)]+)\)", stripped)
        if image_match:
            flush()
            caption, relative = image_match.groups()
            image_path = (source.parent / relative).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            picture = Image(str(image_path))
            picture._restrictSize(165 * mm, 100 * mm)
            story.append(
                KeepTogether(
                    [
                        Spacer(1, 3),
                        picture,
                        Paragraph(inline_markup(caption), STYLES["caption"]),
                    ]
                )
            )
            index += 1
            continue
        if stripped.startswith("|"):
            flush()
            table, index = parse_table(lines, index)
            story.append(table)
            continue
        heading = re.match(r"^(#{2,4})\s+(.+)$", stripped)
        if heading:
            flush()
            level = len(heading.group(1))
            if level == 2:
                story.append(CondPageBreak(38 * mm))
            story.append(Paragraph(inline_markup(heading.group(2)), STYLES[f"h{level}"]))
            index += 1
            continue
        if stripped.startswith(">"):
            flush()
            story.append(Paragraph(inline_markup(stripped.lstrip("> ")), STYLES["callout"]))
            index += 1
            continue
        if re.match(r"^[-*]\s+", stripped):
            flush()
            items = []
            while index < len(lines) and re.match(r"^\s*[-*]\s+", lines[index]):
                value = re.sub(r"^\s*[-*]\s+", "", lines[index])
                index += 1
                while (
                    index < len(lines)
                    and lines[index].strip()
                    and not re.match(r"^\s*[-*]\s+", lines[index])
                ):
                    value += " " + lines[index].strip()
                    index += 1
                items.append(ListItem(Paragraph(inline_markup(value), STYLES["body"])))
            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="circle",
                    leftIndent=15,
                    bulletFontName=FONT,
                    bulletFontSize=5,
                    spaceAfter=4,
                )
            )
            continue
        if re.match(r"^\d+\.\s+", stripped):
            flush()
            items = []
            while index < len(lines) and re.match(r"^\s*\d+\.\s+", lines[index]):
                value = re.sub(r"^\s*\d+\.\s+", "", lines[index])
                index += 1
                while (
                    index < len(lines)
                    and lines[index].strip()
                    and not re.match(r"^\s*\d+\.\s+", lines[index])
                ):
                    value += " " + lines[index].strip()
                    index += 1
                items.append(ListItem(Paragraph(inline_markup(value), STYLES["body"])))
            story.append(
                ListFlowable(
                    items,
                    bulletType="1",
                    start="1",
                    leftIndent=18,
                    bulletFontName=FONT,
                    bulletFontSize=8,
                    spaceAfter=4,
                )
            )
            continue
        paragraph.append(stripped)
        index += 1
    flush()
    return story


def build(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    document = ReportDoc(
        str(output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title="Radeon OneLoop: Single-Radeon Phase-Aware HIL Bimanual Handover",
        author="Phi Media Lab",
        subject="AMD Radeon Hackathon 2026 Track 3 Technical Report",
    )
    document.build(
        parse_markdown(source),
        onFirstPage=document.page_chrome,
        onLaterPages=document.page_chrome,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("technical_report.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1]
        / "output"
        / "pdf"
        / "radeon-oneloop-technical-report.pdf",
    )
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
