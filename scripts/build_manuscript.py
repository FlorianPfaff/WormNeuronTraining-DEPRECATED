"""Render the paper Markdown source as a deterministic, journal-style PDF.

The renderer deliberately supports a small, auditable Markdown subset rather
than depending on a browser engine.  Figure directives have this form::

    <!-- FIGURE: figures/figure-1-method.png -->

The referenced image is resolved below the output directory.  A following
paragraph beginning with ``Figure`` is kept with the image as its caption.
Likewise, paragraphs beginning with ``Table`` are styled as table captions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from html import escape
import os
from pathlib import Path
import re
import shutil
import unicodedata
from urllib.parse import urlparse

from reportlab import rl_config

# Stable object identifiers and a fixed PDF creation timestamp make repeated
# builds byte-for-byte reproducible when their Markdown and images are equal.
rl_config.invariant = True

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4, LETTER  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    CondPageBreak,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "paper" / "manuscript.md"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "paper"

FIGURE_RE = re.compile(r"^\s*<!--\s*FIGURE:\s*(.+?)\s*-->\s*$", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LIST_RE = re.compile(r"^(\s*)([-+*]|\d+[.)])\s+(.+?)\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
CAPTION_RE = re.compile(
    r"^(?:Supplementary\s+)?(?:Figure|Table)\s+(?:S?\d+[A-Za-z]?|[A-Z])(?:[.:]|\s)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Block:
    """A parsed Markdown block."""

    kind: str
    value: object
    level: int = 0


class DeterministicCanvas(canvas.Canvas):
    """Canvas with ReportLab's invariant metadata mode enabled."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs["invariant"] = 1
        kwargs["pageCompression"] = 1
        super().__init__(*args, **kwargs)


def _ascii_text(value: str) -> str:
    """Return Helvetica-safe text without silently producing missing glyphs."""

    replacements = {
        "\u00a0": " ",
        "\u00b0": " deg",
        "\u00b1": "+/-",
        "\u00b5": "u",
        "\u00d7": "x",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2026": "...",
        "\u2190": "<-",
        "\u2192": "->",
        "\u2260": "!=",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u03b1": "alpha",
        "\u03b2": "beta",
        "\u03b3": "gamma",
        "\u03b4": "delta",
        "\u03bc": "mu",
        "\u03c3": "sigma",
        "\u03c4": "tau",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


def _plain_markdown_title(value: str) -> str:
    """Return a metadata/header title without inline Markdown delimiters."""

    return re.sub(r"[*_`]", "", _ascii_text(value))


def _inline_markdown(value: str) -> str:
    """Convert a conservative inline-Markdown subset to ReportLab markup."""

    value = _ascii_text(value.strip())
    tokens: list[str] = []

    def stash(markup: str) -> str:
        token = f"@@MDTOKEN{len(tokens)}@@"
        tokens.append(markup)
        return token

    # Preserve escaped Markdown punctuation before interpreting emphasis.
    value = re.sub(
        r"\\([\\`*_[\]()+.#!-])",
        lambda match: stash(escape(match.group(1))),
        value,
    )

    # Code spans are opaque: Markdown syntax inside them remains literal.
    value = re.sub(
        r"`([^`\n]+)`",
        lambda match: stash(
            '<font name="Courier" size="8" backColor="#F2F2F2">'
            f"{escape(_ascii_text(match.group(1)))}"
            "</font>"
        ),
        value,
    )

    def link_replacement(match: re.Match[str]) -> str:
        label = escape(_ascii_text(match.group(1)))
        destination = _ascii_text(match.group(2).strip())
        parsed = urlparse(destination)
        if parsed.scheme.lower() not in {"http", "https", "mailto"} and not destination.startswith(
            "#"
        ):
            return f"{label} ({escape(destination)})"
        href = escape(destination, quote=True)
        return stash(f'<link href="{href}" color="#245C7A"><u>{label}</u></link>')

    value = re.sub(r"\[([^\]\n]+)\]\(([^)\n]+)\)", link_replacement, value)
    value = escape(value)

    # Strong emphasis is processed first so ***text*** nests correctly.
    value = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"__(?=\S)(.+?)(?<=\S)__", r"<b>\1</b>", value)
    value = re.sub(r"\*(?=\S)(.+?)(?<=\S)\*", r"<i>\1</i>", value)
    value = re.sub(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)", r"<i>\1</i>", value)

    # Tokens can be nested (a link label may include a protected code token),
    # so restore until the string reaches a fixed point.
    previous = None
    while previous != value:
        previous = value
        for index, markup in enumerate(tokens):
            value = value.replace(f"@@MDTOKEN{index}@@", markup)
    return value


def _split_table_row(line: str) -> list[str]:
    """Split a pipe table row, respecting backslash-escaped pipes."""

    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|") and not text.endswith(r"\|"):
        text = text[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in text:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def _is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines) or "|" not in lines[index]:
        return False
    first = _split_table_row(lines[index])
    second = _split_table_row(lines[index + 1])
    return bool(first) and len(first) == len(second) and all(
        TABLE_SEPARATOR_RE.fullmatch(cell.strip()) for cell in second
    )


def _parse_markdown(text: str) -> list[Block]:
    """Parse headings, prose, lists, tables, figures, rules, and code blocks."""

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(Block("paragraph", " ".join(part.strip() for part in paragraph)))
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush_paragraph()
            fence = stripped[:3]
            language = stripped[3:].strip()
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(fence):
                code_lines.append(lines[index])
                index += 1
            if index >= len(lines):
                raise ValueError("unterminated fenced code block")
            blocks.append(Block("code", (language, "\n".join(code_lines))))
            index += 1
            continue

        figure_match = FIGURE_RE.match(line)
        if figure_match:
            flush_paragraph()
            blocks.append(Block("figure", figure_match.group(1).strip()))
            index += 1
            continue

        if stripped.startswith("<!--"):
            flush_paragraph()
            # Ordinary comments carry editing notes, not manuscript content.
            while "-->" not in stripped and index + 1 < len(lines):
                index += 1
                stripped = lines[index].strip()
            index += 1
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            blocks.append(
                Block("heading", heading_match.group(2).strip(), len(heading_match.group(1)))
            )
            index += 1
            continue

        if stripped in {"---", "***", "___"}:
            flush_paragraph()
            blocks.append(Block("rule", ""))
            index += 1
            continue

        if _is_table_start(lines, index):
            flush_paragraph()
            header = _split_table_row(lines[index])
            separators = _split_table_row(lines[index + 1])
            rows: list[list[str]] = []
            index += 2
            while index < len(lines) and lines[index].strip() and "|" in lines[index]:
                row = _split_table_row(lines[index])
                if len(row) != len(header):
                    break
                rows.append(row)
                index += 1
            alignments = [
                "CENTER"
                if cell.startswith(":") and cell.endswith(":")
                else "RIGHT"
                if cell.endswith(":")
                else "LEFT"
                for cell in separators
            ]
            blocks.append(Block("table", (header, rows, alignments)))
            continue

        list_match = LIST_RE.match(line)
        if list_match:
            flush_paragraph()
            indent = len(list_match.group(1).expandtabs(4))
            marker = list_match.group(2)
            body = [list_match.group(3)]
            index += 1
            while index < len(lines):
                continuation = lines[index]
                if not continuation.strip():
                    break
                if LIST_RE.match(continuation) or HEADING_RE.match(continuation):
                    break
                if FIGURE_RE.match(continuation) or _is_table_start(lines, index):
                    break
                leading = len(continuation) - len(continuation.lstrip())
                if leading <= len(list_match.group(1)):
                    break
                body.append(continuation.strip())
                index += 1
            blocks.append(Block("list", (marker, " ".join(body)), indent))
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip()[1:].lstrip())
                index += 1
            blocks.append(Block("quote", " ".join(quote_lines)))
            continue

        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    return blocks


def _styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    body = ParagraphStyle(
        "ManuscriptBody",
        parent=sample["BodyText"],
        fontName="Helvetica",
        fontSize=9.3,
        leading=13.2,
        textColor=colors.HexColor("#22272B"),
        spaceAfter=6.5,
        alignment=TA_LEFT,
        allowWidows=0,
        allowOrphans=0,
    )
    return {
        "title": ParagraphStyle(
            "ManuscriptTitle",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=28,
            textColor=colors.HexColor("#163A4A"),
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "front": ParagraphStyle(
            "FrontMatter",
            parent=body,
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#53636B"),
            spaceAfter=8,
        ),
        "body": body,
        "h1": ParagraphStyle(
            "Heading1",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=colors.HexColor("#163A4A"),
            spaceBefore=16,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=15,
            textColor=colors.HexColor("#184F63"),
            spaceBefore=13,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Heading3",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#285C69"),
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "h4": ParagraphStyle(
            "Heading4",
            parent=body,
            fontName="Helvetica-BoldOblique",
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#2E4D56"),
            spaceBefore=8,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "list": ParagraphStyle(
            "ListItem",
            parent=body,
            leftIndent=18,
            firstLineIndent=-12,
            bulletIndent=4,
            spaceAfter=3,
        ),
        "quote": ParagraphStyle(
            "BlockQuote",
            parent=body,
            leftIndent=18,
            rightIndent=12,
            borderWidth=0,
            borderPadding=6,
            backColor=colors.HexColor("#F3F6F7"),
            textColor=colors.HexColor("#3B4A50"),
            fontName="Helvetica-Oblique",
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=body,
            fontSize=8.1,
            leading=10.6,
            textColor=colors.HexColor("#3C474C"),
            spaceBefore=4,
            spaceAfter=10,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=body,
            fontSize=7.25,
            leading=9.1,
            spaceAfter=0,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=7.1,
            leading=8.8,
            textColor=colors.white,
            spaceAfter=0,
        ),
        "code": ParagraphStyle(
            "CodeBlock",
            parent=body,
            fontName="Courier",
            fontSize=7.1,
            leading=9.2,
            leftIndent=8,
            rightIndent=8,
            borderPadding=7,
            backColor=colors.HexColor("#F1F3F4"),
            textColor=colors.HexColor("#30363A"),
            spaceBefore=4,
            spaceAfter=8,
        ),
    }


def _column_widths(rows: list[list[str]], available_width: float) -> list[float]:
    """Allocate table width by content while retaining a readable minimum."""

    columns = len(rows[0])
    lengths = [
        max(4, min(42, max(len(_ascii_text(row[index])) for row in rows)))
        for index in range(columns)
    ]
    weights = [length**0.55 for length in lengths]
    minimum = min(54.0, available_width / columns * 0.72)
    widths = [max(minimum, available_width * weight / sum(weights)) for weight in weights]
    excess = sum(widths) - available_width
    while excess > 0.01:
        flexible = [index for index, width in enumerate(widths) if width > minimum + 0.01]
        if not flexible:
            widths = [available_width / columns] * columns
            break
        reduction = excess / len(flexible)
        for index in flexible:
            widths[index] = max(minimum, widths[index] - reduction)
        excess = sum(widths) - available_width
    return widths


def _table_flowable(block: Block, styles: dict[str, ParagraphStyle], width: float) -> Table:
    header, body, alignments = block.value  # type: ignore[misc]
    raw_rows = [header, *body]
    data = [
        [
            Paragraph(_inline_markdown(cell), styles["table_head" if row_index == 0 else "table_cell"])
            for cell in row
        ]
        for row_index, row in enumerate(raw_rows)
    ]
    table = Table(
        data,
        colWidths=_column_widths(raw_rows, width),
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
        spaceBefore=4,
        spaceAfter=9,
    )
    commands: list[tuple[object, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#285B6E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#163A4A")),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#D8E0E3")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#82969E")),
    ]
    for row_index in range(1, len(data)):
        if row_index % 2 == 0:
            commands.append(
                ("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F4F7F8"))
            )
    for column_index, alignment in enumerate(alignments):
        commands.append(("ALIGN", (column_index, 1), (column_index, -1), alignment))
    table.setStyle(TableStyle(commands))
    return table


def _figure_flowable(relative_path: str, output_dir: Path, width: float, height: float) -> Image:
    figure_path = (output_dir / relative_path).resolve()
    try:
        figure_path.relative_to(output_dir.resolve())
    except ValueError as error:
        raise ValueError(f"figure path escapes output directory: {relative_path}") from error
    if not figure_path.is_file():
        raise FileNotFoundError(f"figure referenced by manuscript does not exist: {figure_path}")

    image_width, image_height = ImageReader(str(figure_path)).getSize()
    scale = min(width / image_width, height / image_height, 1.0)
    figure = Image(str(figure_path), width=image_width * scale, height=image_height * scale)
    figure.hAlign = "CENTER"
    figure._restrictSize(width, height)
    return figure


def _paragraph(block: Block, styles: dict[str, ParagraphStyle]) -> Paragraph:
    style = styles["caption"] if CAPTION_RE.match(str(block.value)) else styles["body"]
    return Paragraph(_inline_markdown(str(block.value)), style)


def _story(
    blocks: list[Block],
    title: str,
    front_matter: list[Block],
    output_dir: Path,
    available_width: float,
    available_height: float,
) -> list[object]:
    styles = _styles()
    story: list[object] = [
        Spacer(1, 1.85 * inch),
        Paragraph(_inline_markdown(title), styles["title"]),
        HRFlowable(
            width="32%",
            thickness=1.2,
            color=colors.HexColor("#5B91A3"),
            spaceBefore=4,
            spaceAfter=18,
            hAlign="CENTER",
        ),
    ]
    for block in front_matter:
        story.append(Paragraph(_inline_markdown(str(block.value)), styles["front"]))
    story.append(PageBreak())

    index = 0
    while index < len(blocks):
        block = blocks[index]
        next_block = blocks[index + 1] if index + 1 < len(blocks) else None

        if block.kind == "heading":
            style_name = f"h{min(block.level, 4)}"
            story.append(Paragraph(_inline_markdown(str(block.value)), styles[style_name]))
        elif block.kind == "paragraph":
            # Keep a table caption above its table.
            if CAPTION_RE.match(str(block.value)) and next_block and next_block.kind == "table":
                table = _table_flowable(next_block, styles, available_width)
                story.append(KeepTogether([_paragraph(block, styles), table]))
                index += 1
            else:
                story.append(_paragraph(block, styles))
        elif block.kind == "list":
            marker, item = block.value  # type: ignore[misc]
            list_style = ParagraphStyle(
                f"ListIndent{block.level}",
                parent=styles["list"],
                leftIndent=18 + min(block.level, 16),
                bulletIndent=4 + min(block.level, 16),
            )
            bullet = "-" if marker in {"-", "+", "*"} else str(marker)
            story.append(Paragraph(_inline_markdown(str(item)), list_style, bulletText=bullet))
        elif block.kind == "quote":
            story.append(Paragraph(_inline_markdown(str(block.value)), styles["quote"]))
        elif block.kind == "rule":
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.45,
                    color=colors.HexColor("#AAB8BE"),
                    spaceBefore=8,
                    spaceAfter=10,
                )
            )
        elif block.kind == "code":
            language, code = block.value  # type: ignore[misc]
            label = f"{language}\n" if language else ""
            story.append(Preformatted(_ascii_text(label + code), styles["code"], maxLineLength=90))
        elif block.kind == "table":
            table = _table_flowable(block, styles, available_width)
            if next_block and next_block.kind == "paragraph" and CAPTION_RE.match(
                str(next_block.value)
            ):
                story.append(KeepTogether([table, _paragraph(next_block, styles)]))
                index += 1
            else:
                # Manuscript tables are compact enough to move intact to the next page.
                # Avoid leaving a header and a single row orphaned at a page boundary.
                story.append(KeepTogether([table]))
        elif block.kind == "figure":
            # A following caption is kept on the same page as its figure.
            figure = _figure_flowable(
                str(block.value),
                output_dir,
                available_width,
                min(6.15 * inch, available_height * 0.72),
            )
            figure_items: list[object] = [CondPageBreak(2.2 * inch), figure]
            if next_block and next_block.kind == "paragraph" and CAPTION_RE.match(
                str(next_block.value)
            ):
                figure_items.append(_paragraph(next_block, styles))
                index += 1
            else:
                figure_items.append(Spacer(1, 8))
            story.append(KeepTogether(figure_items))
        else:
            raise ValueError(f"unsupported parsed block: {block.kind}")
        index += 1
    return story


def _repo_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must be inside repository root {ROOT}: {path}") from error
    return path


def _extract_title(blocks: list[Block]) -> tuple[str, list[Block], list[Block]]:
    """Extract the first H1 and any explicit front matter that follows it."""

    title_index = next(
        (index for index, block in enumerate(blocks) if block.kind == "heading" and block.level == 1),
        None,
    )
    if title_index is None:
        raise ValueError("manuscript must contain a level-one Markdown title")
    title = str(blocks[title_index].value)

    body = blocks[:title_index] + blocks[title_index + 1 :]
    front_matter: list[Block] = []
    # Plain paragraphs between the title and first section are treated as
    # author/byline text, but no content is invented when they are absent.
    while title_index < len(body) and body[title_index].kind == "paragraph":
        front_matter.append(body.pop(title_index))
    return title, front_matter, body


def _page_callback(title: str, first_page: bool, source_sha256: str):
    header = _plain_markdown_title(title)
    if len(header) > 88:
        header = header[:85].rstrip() + "..."

    def draw(page_canvas: canvas.Canvas, document: SimpleDocTemplate) -> None:
        page_canvas.saveState()
        page_canvas.setTitle(_plain_markdown_title(title))
        page_canvas.setCreator("build_manuscript.py")
        page_canvas.setSubject(f"Scientific manuscript; source-sha256={source_sha256}")
        page_canvas.setKeywords(f"source-sha256={source_sha256}")
        if not first_page:
            page_canvas.setStrokeColor(colors.HexColor("#B8C4C9"))
            page_canvas.setLineWidth(0.35)
            page_canvas.line(
                document.leftMargin,
                document.pagesize[1] - 37,
                document.pagesize[0] - document.rightMargin,
                document.pagesize[1] - 37,
            )
            page_canvas.setFont("Helvetica", 7.5)
            page_canvas.setFillColor(colors.HexColor("#64757D"))
            page_canvas.drawString(document.leftMargin, document.pagesize[1] - 31, header)
        page_canvas.setFont("Helvetica", 7.5)
        page_canvas.setFillColor(colors.HexColor("#64757D"))
        page_canvas.drawCentredString(
            document.pagesize[0] / 2,
            27,
            str(document.page),
        )
        page_canvas.restoreState()

    return draw


def build_manuscript(source: Path, output_dir: Path, page_size_name: str = "letter") -> Path:
    """Build the PDF and copy its exact Markdown source beside it."""

    if not source.is_file():
        raise FileNotFoundError(f"manuscript source does not exist: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_text = source.read_text(encoding="utf-8")
    source_sha256 = sha256(source.read_bytes()).hexdigest()
    blocks = _parse_markdown(source_text)
    title, front_matter, body = _extract_title(blocks)
    page_size = {"letter": LETTER, "a4": A4}[page_size_name]

    destination = output_dir / "manuscript.pdf"
    temporary_pdf = output_dir / ".manuscript.tmp.pdf"
    temporary_source = output_dir / ".manuscript.tmp.md"
    source_copy = output_dir / "manuscript.md"

    document = SimpleDocTemplate(
        str(temporary_pdf),
        pagesize=page_size,
        leftMargin=0.76 * inch,
        rightMargin=0.76 * inch,
        topMargin=0.67 * inch,
        bottomMargin=0.63 * inch,
        title=_plain_markdown_title(title),
        author="",
        subject=f"Scientific manuscript; source-sha256={source_sha256}",
        creator="build_manuscript.py",
        displayDocTitle=True,
    )
    story = _story(
        body,
        title,
        front_matter,
        output_dir,
        document.width,
        document.height,
    )

    try:
        document.build(
            story,
            onFirstPage=_page_callback(
                title,
                first_page=True,
                source_sha256=source_sha256,
            ),
            onLaterPages=_page_callback(
                title,
                first_page=False,
                source_sha256=source_sha256,
            ),
            canvasmaker=DeterministicCanvas,
        )
        shutil.copyfile(source, temporary_source)
        os.replace(temporary_pdf, destination)
        os.replace(temporary_source, source_copy)
    finally:
        temporary_pdf.unlink(missing_ok=True)
        temporary_source.unlink(missing_ok=True)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE.relative_to(ROOT)),
        help="Markdown manuscript path under the repository root",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT)),
        help="Output directory under the repository root",
    )
    parser.add_argument(
        "--page-size",
        choices=("letter", "a4"),
        default="letter",
        help="PDF page size (default: letter)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = _repo_path(args.source, label="source")
    output_dir = _repo_path(args.output_dir, label="output directory")
    destination = build_manuscript(source, output_dir, args.page_size)
    print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
