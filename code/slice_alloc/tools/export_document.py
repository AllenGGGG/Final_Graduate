"""Render the single verified Markdown manuscript to matching Word; no PDF."""
from pathlib import Path
import re
from urllib.parse import quote
import zipfile
from docx import Document
from docx.shared import Cm,Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE
ROOT=Path(__file__).resolve().parents[2]
MD=ROOT/'面向低空智联的多模态网络切片技术.md'

def inline(paragraph, text):
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*")
    pos = 0
    for match in pattern.finditer(text):
        paragraph.add_run(text[pos:match.start()])
        if match.group(1):
            target = quote(match.group(2), safe="/:#?=&%")
            rel = paragraph.part.relate_to(target, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
            link = OxmlElement("w:hyperlink"); link.set(qn("r:id"), rel)
            run = OxmlElement("w:r"); props = OxmlElement("w:rPr")
            color = OxmlElement("w:color"); color.set(qn("w:val"), "245D86"); props.append(color)
            run.append(props); value = OxmlElement("w:t"); value.text = match.group(1); run.append(value)
            link.append(run); paragraph._p.append(link)
        elif match.group(3):
            run = paragraph.add_run(match.group(3)); run.font.name = "DejaVu Sans Mono"; run.font.size = Pt(9)
        else:
            paragraph.add_run(match.group(4)).bold = True
        pos = match.end()
    paragraph.add_run(text[pos:])


def to_docx(text, *, destination=None, validate_manuscript=False):
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2)
    section.left_margin = section.right_margin = Cm(2)
    for style_name in ("Normal", "Title", "Heading 1", "Heading 2", "Heading 3", "List Bullet", "List Number"):
        style = doc.styles[style_name]
        style.font.name = "Noto Sans CJK SC"
        style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")
    doc.styles["Normal"].font.size = Pt(10.5)
    doc.styles["Normal"].paragraph_format.space_after = Pt(6)
    lines, i, code = text.splitlines(), 0, False
    while i < len(lines):
        line = lines[i]; i += 1
        if line.startswith("```"):
            code = not code
            continue
        if code:
            p = doc.add_paragraph(); r = p.add_run(line); r.font.name = "DejaVu Sans Mono"; r.font.size = Pt(8)
            continue
        if not line.strip() or line.startswith("<!--"):
            continue
        image = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
        if image:
            from PIL import Image
            file = ROOT / image.group(2)
            with Image.open(file) as img:
                width = min(17., 22. * img.width / img.height)
            doc.add_picture(str(file), width=Cm(width))
            doc.add_paragraph(image.group(1), style="Caption")
            continue
        if line.startswith("|"):
            rows = [line]
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            data = [[c.strip() for c in row.strip("|").split("|")] for row in rows
                    if not re.fullmatch(r"[| :\-]+", row)]
            table = doc.add_table(rows=1, cols=len(data[0])); table.style = "Light Shading Accent 1"
            for k, row in enumerate(data):
                cells = table.rows[0].cells if k == 0 else table.add_row().cells
                for cell, value in zip(cells, row):
                    inline(cell.paragraphs[0], value)
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            doc.add_heading(heading.group(2), level=len(heading.group(1)) - 1)
        elif re.match(r"^\d+\. ", line):
            inline(doc.add_paragraph(style="List Number"), re.sub(r"^\d+\. ", "", line))
        elif line.startswith("- "):
            inline(doc.add_paragraph(style="List Bullet"), line[2:])
        else:
            inline(doc.add_paragraph(), line)
    doc.core_properties.title = "面向低空智联的多模态网络切片技术"
    doc.core_properties.subject = "背景与第一部分：切片联合智能分配、模型对比与ns-3功能执行验证"
    # Building a standalone report must not overwrite the thesis as a side
    # effect. Callers either save the returned Document or name a destination.
    if destination is None:
        return doc
    destination = Path(destination)
    doc.save(destination)
    with zipfile.ZipFile(destination) as z:
        xml = z.read("word/document.xml").decode()
        for token in ("低空智联下的多模态网络切片智能分配", "低空智联下的多模态网络切片资源预测", "低空智联下的多模态网络切片QoS保障", "search_limit"):
            if validate_manuscript and token not in xml:
                raise ValueError(f"Word missing required content: {token}")
        if len([n for n in z.namelist() if n.startswith("word/media/")]) != len(re.findall(r'^!\[', text, re.M)):
            raise ValueError("Word diagram missing or duplicated")
    return destination


