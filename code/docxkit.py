"""Helpers for building the SPIE A4 manuscript in Word format.

Includes a minimal OMML (Office Math Markup Language) builder so display
equations are native, editable Word equations rather than pictures, as the
venue checklist requires.
"""

from __future__ import annotations

from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from lxml import etree

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def m(tag: str):
    return etree.Element("{%s}%s" % (M, tag))


def _box(tag: str, content):
    element = m(tag)
    if isinstance(content, (list, tuple)):
        for item in content:
            element.append(item)
    else:
        element.append(content)
    return element


def run(text: str, style: str | None = None):
    """A math run. style: 'i' italic, 'p' upright, 'b' bold, 'bi' bold italic."""
    element = m("r")
    if style:
        rpr = etree.SubElement(element, "{%s}rPr" % M)
        sty = etree.SubElement(rpr, "{%s}sty" % M)
        sty.set("{%s}val" % M, style)
    t = etree.SubElement(element, "{%s}t" % M)
    t.set(XML_SPACE, "preserve")
    t.text = text
    return element


def op(text: str):
    return run(text, "p")


def var(text: str):
    return run(text, "i")


def subs(base, sub):
    element = m("sSub")
    element.append(_box("e", base))
    element.append(_box("sub", sub))
    return element


def sups(base, sup):
    element = m("sSup")
    element.append(_box("e", base))
    element.append(_box("sup", sup))
    return element


def subsups(base, sub, sup):
    element = m("sSubSup")
    element.append(_box("e", base))
    element.append(_box("sub", sub))
    element.append(_box("sup", sup))
    return element


def frac(num, den):
    element = m("f")
    element.append(_box("num", num))
    element.append(_box("den", den))
    return element


def rad(content):
    element = m("rad")
    rpr = etree.SubElement(element, "{%s}radPr" % M)
    hide = etree.SubElement(rpr, "{%s}degHide" % M)
    hide.set("{%s}val" % M, "1")
    element.append(m("deg"))
    element.append(_box("e", content))
    return element


def delim(content, begin="(", end=")"):
    element = m("d")
    dpr = etree.SubElement(element, "{%s}dPr" % M)
    beg = etree.SubElement(dpr, "{%s}begChr" % M)
    beg.set("{%s}val" % M, begin)
    fin = etree.SubElement(dpr, "{%s}endChr" % M)
    fin.set("{%s}val" % M, end)
    element.append(_box("e", content))
    return element


def acc(content, char="\u0303"):
    element = m("acc")
    accpr = etree.SubElement(element, "{%s}accPr" % M)
    chr_el = etree.SubElement(accpr, "{%s}chr" % M)
    chr_el.set("{%s}val" % M, char)
    element.append(_box("e", content))
    return element


def bar(content):
    element = m("bar")
    barpr = etree.SubElement(element, "{%s}barPr" % M)
    pos = etree.SubElement(barpr, "{%s}pos" % M)
    pos.set("{%s}val" % M, "top")
    element.append(_box("e", content))
    return element


def nary(symbol, sub, sup, content):
    element = m("nary")
    npr = etree.SubElement(element, "{%s}naryPr" % M)
    chr_el = etree.SubElement(npr, "{%s}chr" % M)
    chr_el.set("{%s}val" % M, symbol)
    lim = etree.SubElement(npr, "{%s}limLoc" % M)
    lim.set("{%s}val" % M, "undOvr")
    element.append(_box("sub", sub))
    element.append(_box("sup", sup))
    element.append(_box("e", content))
    return element


def eqarr(lines):
    element = m("eqArr")
    for line in lines:
        element.append(_box("e", line))
    return element


def omath(*items):
    element = etree.Element("{%s}oMath" % M, nsmap={"m": M, "w": W})
    for item in items:
        if isinstance(item, (list, tuple)):
            for sub_item in item:
                element.append(sub_item)
        else:
            element.append(item)
    return element


def add_equation(doc, items, number: int, text_width_cm: float = 17.14):
    """Insert a numbered display equation: centred equation, right-aligned tag."""
    paragraph = doc.add_paragraph(style="SPIE body text")
    fmt = paragraph.paragraph_format
    fmt.alignment = WD_ALIGN_PARAGRAPH.LEFT
    fmt.first_line_indent = Cm(0)
    fmt.space_before = Pt(3)
    fmt.space_after = Pt(3)
    fmt.keep_together = True
    fmt.tab_stops.add_tab_stop(Cm(text_width_cm / 2), WD_TAB_ALIGNMENT.CENTER)
    fmt.tab_stops.add_tab_stop(Cm(text_width_cm), WD_TAB_ALIGNMENT.RIGHT)
    lead = paragraph.add_run()
    lead.add_tab()
    paragraph._p.append(omath(items))
    tail = paragraph.add_run()
    tail.add_tab()
    tag = paragraph.add_run(f"({number})")
    tag.font.size = Pt(10)
    return paragraph


def _set_cell_border(cell, **kwargs):
    """Set cell borders. kwargs: top/bottom dict(sz=, val=, color=)."""
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right"):
        if edge not in kwargs:
            continue
        spec = kwargs[edge]
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), spec.get("val", "single"))
        element.set(qn("w:sz"), str(spec.get("sz", 4)))
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), spec.get("color", "000000"))


def add_table(doc, caption: str, header, rows, widths_cm, font_size: float = 9.0,
              align=None, first_col_left: bool = True):
    """Add a booktabs-style table with a 9 pt caption above it."""
    cap = doc.add_paragraph(style="SPIE figure caption")
    cap.paragraph_format.space_before = Pt(6)
    cap.paragraph_format.space_after = Pt(3)
    cap.paragraph_format.keep_with_next = True
    run_caption = cap.add_run(caption)
    run_caption.font.size = Pt(9)

    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)

    for index, width in enumerate(widths_cm):
        table.columns[index].width = Cm(width)
        for row in table.rows:
            row.cells[index].width = Cm(width)

    align = align or ["center"] * len(header)
    for col, text in enumerate(header):
        cell = table.rows[0].cells[col]
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if align[col] == "center" else WD_ALIGN_PARAGRAPH.LEFT
        paragraph.paragraph_format.space_before = Pt(2)
        paragraph.paragraph_format.space_after = Pt(2)
        runner = paragraph.add_run(text)
        runner.bold = True
        runner.font.size = Pt(font_size)

    for row_index, values in enumerate(rows, start=1):
        for col, text in enumerate(values):
            cell = table.rows[row_index].cells[col]
            cell.text = ""
            paragraph = cell.paragraphs[0]
            left = (col == 0 and first_col_left)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if left else WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_before = Pt(1)
            paragraph.paragraph_format.space_after = Pt(1)
            runner = paragraph.add_run(str(text))
            runner.font.size = Pt(font_size)

    # Three-line (booktabs) borders: rule above, below header, and at the foot.
    for row in table.rows:
        for cell in row.cells:
            _set_cell_border(cell, left={"val": "nil"}, right={"val": "nil"},
                             top={"val": "nil"}, bottom={"val": "nil"})
    for cell in table.rows[0].cells:
        _set_cell_border(cell, top={"sz": 8}, bottom={"sz": 6})
    for cell in table.rows[-1].cells:
        _set_cell_border(cell, bottom={"sz": 8})

    # Keep rows from splitting across pages, and keep the table body with its head.
    for row_index, row in enumerate(table.rows):
        tr_pr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.append(cant_split)
        if row_index < len(table.rows) - 1:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.keep_with_next = True
    return table


def add_figure(doc, path, width_cm: float, caption: str):
    """Insert a figure and its caption inside one borderless cell.

    Keeping both in a single non-splitting table row stops Word from leaving the
    picture at the foot of a page while the caption moves to the next page.
    """
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Cm(17.1)
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "nil")
        borders.append(element)
    tbl_pr.append(borders)
    margins = OxmlElement("w:tblCellMar")
    for edge in ("top", "left", "bottom", "right"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:w"), "0")
        element.set(qn("w:type"), "dxa")
        margins.append(element)
    tbl_pr.append(margins)
    row = table.rows[0]
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))
    cell = row.cells[0]
    cell.width = Cm(17.1)

    picture_paragraph = cell.paragraphs[0]
    picture_paragraph.style = doc.styles["SPIE figure"]
    picture_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture_paragraph.paragraph_format.space_before = Pt(6)
    picture_paragraph.paragraph_format.space_after = Pt(2)
    picture_paragraph.add_run().add_picture(str(path), width=Cm(width_cm))

    cap = cell.add_paragraph(style="SPIE figure caption")
    cap.paragraph_format.space_after = Pt(8)
    cap.add_run(caption)
    return cap


def add_body(doc, text: str, style: str = "SPIE body text", size: float | None = None):
    """Add a paragraph with light inline markup.

    [[n]] or [[n,m]]   superscript bracketed citation
    $...$              italic symbol
    _{...}             subscript
    ^{...}             superscript
    """
    import re

    paragraph = doc.add_paragraph(style=style)
    paragraph.paragraph_format.space_after = Pt(4)
    pattern = re.compile(r"\[\[([0-9,\u2013\- ]+)\]\]|\$([^$]*)\$|_\{([^}]*)\}|\^\{([^}]*)\}")
    position = 0

    def emit(content: str, italic=False, superscript=False, subscript=False, bracketed=False):
        if not content:
            return
        runner = paragraph.add_run(content)
        if italic:
            runner.italic = True
        if superscript:
            runner.font.superscript = True
        if subscript:
            runner.font.subscript = True
        if size:
            runner.font.size = Pt(size)

    for match in pattern.finditer(text):
        if match.start() > position:
            emit(text[position:match.start()])
        if match.group(1) is not None:
            emit("[" + match.group(1) + "]", superscript=True)
        elif match.group(2) is not None:
            emit(match.group(2), italic=True)
        elif match.group(3) is not None:
            emit(match.group(3), subscript=True)
        else:
            emit(match.group(4), superscript=True)
        position = match.end()
    if position < len(text):
        emit(text[position:])
    return paragraph


def add_plain(doc, text: str, style: str, size: float | None = None, bold: bool = False,
              italic: bool = False, align=None):
    paragraph = doc.add_paragraph(style=style)
    if align is not None:
        paragraph.alignment = align
    runner = paragraph.add_run(text)
    runner.bold = bold or None
    runner.italic = italic or None
    if size:
        runner.font.size = Pt(size)
    return paragraph
