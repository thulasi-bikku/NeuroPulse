#!/usr/bin/env python
"""Build a Word (.docx) version of the manuscript from the LaTeX source.

The LaTeX build supplies the numbering (sections, figures, tables, equations and
citations) through its .aux file, so the Word document is numbered exactly like
the PDF. Equations become native Word equations.

Requirements: pandoc >= 3.1, python-docx, a TeX Live installation (to produce the .aux/.bbl).

Usage (repository root):
    cd paper && latexmk -pdf NeuroPulse_manuscript.tex && cd ..
    python scripts/make_docx.py
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
STEM = "NeuroPulse_manuscript"
OUT = PAPER / f"{STEM}.docx"
FONT = "Times New Roman"
TEXT_WIDTH_CM = 16.0

TITLE = "NeuroPulse: A CNN\u2013Attention Web Prototype for Five-Class Classification of Synthetic EEG-Inspired Features"
AUTHORS = "Jeevana Jyothi\u00b9 and Thulasi Bikku\u00b2"


# ----------------------------------------------------------------------------- LaTeX preprocessing
def balanced(text, start):
    """Return the index just past the brace group that opens at text[start] == '{'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced braces")


def compress(nums):
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            out.append(f"{nums[i]}\u2013{nums[j]}")
        else:
            out.extend(str(n) for n in nums[i:j + 1])
        i = j + 1
    return ", ".join(out)


def preprocess():
    tex = (PAPER / f"{STEM}.tex").read_text(encoding="utf-8")
    aux = (PAPER / f"{STEM}.aux").read_text(encoding="utf-8")
    bbl = (PAPER / f"{STEM}.bbl").read_text(encoding="utf-8")
    cites = dict(re.findall(r"\\bibcite\{([^}]+)\}\{(\d+)\}", aux))
    labels = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux))

    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1).replace("\\noindent", "").strip()
    keywords = re.search(r"\\noindent\\textbf\{Keywords:\}(.*)", tex).group(1).strip()
    affils = re.findall(r"\\affil\[(\d)\]\{(.*)\}", tex)
    body = tex[tex.index(keywords) + len(keywords):tex.index("\\bibliographystyle")]

    # layout-only commands
    for cmd in (r"\\FloatBarrier", r"\\centering", r"\\small", r"\\footnotesize"):
        body = re.sub(cmd + r"\b", "", body)
    body = re.sub(r"\\vspace\{[^}]*\}", "", body)

    # numbered headings
    counters = [0, 0]

    def heading(m):
        level, star, title = m.group(1), m.group(2), m.group(3)
        if star:
            return f"\\section*{{{title}}}"
        if level == "section":
            counters[0] += 1
            counters[1] = 0
            return f"\\section{{{counters[0]}\\ \\ {title}}}"
        counters[1] += 1
        return f"\\subsection{{{counters[0]}.{counters[1]}\\ \\ {title}}}"

    body = re.sub(r"\\(section|subsection)(\*?)\{([^}]*)\}", heading, body)

    # captions: prefix "Figure N." / "Table N."
    counts = {"figure": 0, "table": 0}

    def env(m):
        kind = m.group(1)
        counts[kind] += 1
        label = "Figure" if kind == "figure" else "Table"
        return m.group(0).replace("\\caption{", f"\\caption{{\\textbf{{{label} {counts[kind]}.}} ", 1)

    body = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", env, body, flags=re.S)

    # multi-line align blocks -> one display equation per line (the converter drops align environments)
    def align(m):
        rows = [r.strip() for r in re.split(r"\\\\", m.group(1)) if r.strip()]
        return "\n".join("\\begin{equation}" + r.replace("&", "") + "\\end{equation}" for r in rows)

    body = re.sub(r"\\begin\{align\}(.*?)\\end\{align\}", align, body, flags=re.S)

    # equation labels -> visible numbers
    def eqlabel(m):
        return f"\\qquad ({labels[m.group(1)]})"

    body = re.sub(r"\\label\{(eq:[^}]+)\}", eqlabel, body)
    body = re.sub(r"\\label\{[^}]+\}", "", body)

    # cross-references and citations
    body = re.sub(r"\\eqref\{([^}]+)\}", lambda m: f"({labels[m.group(1)]})", body)
    body = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels[m.group(1)], body)
    body = re.sub(r"[ \t]+\\cite\{", "~\\\\cite{", body)
    body = re.sub(r"\\cite\{([^}]+)\}",
                  lambda m: "[" + compress([int(cites[k.strip()]) for k in m.group(1).split(",")]) + "]", body)

    # tabularx -> tabular with plain column letters
    def tabx(m):
        spec = m.group(1)
        spec = re.sub(r">\{[^}]*(\{[^}]*\})?[^}]*\}", "", spec)
        n = len(re.findall(r"[lcrXp]", re.sub(r"p\{[^}]*\}", "p", spec)))
        return "\\begin{tabular}{" + "l" * n + "}"

    body = re.sub(r"\\begin\{tabularx\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}", tabx, body)
    body = body.replace("\\end{tabularx}", "\\end{tabular}")

    # figures: PNG at 600 dpi instead of the PDF used by LaTeX
    body = re.sub(r"\{(fig_[a-z_]+)\.pdf\}", r"{\1.png}", body)

    # reference list from the .bbl
    items = re.findall(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem\{|\\end\{thebibliography\})", bbl, re.S)
    refs = []
    for key, text in items:
        text = text.replace("\\newblock", " ")
        text = re.sub(r"\{\\em\s+([^{}]*)\}", r"\\emph{\1}", text)
        text = re.sub(r"\s+", " ", text).strip()
        refs.append((int(cites[key]), f"[{cites[key]}]\\ \\ {text}"))
    refs.sort()

    front = "".join(f"\\noindent {n.replace('1', '\u00b9').replace('2', '\u00b2')}{t}\n\n" for n, t in affils)
    doc = ("\\documentclass{article}\\begin{document}\n" + front
           + "\\section*{Abstract}\n" + abstract + "\n\n"
           + "\\noindent\\textbf{Keywords:} " + keywords + "\n\n"
           + body + "\n\\section*{References}\n\n" + "\n\n".join(t for _, t in refs) + "\n\\end{document}\n")
    return doc, counters


# ------------------------------------------------------------------------------ Word formatting
def set_font(style_or_run, size=None, bold=None, italic=None, color=RGBColor(0, 0, 0), name=FONT):
    f = style_or_run.font
    f.name = name
    rpr = style_or_run.element.get_or_add_rPr() if hasattr(style_or_run.element, "get_or_add_rPr") else None
    if rpr is not None:
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.insert(0, rfonts)
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rfonts.set(qn(attr), name)
        for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if rfonts.get(qn(attr)) is not None:
                del rfonts.attrib[qn(attr)]
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color is not None:
        f.color.rgb = color


def para_fmt(style, align=None, before=0, after=0, line=None, keep_next=None, first=None, left=None):
    pf = style.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    if line is not None:
        pf.line_spacing = line
    if align is not None:
        pf.alignment = align
    if keep_next is not None:
        pf.keep_with_next = keep_next
    if first is not None:
        pf.first_line_indent = first
    if left is not None:
        pf.left_indent = left


def borders(pr, **edges):
    """Set borders on a tblPr/tcPr element, inserting the element in schema order.

    edges: top/bottom/... = size in eighths of a point, or None for no border.
    """
    is_table = pr.tag == qn("w:tblPr")
    tag = "w:tblBorders" if is_table else "w:tcBorders"
    old = pr.find(qn(tag))
    if old is not None:
        pr.remove(old)
    b = OxmlElement(tag)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        if not is_table and edge in ("insideH", "insideV"):
            continue
        el = OxmlElement(f"w:{edge}")
        spec = edges.get(edge)
        if spec:
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(spec))
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "nil")
        b.append(el)
    successors = (["shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription"] if is_table
                  else ["shd", "noWrap", "tcMar", "textDirection", "tcFitText", "vAlign", "hideMark"])
    for name in successors:
        nxt = pr.find(qn(f"w:{name}"))
        if nxt is not None:
            nxt.addprevious(b)
            return
    pr.append(b)


def add_page_number(section):
    p = section.footer.paragraphs[0] if section.footer.paragraphs else section.footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)
    set_font(run, size=10)


def format_document(path):
    doc = Document(path)
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.PORTRAIT
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2.5)
    sec.header_distance = sec.footer_distance = Cm(1.25)
    sec.gutter = Cm(0)
    add_page_number(sec)

    st = {s.name: s for s in doc.styles}  # pandoc's names differ in case from python-docx's built-in lookup
    for s in st.values():
        if s.type == 1 and s.font is not None:  # paragraph styles: unify typeface and remove theme colors
            set_font(s)
    for name in ("Normal", "Body Text", "First Paragraph"):
        if name in st:
            set_font(st[name], size=12)
            para_fmt(st[name], WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 6, 1.5)
    if "Compact" in st:
        set_font(st["Compact"], size=11)
        para_fmt(st["Compact"], WD_ALIGN_PARAGRAPH.LEFT, 0, 2, 1.15)
    set_font(st["Title"], size=17, bold=True)
    para_fmt(st["Title"], WD_ALIGN_PARAGRAPH.CENTER, 0, 12, 1.15)
    for name in ("Author", "Date"):
        if name in st:
            set_font(st[name], size=13)
            para_fmt(st[name], WD_ALIGN_PARAGRAPH.CENTER, 0, 8, 1.15)
    for name, size, italic in (("Heading 1", 14, False), ("Heading 2", 12.5, False), ("Heading 3", 12, True)):
        set_font(st[name], size=size, bold=True, italic=italic)
        para_fmt(st[name], WD_ALIGN_PARAGRAPH.LEFT, 16 if name == "Heading 1" else 10, 6, 1.15, keep_next=True)
    for name in ("Image Caption", "Table Caption", "Caption"):
        if name in st:
            set_font(st[name], size=10, italic=False)
            para_fmt(st[name], WD_ALIGN_PARAGRAPH.JUSTIFY, 4, 10, 1.0, keep_next=(name == "Table Caption"))
    if "Captioned Figure" in st:
        para_fmt(st["Captioned Figure"], WD_ALIGN_PARAGRAPH.CENTER, 6, 0, 1.0, keep_next=True)
    if "Figure" in st:
        para_fmt(st["Figure"], WD_ALIGN_PARAGRAPH.CENTER, 6, 0, 1.0, keep_next=True)
    if "Source Code" in st:
        set_font(st["Source Code"], size=9, name="Courier New")
        para_fmt(st["Source Code"], WD_ALIGN_PARAGRAPH.LEFT, 0, 6, 1.0)
    if "Verbatim Char" in st:
        set_font(st["Verbatim Char"], size=10, name="Courier New")

    body = doc.element.body
    paragraphs = list(doc.paragraphs)
    in_refs = False
    for p in paragraphs:
        text = p.text.strip()
        if p.style.name.startswith("Heading"):
            in_refs = text == "References"
            continue
        if text[:1] in ("\u00b9", "\u00b2"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(3)
            for r in p.runs:
                r.italic = True
                r.font.size = Pt(10.5)
        elif in_refs and text.startswith("["):
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf = p.paragraph_format
            pf.left_indent, pf.first_line_indent = Cm(1.0), Cm(-1.0)
            pf.line_spacing, pf.space_after = 1.0, Pt(3)
            for r in p.runs:
                r.font.size = Pt(10)
        elif text.startswith("Keywords:"):
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif p._p.xpath(".//w:drawing") and not text:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True

    # tables: three-line style, compact text, proportional widths
    for tbl in doc.tables:
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        pr = tbl._tbl.tblPr
        borders(pr, top=12, bottom=12)
        rows = tbl.rows
        ncols = len(rows[0].cells)
        lens = [max(len(r.cells[j].text) for r in rows) for j in range(ncols)]
        weights = [min(max(n, 7), 48) for n in lens]
        longest = [max(len(w) for r in rows for w in re.split(r"[ \t\n]+", r.cells[j].text) if w) for j in range(ncols)]
        mins = [max(1.4, 0.19 * n + 0.45) for n in longest]
        spare = max(TEXT_WIDTH_CM - sum(mins), 0)
        widths = [m + spare * w / sum(weights) for m, w in zip(mins, weights)]
        scale = TEXT_WIDTH_CM / sum(widths)
        widths = [w * scale for w in widths]
        for i, row in enumerate(rows):
            for j, cell in enumerate(row.cells):
                cell.width = Cm(widths[j])
                tcpr = cell._tc.get_or_add_tcPr()
                borders(tcpr, bottom=6 if i == 0 else None)
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.space_after = Pt(2)
                    p.paragraph_format.space_before = Pt(2)
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for r in p.runs:
                        r.font.size = Pt(9.5)
                        r.font.name = FONT
                        if i == 0:
                            r.bold = True
        grid = tbl._tbl.tblGrid
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(w / 2.54 * 1440)))

    # images no wider than the text block
    for shape in doc.inline_shapes:
        max_w = Cm(TEXT_WIDTH_CM)
        if shape.width > max_w:
            ratio = max_w / shape.width
            shape.width, shape.height = max_w, int(shape.height * ratio)

    cp = doc.core_properties
    cp.title, cp.author, cp.subject = TITLE, "Jeevana Jyothi; Thulasi Bikku", "Manuscript"
    cp.keywords = "affective computing; EEG-inspired features; synthetic data; CNN; self-attention; Streamlit"
    doc.save(path)


# ----------------------------------------------------------------------------- package repair
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
STYLE_ORDER = ["name", "aliases", "basedOn", "next", "link", "autoRedefine", "hidden", "uiPriority", "semiHidden",
               "unhideWhenUsed", "qFormat", "locked", "personal", "personalCompose", "personalReply", "rsid",
               "pPr", "rPr", "tblPr", "trPr", "tcPr", "tblStylePr"]
TCPR_ORDER = ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap", "tcMar",
              "textDirection", "tcFitText", "vAlign", "hideMark"]
PPR_ORDER = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
             "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
             "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd", "snapToGrid",
             "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc", "textDirection",
             "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr", "pPrChange"]


SETTINGS_ORDER = [
    "writeProtection", "view", "zoom", "removePersonalInformation", "removeDateAndTime", "doNotDisplayPageBoundaries",
    "displayBackgroundShape", "printPostScriptOverText", "printFractionalCharacterWidth", "printFormsData",
    "embedTrueTypeFonts", "embedSystemFonts", "saveSubsetFonts", "saveFormsData", "mirrorMargins",
    "alignBordersAndEdges", "bordersDoNotSurroundHeader", "bordersDoNotSurroundFooter", "gutterAtTop",
    "hideSpellingErrors", "hideGrammaticalErrors", "activeWritingStyle", "proofState", "formsDesign",
    "attachedTemplate", "linkStyles", "stylePaneFormatFilter", "stylePaneSortMethod", "documentType", "mailMerge",
    "revisionView", "trackRevisions", "doNotTrackMoves", "doNotTrackFormatting", "documentProtection",
    "autoFormatOverride", "styleLockTheme", "styleLockQFSet", "defaultTabStop", "autoHyphenation",
    "consecutiveHyphenLimit", "hyphenationZone", "doNotHyphenateCaps", "showEnvelope", "summaryLength",
    "clickAndTypeStyle", "defaultTableStyle", "evenAndOddHeaders", "bookFoldRevPrinting", "bookFoldPrinting",
    "bookFoldPrintingSheets", "drawingGridHorizontalSpacing", "drawingGridVerticalSpacing",
    "displayHorizontalDrawingGridEvery", "displayVerticalDrawingGridEvery", "doNotUseMarginsForDrawingGridOrigin",
    "drawingGridHorizontalOrigin", "drawingGridVerticalOrigin", "doNotShadeFormData", "noPunctuationKerning",
    "characterSpacingControl", "printTwoOnOne", "strictFirstAndLastChars", "noLineBreaksAfter", "noLineBreaksBefore",
    "savePreviewPicture", "doNotValidateAgainstSchema", "saveInvalidXml", "ignoreMixedContent",
    "alwaysShowPlaceholderText", "doNotDemarcateInvalidXml", "saveXmlDataOnly", "useXSLTWhenSaving",
    "saveThroughXslt", "showXMLTags", "alwaysMergeEmptyNamespace", "updateFields", "hdrShapeDefaults", "footnotePr",
    "endnotePr", "compat", "docVars", "rsids", "mathPr", "attachedSchema", "themeFontLang", "clrSchemeMapping",
    "doNotIncludeSubdocsInStats", "doNotAutoCompressPictures", "forceUpgrade", "captions", "readModeInkLockDown",
    "smartTagType", "schemaLibrary", "shapeDefaults", "doNotEmbedSmartTags", "decimalSymbol", "listSeparator"]


def reorder(parent, order, ns):
    rank = {n: i for i, n in enumerate(order)}
    kids = list(parent)
    kids_sorted = sorted(kids, key=lambda k: rank.get(k.tag.rsplit("}", 1)[-1], len(order)))
    if kids != kids_sorted:
        for k in kids:
            parent.remove(k)
        for k in kids_sorted:
            parent.append(k)


def repair_package(path):
    """Fix element-ordering quirks of the converter output so the package passes strict OOXML validation."""
    import zipfile
    from lxml import etree

    tmp = Path(str(path) + ".tmp")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        names = [n for n in zin.namelist() if not n.endswith("/")]
        names.sort(key=lambda n: (n != "[Content_Types].xml", n))
        for name in names:
            data = zin.read(name)
            if name in ("word/styles.xml", "word/document.xml", "word/numbering.xml", "word/settings.xml"):
                root = etree.fromstring(data)
                if name == "word/styles.xml":
                    for el in root.iter(f"{{{W}}}style"):
                        reorder(el, STYLE_ORDER, W)
                    for el in root.iter(f"{{{W}}}tcPr"):
                        reorder(el, TCPR_ORDER, W)
                    for el in root.iter(f"{{{W}}}pPr"):
                        reorder(el, PPR_ORDER, W)
                if name == "word/document.xml":
                    for el in root.iter(f"{{{M}}}rPr"):
                        reorder(el, ["lit", "nor", "scr", "sty", "brk", "aln"], M)
                    for el in root.iter(f"{{{M}}}dPr"):
                        reorder(el, ["begChr", "sepChr", "endChr", "grow", "shp", "ctrlPr"], M)
                    for el in root.iter(f"{{{W}}}tcPr"):
                        reorder(el, TCPR_ORDER, W)
                    for el in root.iter(f"{{{W}}}pPr"):
                        reorder(el, PPR_ORDER, W)
                if name == "word/numbering.xml":
                    for el in root.iter(f"{{{W}}}nsid", f"{{{W}}}tmpl"):
                        v = el.get(f"{{{W}}}val")
                        if v and len(v) < 8:
                            el.set(f"{{{W}}}val", v.zfill(8))
                if name == "word/settings.xml":
                    reorder(root, SETTINGS_ORDER, W)
                data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            zout.writestr(name, data)
    tmp.replace(path)


def main():
    for ext in ("aux", "bbl", "tex"):
        if not (PAPER / f"{STEM}.{ext}").exists():
            sys.exit(f"missing paper/{STEM}.{ext}; run latexmk in paper/ first")
    latex, counters = preprocess()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "manuscript_for_word.tex"
        src.write_text(latex, encoding="utf-8")
        cmd = ["pandoc", str(src), "-f", "latex+smart", "-t", "docx", "-o", str(OUT),
               "--resource-path", str(PAPER / "figures"), "-M", f"title={TITLE}", "-M", f"author={AUTHORS}"]
        subprocess.run(cmd, check=True)
        (PAPER / "word_source_debug.tex").write_text(latex, encoding="utf-8") if "--debug" in sys.argv else None
    format_document(OUT)
    repair_package(OUT)
    print(f"wrote {OUT}  ({counters[0]} numbered sections)")


if __name__ == "__main__":
    main()
