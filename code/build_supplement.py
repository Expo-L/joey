"""Build the supplementary material for the 7-page manuscript."""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt
from docx.oxml.ns import qn

REVISION = Path(__file__).resolve().parent
sys.path.insert(0, str(REVISION))

import docxkit as dk  # noqa: E402

OUT = REVISION / "supplementary_7p.docx"


def build() -> None:
    doc = Document(str(REVISION / "template" / "SPIE_A4_template.docx"))
    body_el = doc.element.body
    for child in list(body_el):
        if child.tag == qn("w:sectPr"):
            continue
        body_el.remove(child)
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(1.93)
    section.right_margin = Cm(1.93)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.0

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Supplementary material")
    run.bold = True
    run.font.size = Pt(14)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub.add_run("Brand-constrained diffusion for broadcast packaging with a late-window temporal residual")
    sub_run.italic = True
    sub_run.font.size = Pt(10)

    dk.add_body(doc, "This supplement holds the configuration tables and the full numeric results that were "
                     "condensed in the main paper. Nothing here changes a claim in the main text. Table numbering "
                     "follows the supplementary series S1 to S5.", "Normal")

    doc.add_paragraph().add_run("S1. Temporal residual configuration").bold = True
    dk.add_table(
        doc,
        "Table S1. Layer configuration of the temporal residual. Each gated call evaluates the module once on the "
        "full 16 frame latent sequence.",
        ["Layer", "Configuration", "Output shape", "Parameters"],
        [["Conv3d (in)", "4 in, 16 out, kernel 3x3x3, padding 1", "16x16x32x32", "1,744"],
         ["GroupNorm", "4 groups, 16 channels", "16x16x32x32", "32"],
         ["Time MLP", "input 1, hidden 32, output 32, SiLU", "2 x 16 (scale, shift)", "1,120"],
         ["Scale and shift", "features times (1 + scale) plus shift", "16x16x32x32", "0"],
         ["Conv3d (out)", "16 in, 4 out, kernel 3x1x1, padding (1,0,0)", "16x4x32x32", "196"],
         ["Total", "", "", "3,092"]],
        [3.4, 6.4, 4.0, 2.4])

    doc.add_paragraph().add_run("S2. Conditions").bold = True
    dk.add_table(
        doc,
        "Table S2. All conditions used in the study. The locked setting is a deterministic bound and not a "
        "competing generator.",
        ["Condition", "Adapter", "Residual", "Role"],
        [["No LoRA", "off", "none", "Baseline without style adaptation"],
         ["LoRA", "on", "none", "Main baseline for the gate comparison"],
         ["Locked", "on", "background held fixed", "Deterministic bound only"],
         ["Full injection", "on", "all 11 calls", "Unrestricted residual use"],
         ["Pre-registered gate", "on", "last 3 calls", "Setting carried over from the submission"],
         ["Wide gate", "on", "last 8 calls", "Sweep optimum on this split"],
         ["No adapter, wide gate", "off", "last 8 calls", "Residual without style adaptation"],
         ["LoRA, no Canny", "on", "none", "Layout condition removed"]],
        [4.2, 1.8, 4.2, 7.0],
        align=["left", "center", "center", "left"])

    doc.add_paragraph().add_run("S3. Full gating schedule").bold = True
    dk.add_table(
        doc,
        "Table S3. Every tested value of K. The change column is baseline minus treatment, so a positive value "
        "means lower jitter with the residual. The p column is unadjusted.",
        ["Gated calls", "Background jitter", "Change", "95% CI", "p", "Pairs agreeing", "Relative"],
        [["0", "0.2676 \u00b1 0.0100", "\u2014", "\u2014", "\u2014", "\u2014", "\u2014"],
         ["1", "0.2664 \u00b1 0.0103", "+0.00116", "0.00012 to 0.00219", "0.039", "17/30", "+0.44%"],
         ["2", "0.2669 \u00b1 0.0103", "+0.00070", "\u22120.00033 to 0.00168", "0.191", "18/30", "+0.26%"],
         ["3", "0.2653 \u00b1 0.0105", "+0.00226", "0.00118 to 0.00331", "3.7 x 1e-4", "22/30", "+0.84%"],
         ["4", "0.2651 \u00b1 0.0112", "+0.00245", "0.00125 to 0.00367", "6.3 x 1e-4", "22/30", "+0.91%"],
         ["6", "0.2614 \u00b1 0.0108", "+0.00618", "0.00435 to 0.00795", "1.0 x 1e-5", "27/30", "+2.31%"],
         ["8", "0.2597 \u00b1 0.0114", "+0.00790", "0.00530 to 0.01062", "1.0 x 1e-5", "26/30", "+2.95%"],
         ["11", "0.2786 \u00b1 0.0088", "\u22120.01106", "\u22120.01361 to \u22120.00856", "1.0 x 1e-5", "2/30",
          "\u22124.13%"]],
        [2.4, 3.0, 2.5, 4.3, 2.1, 2.0, 1.8],
        font_size=8.5)

    doc.add_paragraph().add_run("S4. Alternative measures").bold = True
    dk.add_table(
        doc,
        "Table S4. Alternative measures for the pre-registered gate against independent sampling. SSIM is higher "
        "when better, the other rows are lower when better, and differences are oriented so that a positive value "
        "favours the gate.",
        ["Measure", "Independent", "Gate, K = 3", "Difference", "95% CI", "p"],
        [["Background jitter", "0.2676", "0.2653", "+0.00226", "0.00118 to 0.00331", "3.7e-4"],
         ["Adjacent frame SSIM", "0.2423", "0.2548", "+0.01257", "0.00972 to 0.01550", "1.0e-5"],
         ["High pass jitter", "0.05785", "0.05721", "+0.00064", "0.00011 to 0.00117", "0.029"],
         ["Worst frame pairs", "0.3425", "0.3411", "+0.00146", "\u22120.00050 to 0.00324", "0.146"],
         ["Static region", "0.2635", "0.2612", "+0.00228", "0.00122 to 0.00330", "3.2e-4"],
         ["Moving region", "0.3215", "0.3186", "+0.00281", "0.000003 to 0.00563", "0.063"]],
        [4.0, 2.5, 2.4, 2.6, 3.4, 2.1],
        font_size=8.5)

    doc.add_paragraph().add_run("S5. Reproduction notes").bold = True
    dk.add_body(doc,
                "All numbers come from four runs. The formal run holds 150 sequences, ten held-out assets by three "
                "seeds by five conditions. The schedule sweep adds 180 sequences over six K values. The attribution "
                "run adds 60 sequences for two conditions. The static region analysis reuses the stored frames. "
                "Every condition shares the same three seeds, initial noise, Canny condition and prompt, and the "
                "unit of analysis is the asset and seed pair, so n = 30. Bootstrap intervals use 5,000 resamples, "
                "sign-flip tests use 100,000 draws and the resampling seed is 20260903. The sweep reproduces the "
                "single sample value for K = 3 exactly, 0.2649148503939311, and the offline metric pipeline "
                "reproduces the stored jitter to within 2.5e-7. Removing the Canny condition makes the output "
                "independent of the asset, which we confirmed by comparing stored frames across assets, so that "
                "condition is reported descriptively.", "Normal")

    doc.save(str(OUT))
    print("saved", OUT)


if __name__ == "__main__":
    build()
