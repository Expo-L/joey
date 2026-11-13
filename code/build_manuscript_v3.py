"""Build the 7-page camera-ready version of the manuscript.

Same claims and numbers as the 11-page revision, with the prose condensed and
two tables moved to the supplementary material.
"""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[2]
REVISION = Path(__file__).resolve().parent
sys.path.insert(0, str(REVISION))

import docxkit as dk  # noqa: E402
from docxkit import eqarr, op, rad, run, subs, subsups, sups, var  # noqa: E402

TEMPLATE = REVISION / "template" / "SPIE_A4_template.docx"
FIGURES = ROOT / "manuscript_q4" / "figures"
OUT_DOCX = REVISION / "manuscript_7p.docx"
NB_HYPHEN = "\u2011"


def clear_body(doc):
    body = doc.element.body
    for child in list(body):
        if child.tag == qn("w:sectPr"):
            continue
        body.remove(child)


def heading(doc, text, level=1):
    paragraph = doc.add_paragraph(style="Heading 1" if level == 1 else "Heading 2")
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run(text)
    return paragraph


def body(doc, text):
    return dk.add_body(doc, text, "SPIE body text")


def build():
    doc = Document(str(TEMPLATE))
    clear_body(doc)

    title = doc.add_paragraph(style="SPIE paper title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run(f"Brand{NB_HYPHEN}constrained diffusion for broadcast packaging "
                  f"with a late{NB_HYPHEN}window temporal residual")

    authors = doc.add_paragraph(style="SPIE Authors-Affils")
    authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for name, mark, sep in [("Qiaoyi Zhang", "*a", ", "), ("Shuran Yang", "a", "")]:
        authors.add_run(name)
        sup = authors.add_run(mark)
        sup.font.superscript = True
        if sep:
            authors.add_run(sep)

    affil = doc.add_paragraph(style="SPIE Authors-Affils")
    affil.alignment = WD_ALIGN_PARAGRAPH.CENTER
    lead = affil.add_run("a")
    lead.font.superscript = True
    affil.add_run("Chongqing University of Posts and Telecommunications, Chongqing 400065, China")

    email = doc.add_paragraph(style="*Body of Paper*")
    email.alignment = WD_ALIGN_PARAGRAPH.CENTER
    email.paragraph_format.space_after = Pt(0)
    runner = email.add_run("*Corresponding author: 18717009433@163.com; suzy0323000@qq.com")
    runner.font.size = Pt(10)

    doc.add_paragraph(style="SPIE abstract title").add_run("Abstract")

    abstract = (
        "Broadcast packaging needs a recognisable visual field and also a channel mark, a type system and a safe "
        "area that an editor can still control. We give these jobs to different parts of a small workflow. A frozen "
        "Stable Diffusion v1.5 backbone draws the background, a rank-4 adapter and a Canny edge map carry brand and "
        "layout cues, a deterministic compositor places the editable logo and type layers, and a residual of 3,092 "
        "parameters is added during some denoising calls. The question is not whether to constrain a sequence over "
        "time but which part of the denoising trajectory should carry the constraint. We replaced a timestep "
        "threshold with the last $K$ of the $N$ observed calls and tested $K$ equal to 1, 2, 3, 4, 6, 8 and 11 on "
        "150 held-out synthetic sequences. The adapter reduced mean background jitter from 0.3964 to 0.2676, a fall "
        "of 32.51%. Gating all eleven calls raised jitter to 0.2786, which is 4.13% worse than sampling without the "
        "residual. Later windows helped, and the gain grew with the window width. The pre-registered three-call "
        "setting gave 0.84%, six calls 2.31% and eight calls 2.95%. The decisive boundary is the first and noisiest "
        "calls, and gating them reverses the gain. Alternative measures reproduce the effect, and a static-region "
        "analysis locates it where the motion plan asks for no change, so we read it as a reduction of unwanted "
        "change and not as better motion. All findings are limited to one synthetic source group, 256 × 256 "
        "rendering, ten requested denoising steps and procedural motion."
    )
    dk.add_body(doc, abstract, "SPIE abstract body text")

    keywords = doc.add_paragraph(style="*Keywords*")
    keywords.alignment = WD_ALIGN_PARAGRAPH.LEFT
    keywords.paragraph_format.left_indent = Cm(0)
    keywords.paragraph_format.first_line_indent = Cm(0)
    label = keywords.add_run("Keywords:")
    label.bold = True
    keywords.add_run(" latent diffusion; broadcast graphics; controllable generation; temporal consistency; "
                     "low-rank adaptation; reproducible evaluation")

    heading(doc, "INTRODUCTION")
    body(doc,
         "Programme packaging is a visual contract that repeats between a programme, its channel and its audience. "
         "The palette, the type treatment, the motion rhythm and the spatial hierarchy have to stay legible across "
         "idents, title sequences and short promotional units. Television scholarship treats programmes as objects "
         "that carry a brand,[[17]] and channel branding as a practice that is maintained over time.[[18]] A good "
         "single frame is therefore not enough. The editor still has to move the title, keep a safe area, replace a "
         "logo and repeat a motion decision on the next delivery.")
    body(doc,
         "Diffusion models made high quality synthesis practical, and latent diffusion lowered the cost by working "
         "in a compressed space with cross-attention conditioning.[[1]] Both properties suit packaging work, since "
         "the background sets atmosphere, material, colour and rhythm. A text prompt, however, is a poor way to "
         "specify the parts of broadcast design that are defined geometrically. ControlNet showed that a frozen "
         "backbone can take an edge map,[[3]] and low-rank adaptation gives a compact way to bias appearance without "
         "retraining every weight.[[2]]")
    body(doc,
         "Temporal generation adds a second problem. Frames sampled independently fluctuate even when the intended "
         "scene is stable. Video-diffusion systems deal with this through video-scale training,[[7,8]] and lighter "
         "methods change the sampling procedure or reuse an image prior.[[10\u201313]] They share one lesson. A "
         "system has to decide when and where a temporal constraint may influence generation, because strong "
         "coupling removes colour, texture or framing changes that the designer wanted.")
    body(doc,
         "This paper asks which part of the denoising trajectory should carry a light temporal residual. We treat "
         "late as the last $K$ of the $N$ observed calls and test $K$ on one held-out split. The answer changes the "
         "practical rule, because the boundary that matters is not how many calls are gated but whether the "
         "noisiest calls are touched at all. We report every asset and seed pair, with effect sizes and bootstrap "
         "intervals, and we separate the contribution of the adapter, the edge condition, the residual and the "
         "compositor.")

    heading(doc, "RELATED WORK")
    body(doc,
         "Latent diffusion denoises in a learned compressed space.[[1]] Control methods attach a spatial condition "
         "to a frozen backbone, and recent surveys organise the design space.[[4]] Low-rank adaptation learns a "
         "low-rank update instead of retraining the model.[[2]] Later work measures what this costs in capacity and "
         "forgetting,[[5]] and weight-decomposed variants refine the update rule.[[6]] We use the adapter only to "
         "bias appearance, at rank 4.")
    body(doc,
         "Video diffusion improved as training corpora and spatio-temporal backbones grew.[[7\u20139]] A second line "
         "keeps a text-to-image model and changes how it is sampled. Motion modules,[[10]] noise rescheduling,[[11]] "
         "deformation fields[[12]] and keyframe propagation[[13]] obtain short range consistency without video-scale "
         "training. Our module follows that line and is simpler than any of them. It has no attention, no optical "
         "flow and no explicit correspondence between frames.")
    body(doc,
         "Video evaluation has relied on distribution distances such as FVD, which need a matched reference "
         "distribution. Recent benchmarks split quality into human-aligned dimensions,[[14]] measure metamorphic "
         "coherence,[[15]] and show that FVD-type features are not sensitive to temporal distortion.[[16]] Work on "
         "channel idents shows how design, sound and brand strategy meet in packaging,[[17,18]] while generative "
         "design systems separate background synthesis from layout or typography.[[19\u201321]] Motion graphics "
         "research has started to formalise checkable spatio-temporal properties.[[22,23]]")

    heading(doc, "METHOD")
    heading(doc, "System overview", 2)
    body(doc,
         "A brief gives the programme intent, the palette and the duration. A brand kit gives the logo and the type "
         "system. A spatial and motion plan gives an edge image, a safe zone and a frame by frame foreground "
         "trajectory. The diffusion path takes a fixed prompt, the Canny condition and the optional adapters, and "
         "returns a background sequence. A deterministic compositor then places the foreground layers, the alpha "
         "treatment and the trajectory. The boundary is deliberate, because the foreground can be replaced without "
         "running diffusion again and deterministic placement stays out of the quality ranking. The generator is a "
         "local Stable Diffusion v1.5 checkpoint. A rank-4 adapter is trained for 500 steps on 150 local assets and "
         "adds 797,184 parameters to the attention projections. A pretrained Canny ControlNet supplies layout at "
         "conditioning scale 1.0, and sampling runs at 256 × 256 for ten requested steps on one RTX 4070 Laptop "
         "GPU with 8 GB of memory (Figure 1).")
    dk.add_figure(doc, FIGURES / "fig1_method_overview.png", 9.4,
                  "Figure 1. Brand constrained diffusion workflow. The brief, brand kit and motion plan feed a frozen "
                  "latent diffusion backbone with a rank-4 adapter, a Canny condition and a timestep conditioned "
                  "residual. A deterministic compositor places the editable logo and type layers. The strip is the "
                  "fixed first held-out sample, commons_0150_v00 with seed 20262903.")

    heading(doc, "Temporal residual", 2)
    body(doc,
         "Let $z$_{t} be the latent sequence at timestep $t$ for sixteen frames, and let $R$_{\u03c6} be the "
         "adapter. Write \u03c4 = $t$/999 for the normalised timestep and $u$_{t} for the adapted latent. The "
         "residual is mixed in before the ControlNet and UNet are evaluated:")
    dk.add_equation(doc, [
        subs(var("u"), var("t")), op(" = "), subs(var("z"), var("t")), op(" + "), var("\u03b1"),
        var("g"), op("("), var("K"), op(")"),
        op("["), subs(var("R"), var("\u03c6")), op("("), subs(var("z"), var("t")), op(", "), var("\u03c4"),
        op(") \u2212 "), subs(var("z"), var("t")), op("]")
    ], 1)
    body(doc, "where \u03b1 = 0.10 and the gate selects the last $K$ of the $N$ = 11 calls.")
    dk.add_equation(doc, [
        var("g"), op("("), var("K"), op(") = "), run("1", "b"),
        op("["), var("k"), op(" > "), var("N"), op(" \u2212 "), var("K"), op("]"),
        op(",   "), var("k"), op(" = 1, \u2026, "), var("N")
    ], 2)
    body(doc,
         "The adapter holds two convolutions and one time-conditioning branch. The latent sequence enters as a "
         "tensor whose first axis is the frame index, and the forward pass is")
    dk.add_equation(doc, [
        eqarr([
            [var("h"), op(" = "), run("GN", "p"), op("("),
             subs(run("Conv3d", "p"), run("in", "p")), op("("), subs(var("z"), var("t")), op("))")],
            [var("r"), op(" = "),
             subs(run("Conv3d", "p"), run("out", "p")),
             op("("), run("SiLU", "p"), op("("), var("h"), op(" \u00d7 "), op("(1 + "), var("\u03b3"),
             op(") + "), var("\u03b2"), op("))")],
        ])
    ], 3)
    body(doc,
         "The input convolution spans the frame axis, so each frame gathers its two neighbours, and the output "
         "convolution mixes frames but not space. The residual at frame $i$ therefore depends on frames $i$ \u2212 2 "
         "to $i$ + 2, a five-frame receptive field, with 3 × 3 spatial support from the first layer. A two-layer "
         "perceptron maps \u03c4 to a scale and a shift, (\u03b3, \u03b2) = $W$_{2}SiLU($W$_{1}\u03c4). The output "
         "layer starts at zero, so an untrained adapter is an identity map. The module holds 3,092 parameters, and "
         "its measured relative residual is 4.2 to 4.7% of the latent norm per gated call.")
    body(doc,
         "The adapter never sees the UNet, the sampler or decoded pixels. Each step draws a timestep, noise that is "
         "independent for every frame, and one noise field shared across frames. The target is a latent sequence "
         "whose noise is correlated across frames at \u03c1 = 0.8:")
    dk.add_equation(doc, [
        subs(var("\u03b5"), op("tgt")), op(" = "), var("\u03c1"), subs(var("\u03b5"), op("sh")), op(" + "),
        rad([op("1 \u2212 "), sups(var("\u03c1"), op("2"))]), subs(var("\u03b5"), op("ind"))
    ], 4)
    body(doc,
         "Both latents are the clean sequence noised to the same timestep and differ only in the noise field. The "
         "loss sums a reconstruction term and a term on adjacent frame differences,")
    dk.add_equation(doc, [
        var("L"), op(" = "), run("SmoothL1", "p"), op("("),
        subs(var("R"), var("\u03c6")), op("("), subs(var("z"), var("t")), op(", "), var("\u03c4"), op("), "),
        subsups(var("z"), var("t"), var("\u03c1")), op(") + 0.2 "), run("SmoothL1", "p"), op("("),
        var("\u0394"), subs(var("R"), var("\u03c6")), op(", "), var("\u0394"), subsups(var("z"), var("t"), var("\u03c1")),
        op(")")
    ], 5)
    body(doc,
         "The first term maps an independently noised latent to the correlated target at the same timestep, which "
         "is close to learning a conditional mean. The second keeps the output from becoming a globally contracted "
         "copy. This teaches a local, noise conditioned correction and cannot teach real motion, because the "
         "temporal structure of the target is imposed rather than measured. Training runs for 800 steps on 150 "
         "sequences and takes 43.8 s, with the loss falling from 0.1091 to 0.0346.")

    heading(doc, "Gating schedule", 2)
    body(doc,
         "The scheduler is the PNDM default of the checkpoint. For ten requested steps it exposes eleven calls at "
         "normalised timesteps 0.902, 0.802, 0.802, 0.702, 0.602, 0.502, 0.402, 0.301, 0.201, 0.101 and 0.001. The "
         "submitted version wrote the gate as a threshold of 0.30, which is only 0.0013 below the 0.301 call, so "
         "rounding decided how many calls were gated. We count calls instead, as in Eq. (2). $K$ = $N$ is full "
         "injection and $K$ = 0 means no residual. $K$ = 3 is the pre-registered setting.")
    body(doc,
         "Two arguments suggest that gains appear late. The module replaces independent frame noise with correlated "
         "noise, so it behaves like a low pass filter along the frame axis. During the noisiest steps the UNet is "
         "deciding global structure, and a prior of this kind fights that decision while the disturbance is "
         "amplified downstream. Once latents approach the data manifold, frame differences are dominated by "
         "sampling noise and the same operator removes part of it. We therefore expect full injection to be no "
         "better than none and the gain to grow with the window until the noisiest calls enter it.")

    heading(doc, "Deterministic compositor", 2)
    body(doc,
         "The compositor resizes the editable asset, applies the planned scale and angle, and alpha composites it "
         "at the specified centre. The safe area and the trajectory are stored explicitly. Bounding box IoU, safe "
         "area violation, centre error and trajectory residual are checks on the compositor, not evidence about a "
         "generated background, and we do not use them for that purpose.")

    heading(doc, "EXPERIMENTAL DESIGN")
    body(doc,
         "The adapters are trained on synthetic sequence indices 0 to 149 and evaluated on 150 to 159, so no "
         "evaluated asset was seen in training. Each sequence has 16 procedural frames. The held-out group is "
         "lica-world_GDB__typography-7, which is one source group and not ten independent brands. This is the main "
         "limitation of the study. Each asset is rendered with three fixed seeds under seven conditions, which "
         "gives 150 sequences for the five pre-registered conditions and 60 more for the attribution runs. The "
         "unit of analysis is the asset and seed pair, so $n$ = 30.")
    body(doc,
         "The primary outcome is background jitter, the mean absolute RGB difference between adjacent generated "
         "background frames on a 0 to 1 scale. Lower values mean less change. Frame jitter repeats the calculation "
         "after compositing. Edge recall is the fraction of generated edges that overlaps the Canny condition, "
         "logo fidelity is one minus the mean absolute difference inside the planned crop, and frame error is the "
         "full frame mean absolute error against the target. We also compute adjacent frame SSIM, high pass jitter "
         "after subtracting a blurred copy, the 95th percentile of frame pair differences, and jitter on static "
         "regions, which are pixels that the target frames change by less than 8 of 255. Those pixels cover 93% of "
         "the frame on this split.")
    body(doc,
         "For the three pre-specified contrasts we compute paired differences, baseline minus treatment, with "
         "5,000-resample bootstrap intervals, 100,000-draw sign-flip $p$ values and Holm adjustment over three "
         "tests. The sweep, the attribution conditions and the alternative measures are sensitivity analyses and "
         "are not treated as confirmatory. An audit checks the 150 rows, the condition balance, the full grid, "
         "finite values, output media and the separation of training and evaluation indices. The sweep reproduces "
         "the single sample value for $K$ = 3 exactly, 0.2649148503939311.")

    heading(doc, "RESULTS")
    body(doc,
         "The adapter produced the largest change. Mean jitter fell from 0.3964 to 0.2676, a paired reduction of "
         "0.12886 (95% CI 0.12374 to 0.13414, Holm-adjusted $p$ = 3.0 × 10⁻⁵), or 32.51%, and all 30 pairs moved "
         "that way (Table 1). Full injection raised jitter to 0.2786, 4.13% worse than sampling without the "
         "residual (paired \u22120.01106, 95% CI \u22120.01361 to \u22120.00856, 2 of 30 pairs in the other "
         "direction). The three-call gate lowered it to 0.2653, a paired gain of 0.00226 (95% CI 0.00118 to "
         "0.00331, Holm-adjusted $p$ = 3.7 × 10⁻⁴, 0.84%, 22 of 30 pairs, effect size 0.74). The locked condition "
         "has zero jitter by construction and marks the lower limit of the measure.")
    dk.add_table(
        doc,
        "Table 1. Held-out evaluation. Values are mean and sample SD over 30 asset and seed sequences per "
        "condition. Jitter and error are lower when better.",
        ["Condition", "Background jitter", "Frame jitter", "Edge recall", "Logo fidelity", "Frame error"],
        [["No adapter", "0.3964 \u00b1 0.0084", "0.3863 \u00b1 0.0081", "0.8566 \u00b1 0.0335",
          "0.9117 \u00b1 0.0401", "0.3142 \u00b1 0.0090"],
         ["Adapter", "0.2676 \u00b1 0.0100", "0.2592 \u00b1 0.0102", "0.9190 \u00b1 0.0322",
          "0.9287 \u00b1 0.0331", "0.2416 \u00b1 0.0061"],
         ["Locked", "0.0000 \u00b1 0.0000", "0.0023 \u00b1 0.0013", "0.8816 \u00b1 0.0647",
          "0.9394 \u00b1 0.0290", "0.2096 \u00b1 0.0677"],
         ["Full injection", "0.2786 \u00b1 0.0088", "0.2700 \u00b1 0.0097", "0.9099 \u00b1 0.0368",
          "0.9266 \u00b1 0.0312", "0.2481 \u00b1 0.0062"],
         ["Gate, K = 3", "0.2653 \u00b1 0.0105", "0.2569 \u00b1 0.0102", "0.9219 \u00b1 0.0328",
          "0.9292 \u00b1 0.0309", "0.2405 \u00b1 0.0064"]],
        [3.0, 2.9, 2.8, 2.7, 2.8, 2.7],
        font_size=8.5)
    body(doc,
         "Table 2 separates the components on the same assets and seeds. Removing the adapter costs 0.1289, which "
         "is by far the largest term. Adding the wide gate to the adapted pipeline gains 0.0079, and the same gate "
         "without the adapter gains only 0.0019, so the residual works better once the latent distribution has been "
         "adapted. Removing the Canny condition raises jitter by 0.0726 and lowers edge recall from 0.9190 to "
         "0.8133, but it also makes the output independent of the asset, which we checked on the stored frames, so "
         "that row is descriptive rather than paired. The compositor shifts frame jitter by a fixed 0.0066 and does "
         "not change the ranking of any pair.")
    dk.add_table(
        doc,
        "Table 2. Component attribution. Each row changes one part of the pipeline and reports the paired change in "
        "jitter, treatment minus baseline, so a negative value means the change lowered jitter.",
        ["Comparison", "Background jitter change", "Frame jitter change", "Isolates"],
        [["Adapter added", "\u22120.1289", "\u22120.1271", "Style adaptation"],
         ["Wide gate, adapter present", "\u22120.0079", "\u22120.0079", "Residual injection"],
         ["Wide gate, adapter absent", "\u22120.0019", "\u22120.0021", "Residual alone"],
         ["Full gate instead of late", "+0.0190", "+0.0186", "Over-injection"],
         ["Canny condition removed", "+0.0726", "+0.0731", "Layout condition"],
         ["Deterministic compositor", "\u22120.0066", "\u22120.0066", "Foreground pasting"]],
        [5.6, 4.0, 3.6, 3.9],
        font_size=8.5,
        align=["left", "center", "center", "left"])
    body(doc,
         "The sweep shows three things (Table 3, Figure 2). Full injection is clearly harmful, with 28 of 30 pairs "
         "moving against it. The gain grows as the window widens, from 0.44% at $K$ = 1 and 0.26% at $K$ = 2, which "
         "has an interval spanning zero, to 2.31% at $K$ = 6 and 2.95% at $K$ = 8, and then reverses to \u22124.13% "
         "when all eleven calls are gated. Six to eight calls form a plateau, since the difference between them is "
         "only 0.0017. The pre-registered $K$ = 3 sits at the low end: moving to $K$ = 8 lowers jitter by a further "
         "0.00564 (95% CI 0.00313 to 0.00826) with 24 of 30 pairs agreeing. Because $K$ was chosen on the same "
         "split we do not present $K$ = 8 as a new setting. The rule we take is to leave the noisiest calls to the "
         "UNet and to widen the window after that point.")
    dk.add_table(
        doc,
        "Table 3. Gating schedule. The difference column is baseline minus treatment, so a positive value means "
        "lower jitter with the residual.",
        ["Gated calls", "Jitter", "Change", "95% CI", "p", "Pairs"],
        [["0", "0.2676 \u00b1 0.0100", "\u2014", "\u2014", "\u2014", "\u2014"],
         ["3", "0.2653 \u00b1 0.0105", "+0.00226", "0.00118 to 0.00331", "3.7 × 10⁻⁴", "22/30"],
         ["6", "0.2614 \u00b1 0.0108", "+0.00618", "0.00435 to 0.00795", "1.0 × 10⁻⁵", "27/30"],
         ["8", "0.2597 \u00b1 0.0114", "+0.00790", "0.00530 to 0.01062", "1.0 × 10⁻⁵", "26/30"],
         ["11", "0.2786 \u00b1 0.0088", "\u22120.01106", "\u22120.01361 to \u22120.00856", "1.0 × 10⁻⁵", "2/30"]],
        [2.8, 3.4, 2.8, 4.6, 2.4, 2.0],
        font_size=8.5)
    dk.add_figure(doc, FIGURES / "fig4_gating_sweep.png", 7.8,
                  "Figure 2. Gating schedule. Mean jitter against the number of gated calls. Bars are sample standard "
                  "deviations, the dashed line marks the no-residual level, the diamond is the pre-registered "
                  "setting, the star the sweep minimum and the open square full injection.")
    body(doc,
         "The gain survives two of the three alternative measures. Adjacent frame SSIM improves by 0.01257 (95% CI "
         "0.00972 to 0.01550, $p$ = 1.0 × 10⁻⁵) and high pass jitter by 0.00064 (95% CI 0.00011 to 0.00117, $p$ = "
         "0.029). The worst frame pairs do not reach significance, with 0.00146 and an interval from \u22120.00050 "
         "to 0.00324. On static regions the paired reduction is 0.00228 (95% CI 0.00122 to 0.00330, $p$ = 3.2 × "
         "10⁻⁴), almost the same as on the full frame, while the moving 7% shows 0.00281 with an interval that "
         "includes zero (95% CI 0.000003 to 0.00563, $p$ = 0.063). The gain therefore appears where nothing was "
         "supposed to change, and we read it as a reduction of unwanted frame to frame change rather than better "
         "motion. Every stabilisation claim here is a claim about this proxy. The adapter holds 797,184 parameters "
         "and the residual 3,092, and training both takes under three minutes on one 8 GB laptop GPU.")

    heading(doc, "DISCUSSION")
    body(doc,
         "The main result is a design decision. Using the residual through the whole trajectory made the measure "
         "worse by 4.13%, while restricting it to later calls helped by 0.84 to 2.95%. Two mechanisms fit that "
         "pattern. The earliest steps fix global structure and should be left to the UNet, because a residual "
         "trained on simplified noise knows nothing about layout and its disturbance is amplified downstream. "
         "Later, latents are near the data manifold and a local filter removes part of the sampling noise. The "
         "sweep also shows that later is not always better, since one call gives 0.44% and eight give 2.95%. What "
         "matters is excluding the noisiest start. We are careful about what this does not show. It measures when "
         "to intervene, not why the intervention works.")
    body(doc,
         "The division of work is useful independently of the numbers. A workflow is easier to revise when the "
         "generator supplies atmosphere and texture while the compositor keeps authority over the title, the logo "
         "and the safe area, and checks for broadcast safety then become deterministic rather than probabilistic. "
         "Our results do not licence a production claim. The renders are 256 × 256 and were made for an "
         "integration test. Sequences are 16 frames long, so a unit of several seconds is outside the evidence, and "
         "the five-frame receptive field cannot cover the displacements that longer units would contain. The motion "
         "is procedural rather than filmed, no real rushes were used, and the foreground is composited "
         "deterministically, so the foreground metrics describe the compositor rather than the generator.")
    body(doc,
         "Five limitations follow. All ten held-out sequences come from one source group, so the paired units are "
         "not independent brands and a real study should stratify by genre, channel, graphic density, type system "
         "and motion family and split at the brand level. The renders are not delivery quality. The primary measure "
         "cannot separate wanted from unwanted change, and the static region analysis only partly fills that gap. "
         "No external video-diffusion baseline was run. The residual is trained on synthetic noise pairs, so its "
         "loss curve shows that a small optimisation completed and nothing more. The conclusion would also change "
         "with another scheduler, a longer sequence or a correlation estimated from real footage, and those "
         "experiments need a separate validation split that also fixes the window.")

    heading(doc, "CONCLUSION")
    body(doc,
         "We described a compact diffusion workflow for broadcast style packaging. A frozen backbone draws the "
         "background, a rank-4 adapter and an edge condition carry brand and layout cues, a deterministic "
         "compositor keeps the editable marks and the safe area, and a residual of 3,092 parameters is added during "
         "selected calls. On 150 held-out sequences the adapter reduced jitter by 32.51%, adding the residual to "
         "every call made jitter 4.13% worse, and restricting it to later calls gave 0.84 to 2.95%. The decisive "
         "boundary is the first and noisiest calls. The effect is a proxy finding that appears on average and in "
         "the parts of the frame the plan leaves unchanged, and it does not survive a worst-frame criterion. The "
         "next study needs real programme material, splits at the source level, an external video baseline, a "
         "validation split that fixes the window, and a blinded evaluation by broadcast designers.")

    heading(doc, "DATA AND CODE AVAILABILITY")
    body(doc,
         "The experiment configuration, the per-sequence metrics, the audit records, the figure source data and the "
         "extended tables are available at https://github.com/Expo-L/joey. The repository also holds the scripts "
         "that produce the sweep, the attribution runs and the alternative measures, so the reported values can be "
         "regenerated without the third-party assets. Those assets are not redistributed, and any reuse must "
         "respect their original licences. The evaluation used no human participants.")

    ref_head = doc.add_paragraph(style="SPIEreferences")
    ref_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ref_head.paragraph_format.keep_with_next = True
    ref_head.add_run("REFERENCES")

    references = [
        "R. Rombach, A. Blattmann, D. Lorenz, P. Esser and B. Ommer, \u201cHigh-resolution image synthesis with "
        "latent diffusion models,\u201d in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition, "
        "10684\u201310695 (2022). https://doi.org/10.1109/CVPR52688.2022.01042",
        "E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang and W. Chen, \u201cLoRA: low-rank adaptation of "
        "large language models,\u201d in Int. Conf. Learning Representations (2022). "
        "https://openreview.net/forum?id=nZeVKeeFYf9",
        "L. Zhang, A. Rao and M. Agrawala, \u201cAdding conditional control to text-to-image diffusion models,\u201d "
        "in Proc. IEEE/CVF Int. Conf. Computer Vision, 3836\u20133847 (2023). "
        "https://doi.org/10.1109/ICCV51070.2023.00355",
        "P. Cao, F. Zhou, Q. Song and L. Yang, \u201cControllable generation with text-to-image diffusion models: a "
        "survey,\u201d IEEE Trans. Pattern Analysis and Machine Intelligence (2025). "
        "https://arxiv.org/abs/2403.04279",
        "D. Biderman, J. Portes, J. J. Gonzalez Ortiz, M. Paul, P. Greengard, C. Jennings, D. King, S. Land, M. Li, "
        "H. Lin, J. Pauly, A. P. S. Kim and A. G. Wilson, \u201cLoRA learns less and forgets less,\u201d Trans. "
        "Machine Learning Research (2024). https://arxiv.org/abs/2405.09673",
        "S.-Y. Liu, C.-Y. Wang, H. Yin, P. Molchanov, Y.-C. F. Wang, K.-T. Cheng and M.-H. Chen, \u201cDoRA: "
        "weight-decomposed low-rank adaptation,\u201d in Proc. Int. Conf. Machine Learning (2024). "
        "https://arxiv.org/abs/2402.09353",
        "Z. Yang, J. Teng, W. Zheng, J. Tang, S. Xu, J. Wu, P. Li, L. Li, S. Li, Z. Wang, Y. Zhang, Q. He and "
        "B. Zhou, \u201cCogVideoX: text-to-video diffusion models with an expert transformer,\u201d in Int. Conf. "
        "Learning Representations (2025). https://arxiv.org/abs/2408.06072",
        "Wan Team, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, J. Zeng, J. Wang, "
        "J. Wang, J. Zhou, J. Lin and K. Zhu, \u201cWan: open and advanced large-scale video generative models,\u201d "
        "arXiv:2503.20314 (2025). https://arxiv.org/abs/2503.20314",
        "Y. Wang, X. Liu, W. Pang, J. Zhang, Y. Yu, L. Kong, C. Liang, Q. Zhao and Y. Liu, \u201cSurvey of video "
        "diffusion models: foundations, implementations, and applications,\u201d Trans. Machine Learning Research "
        "(2025). https://arxiv.org/abs/2504.16081",
        "Y. Guo, C. Yang, A. Rao, Z. Liang, Y. Wang, Y. Qiao, M. Agrawala, D. Lin and B. Dai, \u201cAnimateDiff: "
        "animate your personalized text-to-image diffusion models without specific tuning,\u201d in Int. Conf. "
        "Learning Representations (2024). https://arxiv.org/abs/2307.04725",
        "H. Qiu, M. Xia, Y. Zhang, Y. Yang, X. Wang, Y. Zhou and W.-F. Song, \u201cFreeNoise: tuning-free longer "
        "video diffusion via noise rescheduling,\u201d in Int. Conf. Learning Representations (2024). "
        "https://arxiv.org/abs/2310.15169",
        "H. Ouyang, Q. Wang, Y. Xiao, Q. Bai, J. Zhang, K. Zheng, X. Zhou, Q. Chen and Y. Shen, \u201cCoDeF: content "
        "deformation fields for temporally consistent video processing,\u201d in Proc. IEEE/CVF Conf. Computer Vision "
        "and Pattern Recognition, 8089\u20138099 (2024). https://arxiv.org/abs/2308.07926",
        "S. Yang, Y. Zhou, Z. Liu and C. C. Loy, \u201cRerender A Video: zero-shot text-guided video-to-video "
        "translation,\u201d in SIGGRAPH Asia (2023). https://arxiv.org/abs/2306.07954",
        "Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit, Y. Wang, "
        "X. Chen, L. Wang, D. Lin, Y. Qiao and Z. Liu, \u201cVBench: comprehensive benchmark suite for video "
        "generative models,\u201d in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition, 21807\u201321818 "
        "(2024). https://arxiv.org/abs/2311.17982",
        "S. Yuan, J. Huang, Y. Xu, Y. Liu, S. Zhang, Y. Ge, Y. Ge, Q. Jin, S. Liang, Y. Wang, Y. Wang and L. Yuan, "
        "\u201cChronoMagic-Bench: a benchmark for metamorphic evaluation of text-to-time-lapse video generation,\u201d "
        "in Advances in Neural Information Processing Systems, Datasets and Benchmarks Track (2024). "
        "https://arxiv.org/abs/2406.18522",
        "G. Y. Luo, G. M. Favero, Z. H. Luo, J. Wang, Y. Kang, C. Jiang and S.-N. Lim, \u201cBeyond FVD: enhanced "
        "evaluation metrics for video generation quality,\u201d in Int. Conf. Learning Representations (2025). "
        "https://arxiv.org/abs/2410.05203",
        "M. Morton, \u201cGraphic design, music and sound in the BBC\u2019s channel idents, 1991\u20132021,\u201d "
        "Critical Studies in Television 17(2), 117\u2013134 (2022). https://doi.org/10.1177/17496020211067736",
        "A. Asmar, T. Raats and L. Van Audenhove, \u201cStreaming difference(s): Netflix and the branding of "
        "diversity,\u201d Critical Studies in Television 18(1), 24\u201340 (2023). "
        "https://doi.org/10.1177/17496020221129516",
        "H. Weng, D. Huang, Y. Qiao, Z. Lin, J. Wang, T. He and F. Zhao, \u201cDesigen: a pipeline for controllable "
        "design template generation,\u201d in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition, "
        "12721\u201312732 (2024). https://arxiv.org/abs/2403.09093",
        "A. Hertz, A. Voynov, S. Fruchter and D. Cohen-Or, \u201cStyle aligned image generation via shared "
        "attention,\u201d in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition, 4775\u20134785 (2024). "
        "https://arxiv.org/abs/2312.02133",
        "J. Liu, P. Zhang, Y. Zhang, P. Yan, H. Zhou, X. Zhou, F. Guo and L. Jin, \u201cPosterVerse: a full-workflow "
        "framework for commercial-grade poster generation with HTML-based scalable typography,\u201d in AAAI Conf. "
        "Artificial Intelligence (2026). https://arxiv.org/abs/2601.03993",
        "J. Ma and M. Agrawala, \u201cMoVer: motion verification for motion graphics animations,\u201d ACM Trans. "
        "Graphics, SIGGRAPH 2025 issue (2025). https://doi.org/10.1145/3731209",
        "S. Park, I. Bae, S. Shin and H.-G. Jeon, \u201cKinetic typography diffusion model,\u201d in European Conf. "
        "Computer Vision (2024). https://arxiv.org/abs/2407.10476",
    ]
    for item in references:
        paragraph = doc.add_paragraph(style="SPIE reference listing")
        paragraph.paragraph_format.space_after = Pt(1)
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.0
        runner = paragraph.add_run(item)
        runner.font.size = Pt(7.5)

    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT_DOCX))
    print("saved", OUT_DOCX)


if __name__ == "__main__":
    build()
