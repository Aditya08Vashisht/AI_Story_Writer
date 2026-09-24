"""Render video frames with Pillow. No image model, no Graphviz.

Every frame is drawn from a spec, so the same script produces byte-identical
output on every run. That reproducibility is worth more than visual flair for a
research artifact you have to defend -- and a diffusion model cannot render
legible text anyway, which is the whole payload of an explainer.

The diagram is drawn here rather than shelling out to Graphviz, because every
dependency that has failed on this cluster has been an install problem. Boxes
and arrows need no external binary.

THE DEVANAGARI TRAP: Pillow needs libraqm to shape complex scripts. Without it
matras and conjuncts render in the wrong order even with the correct font
loaded -- text that looks almost right, which is worse than obviously broken.
`ensure_devanagari_support()` refuses rather than emitting malformed text.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFont, features

W, H = 1280, 720
MARGIN = 88

BG = (11, 15, 23)
PANEL = (18, 24, 34)
TXT = (230, 237, 246)
DIM = (143, 163, 189)
ACCENT = (79, 156, 249)
ACCENT2 = (34, 197, 94)
LINE = (36, 48, 68)

DEVA = re.compile(r"[\u0900-\u097F]")

FONT_DIR = Path(os.environ.get("STORYTUTOR_FONT_DIR", "/scratch/%s/fonts" % os.environ.get("USER", "x")))
FONT_URLS = {
    "NotoSans-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf",
    "NotoSansDevanagari-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf",
}


class RenderError(RuntimeError):
    pass


def has_devanagari(text: str) -> bool:
    return bool(DEVA.search(text or ""))


def ensure_devanagari_support() -> None:
    """Refuse to render Devanagari without a text shaper."""
    if not features.check("raqm"):
        raise RenderError(
            "Pillow was built without libraqm, so Devanagari cannot be shaped "
            "correctly -- matras and conjuncts would render in the wrong order, "
            "producing text that looks almost right. Install a Pillow wheel with "
            "raqm support (recent manylinux wheels include it), or render only "
            "English by passing --language english."
        )


def _download_fonts() -> None:
    import urllib.request

    FONT_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FONT_URLS.items():
        target = FONT_DIR / name
        if target.exists():
            continue
        print(f"downloading font {name} ...")
        urllib.request.urlretrieve(url, target)


def load_font(size: int, devanagari: bool = False) -> ImageFont.FreeTypeFont:
    """Noto Sans, or Noto Sans Devanagari when the text needs it."""
    name = "NotoSansDevanagari-Regular.ttf" if devanagari else "NotoSans-Regular.ttf"
    path = FONT_DIR / name
    if not path.exists():
        try:
            _download_fonts()
        except Exception as exc:  # noqa: BLE001
            print(f"font download failed ({exc}); falling back to a system font")
    if path.exists():
        return ImageFont.truetype(str(path), size)

    for candidate in (
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf" if devanagari else None,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    words, lines, cur = (text or "").split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _base() -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 6], fill=ACCENT)
    return img, d


def _footer(d: ImageDraw.ImageDraw, source_line: str) -> None:
    """Source citation, drawn at the TOP of the card.

    It used to sit at the bottom, where burned-in subtitles land -- the two
    overlapped on every rendered video ('NCERT Class 6, Social Science, cOceans
    & Continents'). A dark pill behind it keeps it legible over an illustration.
    """
    if not source_line:
        return
    f = load_font(20)
    tw = d.textlength(source_line, font=f)
    x, y = MARGIN, 26
    d.rounded_rectangle([x - 12, y - 6, x + tw + 12, y + f.size + 8], radius=10, fill=(11, 15, 23))
    d.text((x, y), source_line, font=f, fill=DIM)


def _draw_block(d, text, font, x, y, max_w, fill, line_gap=14) -> int:
    for line in _wrap(d, text, font, max_w):
        d.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def title_card(title: str, subtitle: str, source_line: str = "") -> Image.Image:
    deva = has_devanagari(title + subtitle)
    if deva:
        ensure_devanagari_support()
    img, d = _base()
    f_title = load_font(64, deva)
    f_sub = load_font(30, deva)

    y = 230
    y = _draw_block(d, title, f_title, MARGIN, y, W - 2 * MARGIN, TXT, 18)
    if subtitle:
        d.text((MARGIN, y + 22), subtitle, font=f_sub, fill=ACCENT)
    _footer(d, source_line)
    return img


def idea_card(heading: str, body: str, source_line: str = "") -> Image.Image:
    deva = has_devanagari(heading + body)
    if deva:
        ensure_devanagari_support()
    img, d = _base()
    f_head = load_font(30, deva)
    f_body = load_font(46, deva)

    d.text((MARGIN, 140), heading.upper() if not deva else heading, font=f_head, fill=ACCENT)
    _draw_block(d, body, f_body, MARGIN, 210, W - 2 * MARGIN, TXT, 18)
    _footer(d, source_line)
    return img


def diagram_card(heading: str, nodes: Sequence[str], edges: Sequence[Sequence[str]],
                 source_line: str = "") -> Image.Image:
    """Boxes and arrows, laid out top-to-bottom or left-to-right by count."""
    deva = has_devanagari(heading + " ".join(nodes))
    if deva:
        ensure_devanagari_support()
    img, d = _base()
    d.text((MARGIN, 96), heading, font=load_font(30, deva), fill=ACCENT)

    nodes = [n for n in nodes if n][:6]
    if not nodes:
        return img

    f_node = load_font(26, deva)
    horizontal = len(nodes) <= 3
    boxes = {}

    if horizontal:
        bw, bh = min(300, (W - 2 * MARGIN - 40 * (len(nodes) - 1)) // len(nodes)), 130
        y = 300
        for i, n in enumerate(nodes):
            x = MARGIN + i * (bw + 40)
            boxes[n.strip().lower()] = (x, y, x + bw, y + bh)
    else:
        bw, bh = 420, 76
        gap = 26
        total = len(nodes) * bh + (len(nodes) - 1) * gap
        y0 = (H - total) // 2 + 20
        x = (W - bw) // 2
        for i, n in enumerate(nodes):
            y = y0 + i * (bh + gap)
            boxes[n.strip().lower()] = (x, y, x + bw, y + bh)

    for n in nodes:
        x0, y0, x1, y1 = boxes[n.strip().lower()]
        d.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=PANEL, outline=ACCENT, width=2)
        for line in _wrap(d, n, f_node, x1 - x0 - 24)[:3]:
            tw = d.textlength(line, font=f_node)
            d.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0) / 2 - f_node.size / 2), line,
                   font=f_node, fill=TXT)

    for edge in edges:
        if len(edge) != 2:
            continue
        a, b = boxes.get(str(edge[0]).strip().lower()), boxes.get(str(edge[1]).strip().lower())
        if not a or not b:
            continue
        if horizontal:
            start, end = (a[2], (a[1] + a[3]) // 2), (b[0], (b[1] + b[3]) // 2)
        else:
            start, end = ((a[0] + a[2]) // 2, a[3]), ((b[0] + b[2]) // 2, b[1])
        d.line([start, end], fill=ACCENT2, width=3)
        # arrowhead
        if horizontal:
            d.polygon([end, (end[0] - 12, end[1] - 7), (end[0] - 12, end[1] + 7)], fill=ACCENT2)
        else:
            d.polygon([end, (end[0] - 7, end[1] - 12), (end[0] + 7, end[1] - 12)], fill=ACCENT2)

    _footer(d, source_line)
    return img


def check_card(heading: str, question: str, source_line: str = "") -> Image.Image:
    deva = has_devanagari(heading + question)
    if deva:
        ensure_devanagari_support()
    img, d = _base()
    f_head = load_font(30, deva)
    f_q = load_font(42, deva)

    d.text((MARGIN, 150), heading, font=f_head, fill=ACCENT2)
    d.rounded_rectangle([MARGIN - 24, 210, W - MARGIN + 24, 470], radius=18,
                        fill=PANEL, outline=LINE, width=2)
    _draw_block(d, question, f_q, MARGIN, 250, W - 2 * MARGIN, TXT, 16)
    _footer(d, source_line)
    return img


# ------------------------------------------------------------ illustrated

def _cover(image: Image.Image) -> Image.Image:
    """Scale to fill 1280x720 and centre-crop, preserving aspect."""
    img = image.convert("RGB")
    scale = max(W / img.width, H / img.height)
    img = img.resize((int(img.width * scale + 0.5), int(img.height * scale + 0.5)), Image.LANCZOS)
    left, top = (img.width - W) // 2, (img.height - H) // 2
    return img.crop((left, top, left + W, top + H))


def _scrim(base: Image.Image, side: str) -> Image.Image:
    """Darken one side of the picture so text over it stays readable.

    Near-solid behind the text, then a smooth fade, so the illustration still
    reads as one image instead of a picture with a panel pasted on it. A plain
    linear fade was tried first: over a bright picture the ends of longer
    lines sat on almost no shade and were hard to read.
    """
    def ramp(t: float) -> float:          # 1 inside the text area, eased to 0
        t = min(1.0, max(0.0, t))
        return 1 - t * t * (3 - 2 * t)

    if side == "left":
        solid, fade = int(W * 0.58), int(W * 0.30)
        line = Image.new("L", (W, 1))
        line.putdata([int(225 * ramp((x - solid) / fade)) for x in range(W)])
        mask = line.resize((W, H))
    else:  # bottom
        solid, fade = H - 330, int(H * 0.28)
        col = Image.new("L", (1, H))
        col.putdata([int(230 * ramp((solid - y) / fade)) for y in range(H)])
        mask = col.resize((W, H))
    # A light top band keeps the source pill readable on bright pictures.
    band = Image.new("L", (1, H))
    band.putdata([int(150 * max(0.0, 1 - y / 110)) for y in range(H)])
    mask = ImageChops.lighter(mask, band.resize((W, H)))
    shade = Image.new("RGB", (W, H), BG)
    return Image.composite(shade, base, mask)


def illustrated_card(image: Image.Image, key: str, heading: str, body: str,
                     source_line: str = "", subtitle: str = "") -> Image.Image:
    """An illustration with the scene's text drawn over it by Pillow.

    The image model only ever paints the picture; every word here is drawn by
    Pillow, so Devanagari is shaped correctly and nothing on screen is a
    diffusion model's attempt at spelling.
    """
    text = " ".join([heading, body, subtitle])
    deva = has_devanagari(text)
    if deva:
        ensure_devanagari_support()

    side = "bottom" if key == "check" else "left"
    img = _scrim(_cover(image), side)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 6], fill=ACCENT)

    if key == "title":
        y = _draw_block(d, heading, load_font(62, deva), MARGIN, 250, int(W * 0.52), TXT, 16)
        if subtitle:
            d.text((MARGIN, y + 18), subtitle, font=load_font(30, deva), fill=ACCENT)
    elif key == "check":
        d.text((MARGIN, H - 290), heading, font=load_font(30, deva), fill=ACCENT2)
        _draw_block(d, body, load_font(40, deva), MARGIN, H - 240, W - 2 * MARGIN, TXT, 14)
    else:  # idea
        d.text((MARGIN, 150), heading, font=load_font(30, deva), fill=ACCENT)
        _draw_block(d, body, load_font(44, deva), MARGIN, 212, int(W * 0.50), TXT, 16)

    _footer(d, source_line)
    return img


def render_scene(scene, script, image: Optional[Image.Image] = None) -> Image.Image:
    """Dispatch a scene to its card renderer.

    With an illustration, title/idea/check are drawn over the picture. The
    diagram is always drawn from its node/edge spec and never illustrated,
    because it is the one card whose content must be exactly right.
    """
    src = script.source_line
    if image is not None and scene.key != "diagram":
        sub = ""
        if scene.key == "title":
            sub = " · ".join(x for x in [
                f"Class {script.class_level}" if script.class_level else "",
                script.subject.replace("_", " ").title() if script.subject else "",
            ] if x)
            heading = script.title or scene.heading
            return illustrated_card(image, "title", heading, "", src, sub)
        if scene.key == "check":
            return illustrated_card(image, "check", scene.heading or "Your turn", scene.body, src)
        return illustrated_card(image, "idea", scene.heading, scene.body, src)

    if scene.key == "title":
        sub = " · ".join(x for x in [
            f"Class {script.class_level}" if script.class_level else "",
            script.subject.replace("_", " ").title() if script.subject else "",
        ] if x)
        return title_card(script.title or scene.heading, sub, src)
    if scene.key == "diagram":
        return diagram_card(scene.heading or "How it works",
                            script.diagram_nodes, script.diagram_edges, src)
    if scene.key == "check":
        return check_card(scene.heading or "Your turn", scene.body, src)
    return idea_card(scene.heading, scene.body, src)
