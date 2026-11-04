"""Build the README media from a real run of the demo (nothing is typed in by hand).

    python docs/make_media.py        # needs: pip install pillow matplotlib

docs/demo.gif                    terminal animation of `python -m tabular_ds_mcp.demo`
docs/random_vs_structured.png    random K-fold vs time-ordered / group folds, from the same run
"""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from tabular_ds_mcp import demo

DOCS = Path(__file__).resolve().parent
COMMAND = "python -m tabular_ds_mcp.demo"
BG, FG, DIM = "#0d1117", "#c9d1d9", "#8b949e"
CYAN, GREEN, YELLOW, RED, PURPLE = "#79c0ff", "#3fb950", "#e3b341", "#ff7b72", "#d2a8ff"


def run_demo() -> tuple[list[str], dict]:
    """Run the demo in a temporary folder so the printed paths stay short."""
    buf, here = io.StringIO(), os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            with contextlib.redirect_stdout(buf):
                out = demo.run("demo")
        finally:
            os.chdir(here)
    return buf.getvalue().rstrip("\n").split("\n"), out


# ═══════════════════════════ terminal animation ═════════════════════════════
def _colour_lines(lines: list[str]) -> list[list[tuple[str, str]]]:
    """Colour every output line; a warning keeps its colour on its wrapped lines."""
    out, in_warn = [], False
    for line in lines:
        body = line[demo.LABEL:]
        in_warn = body.startswith("! ") or (in_warn and body.startswith("  "))
        if in_warn:
            out.append([(line, YELLOW)])
        elif line.startswith("Why the validation"):
            out.append([(line, PURPLE)])
        elif line.startswith(("tabular-ds-mcp demo", "done in")):
            out.append([(line, DIM)])
        elif body.startswith("→"):
            out.append([(line, RED)])
        else:
            label = line[:demo.LABEL]
            segs = [(label, CYAN)]
            if "ready to upload: True" in body:
                a, b = body.split("ready to upload: True", 1)
                segs += [(a, FG), ("ready to upload: True", GREEN), (b, FG)]
            else:
                segs.append((body, FG))
            out.append(segs)
    return out


def make_gif(lines: list[str], path: Path) -> None:
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15) if Path("/System/Library/Fonts/Menlo.ttc").exists() \
        else ImageFont.load_default(15)
    cw, lh, pad, bar = font.getlength("M"), 21, 18, 34
    cols = max(demo.WIDTH, max(len(l) for l in lines)) + 2
    w = int(cols * cw + 2 * pad)
    coloured = _colour_lines(lines)
    h = bar + pad + (len(lines) + 2) * lh + pad

    def frame(n_lines: int, typed: str, cursor: bool) -> Image.Image:
        im = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, w, bar], fill="#161b22")
        for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
            d.ellipse([14 + 20 * i, 11, 26 + 20 * i, 23], fill=c)
        y = bar + pad
        d.text((pad, y), "$ ", font=font, fill=GREEN)
        d.text((pad + 2 * cw, y), typed + ("█" if cursor and n_lines == 0 else ""), font=font, fill=FG)
        for segs in coloured[:n_lines]:
            y += lh
            x = pad
            for text, colour in segs:
                d.text((x, y), text, font=font, fill=colour)
                x += len(text) * cw
        return im

    frames, durations = [], []
    for i in range(0, len(COMMAND) + 1, 2):                         # typing the command
        frames.append(frame(0, COMMAND[:i], True)); durations.append(60)
    frames.append(frame(0, COMMAND, True)); durations.append(500)
    for n in range(1, len(lines) + 1):                               # output appears line by line
        step = lines[n - 1].startswith(("[", "time_col", "group_col"))
        frames.append(frame(n, COMMAND, False)); durations.append(700 if step else 120)
    durations[-1] = 6000                                             # hold the final screen
    ref = frames[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)    # one palette for all frames
    pal = [f.quantize(palette=ref, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)


# ═══════════════════════════ comparison chart ═══════════════════════════════
def make_chart(out: dict, path: Path) -> None:
    t, g = out["time"]["random_split_comparison"], out["group"]["random_split_comparison"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), dpi=160)
    panels = [
        (axes[0], "Drifting daily series  ·  time_col='date'", "RMSE (lower is better)",
         t["random_split"], t["time-ordered"], "time-ordered folds", None),
        (axes[1], "Customers in many rows  ·  group_col='customer_id'", "ROC-AUC (0.5 = no skill)",
         g["random_split"], g["group"], "unseen customers", 0.5),
    ]
    for ax, title, ylabel, rnd, real, real_name, chance in panels:
        bars = ax.bar(["random K-fold", real_name], [rnd, real], color=["#9aa5b1", "#e8590c"], width=0.55)
        for b, v in zip(bars, [rnd, real]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11,
                    fontweight="bold")
        if chance is not None:
            ax.axhline(chance, color="#555", ls="--", lw=1)
            ax.text(0.5, chance + 0.012, "no skill (0.5)", va="bottom", ha="center", fontsize=9, color="#555")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_ylim(0, max(rnd, real) * 1.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9.5)
    fig.suptitle("Random K-fold looks far better than it should; the tools measure by how much", fontsize=12,
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    lines, out = run_demo()
    make_gif(lines, DOCS / "demo.gif")
    make_chart(out, DOCS / "random_vs_structured.png")
    for p in ("demo.gif", "random_vs_structured.png"):
        print(p, f"{(DOCS / p).stat().st_size / 1024:.0f} KB")
