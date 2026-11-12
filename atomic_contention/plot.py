#!/usr/bin/env python3
"""Plot the benchmark CSVs: median time vs updates per target, double and vec3 side by side.

Usage: plot.py [results_dir=results]  ->  <results_dir>/plot.pdf
Standard library only (no matplotlib); the SVG it draws is converted with rsvg-convert.
"""
import csv
import glob
import math
import os
import subprocess
import sys
import tempfile

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
JULIA = "julia_wg64"  # wg256 is within noise of wg64, see summary.md

rows = []
for path in glob.glob(os.path.join(RESULTS, "*.csv")):
    with open(path) as f:
        rows += list(csv.DictReader(f))
N = int(rows[0]["N"])


def series(impl, variant):
    pts = []
    for r in rows:
        if r["impl"] == impl and r["variant"] == variant and r["median_ms"] != "nan":
            pts.append((N / int(r["M"]), float(r["median_ms"])))
    return sorted(pts)


# (label, impl, variant, css class, marker)
PANELS = [
    ("double", [
        ("non-atomic baseline (Kokkos, racy)", "kokkos", "scalar_plain", "base", None),
        ("Kokkos  atomic_add(double)", "kokkos", "scalar_atomic", "s1", "circle"),
        ("Julia  Atomix.@atomic (KA, wg 64)", JULIA, "scalar_atomic", "s2", "square"),
    ]),
    ("vec3", [
        ("non-atomic baseline (Kokkos, racy)", "kokkos", "vec3_plain", "base", None),
        ("Kokkos  3× atomic_add on components", "kokkos", "vec3_atomic_cmpt", "s1", "circle"),
        ("Julia  3× Atomix.@atomic on components", JULIA, "vec3_atomic_cmpt", "s2", "square"),
        ("Kokkos  atomic_add(NeoN::Vec3) — lock-based", "kokkos", "vec3_atomic_struct", "s3", "triangle"),
    ]),
]

# geometry
W, H = 1240, 600
TOP, BOTTOM = 150, 80
PW, GAP, LEFT = 520, 90, 80
PH = H - TOP - BOTTOM
XMIN, XMAX = 1, N
YMIN, YMAX = 0.3, 100.0


def sx(x, x0):
    return x0 + (math.log2(x) - math.log2(XMIN)) / (math.log2(XMAX) - math.log2(XMIN)) * PW


def sy(y):
    return TOP + PH - (math.log10(y) - math.log10(YMIN)) / (math.log10(YMAX) - math.log10(YMIN)) * PH


def marker(kind, x, y, cls):
    if kind == "circle":
        return f'<circle class="m {cls}" cx="{x:.1f}" cy="{y:.1f}" r="4.5"/>'
    if kind == "square":
        return f'<rect class="m {cls}" x="{x-4:.1f}" y="{y-4:.1f}" width="8" height="8" rx="1"/>'
    return (f'<path class="m {cls}" d="M{x:.1f},{y-5.5:.1f} L{x+5:.1f},{y+3.5:.1f} '
            f'L{x-5:.1f},{y+3.5:.1f} Z"/>')


def fmt_count(v):
    v = int(v)
    if v >= 2**20:
        return f"{v >> 20}M"
    if v >= 2**10:
        return f"{v >> 10}k"
    return str(v)


out = []
out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           'font-family="Inter, DejaVu Sans, Helvetica, Arial, sans-serif">')
LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a8984", grid="#e6e5e1",
             band="#f0efec", s1="#2a78d6", s2="#eb6834", s3="#1baf7a", base="#a3a29c")
DARK = dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#8f8e88", grid="#2e2e2c",
            band="#262624", s1="#3987e5", s2="#d95926", s3="#199e70", base="#6b6a65")


def css(c):
    # plain colors per class (rsvg-convert doesn't support CSS custom properties)
    return f"""
  .bg {{ fill: {c['surface']}; }}
  .title, .ptitle, .leg, .lbl {{ fill: {c['ink']}; }}
  .sub, .tick, .axl, .note {{ fill: {c['ink2']}; }}
  .grid {{ stroke: {c['grid']}; }}
  .axis {{ stroke: {c['muted']}; }}
  .band {{ fill: {c['band']}; }}
  .line.base {{ stroke: {c['base']}; }}
  .line.s1 {{ stroke: {c['s1']}; }} .line.s2 {{ stroke: {c['s2']}; }} .line.s3 {{ stroke: {c['s3']}; }}
  .m {{ stroke: {c['surface']}; }}
  .m.s1 {{ fill: {c['s1']}; }} .m.s2 {{ fill: {c['s2']}; }} .m.s3 {{ fill: {c['s3']}; }}"""


out.append(f"""<style>
  .title {{ font-size: 20px; font-weight: 600; }}
  .sub {{ font-size: 13px; }}
  .ptitle {{ font-size: 15px; font-weight: 600; }}
  .tick {{ font-size: 11.5px; }}
  .axl {{ font-size: 12.5px; }}
  .note {{ font-size: 11.5px; }}
  .grid {{ stroke-width: 1; }}
  .axis {{ stroke-width: 1; }}
  .line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
  .line.base {{ stroke-dasharray: 5 4; stroke-width: 1.5; }}
  .m {{ stroke-width: 2; }}
  .leg {{ font-size: 12.5px; }}
  .lbl {{ font-size: 12px; font-weight: 600; }}{css(LIGHT)}
  @media (prefers-color-scheme: dark) {{{css(DARK)}
  }}
</style>""")
out.append(f'<rect class="bg" width="{W}" height="{H}"/>')
out.append(f'<text class="title" x="{LEFT}" y="34">Atomic adds under contention: Julia/Atomix vs Kokkos</text>')
out.append(f'<text class="sub" x="{LEFT}" y="56">RTX 4070 · N = 2^{int(math.log2(N))} updates scattered '
           'onto M targets · median time per kernel launch (log–log, lower is better) · '
           'contention increases to the right</text>')

# legend (shared by both panels)
LEGEND = [
    ("non-atomic baseline (Kokkos, racy)", "base", None),
    ("Kokkos  Kokkos::atomic_add  (vec3: 3×, one per component)", "s1", "circle"),
    ("Julia  Atomix.@atomic, KA workgroup 64  (vec3: 3×, one per component)", "s2", "square"),
    ("Kokkos  atomic_add(NeoN::Vec3) — lock-based, as in NeoN  (vec3 only)", "s3", "triangle"),
]
lx, ly = LEFT, 88
for i, (label, cls, mk) in enumerate(LEGEND):
    x = lx + (i % 2) * 520
    y = ly + (i // 2) * 22
    out.append(f'<line class="line {cls}" x1="{x}" y1="{y}" x2="{x+26}" y2="{y}"/>')
    if mk:
        out.append(marker(mk, x + 13, y, cls))
    out.append(f'<text class="leg" x="{x+34}" y="{y+4}">{label}</text>')

for p, (ptitle, sers) in enumerate(PANELS):
    x0 = LEFT + p * (PW + GAP)
    # target array bigger than the 36 MB L2 -> every update goes to DRAM
    elem = 8 if ptitle == "double" else 24
    bx1 = sx(max(N * elem / (36 * 2**20), 1), x0)
    out.append(f'<rect class="band" x="{x0}" y="{TOP}" width="{bx1-x0:.1f}" height="{PH}"/>')
    out.append(f'<text class="note" x="{x0+6}" y="{TOP+PH-22}">target array</text>')
    out.append(f'<text class="note" x="{x0+6}" y="{TOP+PH-8}">&gt; L2 (36 MB)</text>')

    for e in range(-1, 3):
        for mult in (1, 3) if e < 2 else (1,):
            v = mult * 10**e
            if YMIN <= v <= YMAX:
                y = sy(v)
                out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+PW}" y2="{y:.1f}"/>')
                lab = f"{v:g}"
                out.append(f'<text class="tick" x="{x0-8}" y="{y+4:.1f}" text-anchor="end">{lab}</text>')
    for k in range(0, int(math.log2(N)) + 1, 4):
        x = sx(2**k, x0)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{TOP+PH}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{TOP+PH+18}" text-anchor="middle">{fmt_count(2**k)}</text>')
    out.append(f'<line class="axis" x1="{x0}" y1="{TOP+PH}" x2="{x0+PW}" y2="{TOP+PH}"/>')
    out.append(f'<text class="ptitle" x="{x0}" y="{TOP-12}">{ptitle}</text>')
    out.append(f'<text class="axl" x="{x0+PW/2}" y="{TOP+PH+44}" text-anchor="middle">'
               'updates per target address (N / M)</text>')
    if p == 0:
        out.append(f'<text class="axl" transform="translate({x0-52},{TOP+PH/2}) rotate(-90)" '
                   'text-anchor="middle">time per launch [ms]</text>')

    for label, impl, variant, cls, mk in sers:
        pts = series(impl, variant)
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(x, x0):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
        out.append(f'<path class="line {cls}" d="{d}"/>')
        if mk:
            out.extend(marker(mk, sx(x, x0), sy(y), cls) for x, y in pts)
        if variant == "vec3_atomic_struct":
            x, y = pts[-1]
            px, py = sx(x, x0), sy(y)
            cmpt = dict(series("kokkos", "vec3_atomic_cmpt"))[x]
            lines = [("lbl", "lock-based Vec3"),
                     ("note", f"{y:.0f} ms = {y / cmpt:.0f}× the component atomics"),
                     ("note", f"not run beyond {fmt_count(x)} updates/target"),
                     ("note", "(2^16 updates on 1 target took 5.5 s)")]
            for k, (c, t) in enumerate(lines):
                out.append(f'<text class="{c}" x="{px-12:.1f}" y="{py-34+15*k:.1f}" '
                           f'text-anchor="end">{t}</text>')

out.append(f'<text class="note" x="{LEFT}" y="{H-16}">Julia with workgroup 256 is within noise of 64 '
           '(see summary.md). Vec3s are interleaved xyz doubles on both sides; same index pattern '
           '(splitmix64 % M) on both sides.</text>')
out.append("</svg>")
pdf_path = os.path.join(RESULTS, "plot.pdf")
with tempfile.NamedTemporaryFile("w", suffix=".svg") as f:
    f.write("\n".join(out))
    f.flush()
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", pdf_path, f.name], check=True)
print(pdf_path)
