#!/usr/bin/env python3
"""
Render a vascular tree CSV directory as an interactive 3D HTML viewer.

Each tree CSV is subsampled to the N largest-diameter segments (default
8000) so a 100 M-segment tree renders in a browser. The trunk + major
branches dominate this subsample so the topology is preserved.

Output is a single self-contained HTML file using Plotly.js.

Usage:
    python build_tree_viewer.py <tree_csv_dir> [<output_html>] [<top_n>]

    tree_csv_dir : directory containing *_segments.csv files
                   (e.g. VascularTreeSim.jl/output)
    output_html  : output path (default: <tree_csv_dir>/tree_viewer.html)
    top_n        : segments to keep per tree (default 8000)

Examples:
    # Max-dilated tree
    python build_tree_viewer.py ../output

    # At-rest tree with more detail
    python build_tree_viewer.py ../output_at_rest viewer_rest.html 15000
"""

import csv
import json
import os
import sys
import re


PALETTE = {
    "LAD": "#1f77ff",
    "LCX": "#e3342f",
    "RCA": "#22aa44",
    # generic palette for non-coronary trees
    "_fallback": ["#1f77ff", "#e3342f", "#22aa44",
                  "#ff7f0e", "#9467bd", "#17becf",
                  "#d62728", "#bcbd22", "#e377c2", "#8c564b"],
}


def auto_color(name, idx):
    return PALETTE.get(name.upper(), PALETTE["_fallback"][idx % 10])


CSV_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9]*?)(?:_grown)?(?:_segments)?\.csv$")
AUX_FILES = {"domain_points.csv", "chambers_points.csv", "pericardium_points.csv",
             "great_vessels_points.csv", "coronary_arteries_points.csv"}


def discover_trees(tree_dir):
    """Returns dict {tree_name: csv_path} for every tree CSV in dir."""
    found = {}
    for f in sorted(os.listdir(tree_dir)):
        if f in AUX_FILES or not f.endswith(".csv"):
            continue
        m = CSV_PATTERN.match(f)
        if m is None:
            continue
        name = m.group(1).upper()
        found[name] = os.path.join(tree_dir, f)
    return found


def load_top_segments(csv_path, top_n):
    """Stream the CSV once, keep the N largest-diameter segments.

    Returns a list of (d_um, x1, y1, z1, x2, y2, z2, label).
    """
    import heapq
    heap = []  # min-heap of (d_um, tuple) so we evict smallest
    counter = 0
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {col: i for i, col in enumerate(header)}
        x1i, y1i, z1i = idx["x1_cm"], idx["y1_cm"], idx["z1_cm"]
        x2i, y2i, z2i = idx["x2_cm"], idx["y2_cm"], idx["z2_cm"]
        di = idx["diameter_um"]
        li = idx.get("label", -1)
        for row in reader:
            d = float(row[di])
            item = (d,
                    float(row[x1i]), float(row[y1i]), float(row[z1i]),
                    float(row[x2i]), float(row[y2i]), float(row[z2i]),
                    row[li] if li >= 0 else "")
            counter += 1
            if len(heap) < top_n:
                heapq.heappush(heap, item)
            elif d > heap[0][0]:
                heapq.heapreplace(heap, item)
    heap.sort(key=lambda t: t[0])  # ascending diameter
    return counter, heap


def build_plotly_trace(name, color, segs):
    """Build a Plotly scatter3d trace with mode=lines for one tree.

    Segments are concatenated with None separators so the whole tree
    is one trace (much faster than per-segment traces).
    """
    xs, ys, zs, hover = [], [], [], []
    for (d, x1, y1, z1, x2, y2, z2, label) in segs:
        xs += [x1, x2, None]
        ys += [y1, y2, None]
        zs += [z1, z2, None]
        text = f"{name}<br>d = {d:.1f} μm<br>{label}"
        hover += [text, text, None]
    return {
        "type": "scatter3d",
        "mode": "lines",
        "name": name,
        "x": xs, "y": ys, "z": zs,
        "line": {"color": color, "width": 3},
        "hoverinfo": "text",
        "text": hover,
    }


def build_html(traces, title):
    traces_json = json.dumps(traces)
    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; background: #0c0e13; color: #e2e2e2; font-family: sans-serif; }}
  #plot {{ width: 100vw; height: 100vh; }}
  #legend {{ position: fixed; left: 12px; top: 12px; background: rgba(0,0,0,0.55); padding: 8px 12px; border-radius: 4px; font-size: 13px; max-width: 280px; line-height: 1.4; }}
  #legend b {{ color: #fff; }}
</style>
</head><body>
<div id="plot"></div>
<div id="legend"><b>{title}</b><br>Drag to rotate, scroll to zoom, double-click to reset. Per-tree visibility toggles in the legend.</div>
<script>
  var traces = {traces_json};
  var layout = {{
    paper_bgcolor: '#0c0e13', plot_bgcolor: '#0c0e13',
    scene: {{
      bgcolor: '#0c0e13',
      xaxis: {{title: 'x (cm)', color: '#aaa', gridcolor: '#222'}},
      yaxis: {{title: 'y (cm)', color: '#aaa', gridcolor: '#222'}},
      zaxis: {{title: 'z (cm)', color: '#aaa', gridcolor: '#222'}},
      aspectmode: 'data',
    }},
    margin: {{l: 0, r: 0, t: 0, b: 0}},
    legend: {{x: 0.99, xanchor: 'right', y: 0.99, font: {{color: '#ddd'}}, bgcolor: 'rgba(0,0,0,0.55)'}},
  }};
  Plotly.newPlot('plot', traces, layout, {{responsive: true}});
</script>
</body></html>
"""


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tree_dir = os.path.abspath(sys.argv[1])
    default_html = os.path.join(tree_dir, "tree_viewer.html")
    out_html = sys.argv[2] if len(sys.argv) >= 3 else default_html
    top_n = int(sys.argv[3]) if len(sys.argv) >= 4 else 8000

    trees = discover_trees(tree_dir)
    if not trees:
        sys.exit(f"No tree CSVs in {tree_dir}")

    print(f"Tree dir : {tree_dir}")
    print(f"Output   : {out_html}")
    print(f"Top-N    : {top_n} segments per tree")
    print()

    traces = []
    total_segs = 0
    for i, (name, path) in enumerate(sorted(trees.items())):
        color = auto_color(name, i)
        n_total, top = load_top_segments(path, top_n)
        d_min = top[0][0] if top else 0.0
        d_max = top[-1][0] if top else 0.0
        total_segs += n_total
        print(f"  {name}: {n_total:,} segs total → top {len(top)} kept, d ∈ [{d_min:.1f}, {d_max:.1f}] μm")
        traces.append(build_plotly_trace(name, color, top))

    title = os.path.basename(tree_dir.rstrip("/"))
    html = build_html(traces, f"{title} — {total_segs:,} total segments, showing largest-d {top_n} per tree")
    os.makedirs(os.path.dirname(os.path.abspath(out_html)) or ".", exist_ok=True)
    with open(out_html, "w") as f:
        f.write(html)
    print(f"\nWrote {out_html} ({os.path.getsize(out_html) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
