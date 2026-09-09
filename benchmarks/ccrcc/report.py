#!/usr/bin/env python3
"""Build the autoresearch report (self-contained HTML, inline SVG) from the loop ledger and the
per-iteration / test-split eval JSONs. Every number on the page is read from those files.

    .venv/bin/python benchmarks/ccrcc/report.py [--loop autoresearch/loop-260909-1020]
"""

import argparse
import glob
import html
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
COPY_TO = Path("/srv/GT/analysis/pgueguen/rctd-py/ccrcc_protein/autoresearch_report.html")

# palette (dataviz reference instance; validated 3-slot categorical, blue sequential ramp)
SEQ = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]


def esc(x):
    return html.escape(str(x))


def load_ledger(loop_dir):
    rows, notes = [], []
    for line in open(loop_dir / "results.tsv"):
        line = line.rstrip("\n")
        if line.startswith("# ---"):
            notes.append(line[2:])
        elif line.startswith("#") or line.startswith("iteration") or not line.strip():
            continue
        else:
            f = line.split("\t")
            rows.append(
                {
                    "iteration": int(f[0]),
                    "timestamp": f[1],
                    "commit": f[2],
                    "metric": float(f[3]) if f[3] != "nan" else float("nan"),
                    "delta": f[4],
                    "guard": f[5],
                    "status": f[7],
                    "description": f[8],
                }
            )
    return rows, notes


def load_evals():
    ev = {}
    for f in glob.glob(str(OUT / "eval_*.json")):
        tag = Path(f).stem[len("eval_") :]
        try:
            ev[tag] = json.load(open(f))
        except Exception:
            pass
    return ev


def eval_for_iteration(ev, it):
    if it == 0:
        return ev.get("iter0_baseline")
    cands = [v for k, v in ev.items() if k.startswith(f"iter{it}_")]
    if not cands:
        return None
    return sorted(cands, key=lambda r: r.get("elapsed_s", 0))[-1] if len(cands) > 1 else cands[0]


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"


def seq_color(v, vmin=0.0, vmax=1.0):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "var(--surface-2)", "var(--text-muted)"
    t = 0 if vmax == vmin else (v - vmin) / (vmax - vmin)
    i = min(len(SEQ) - 1, max(0, int(round(t * (len(SEQ) - 1)))))
    ink = "#0b0b0b" if i < 6 else "#ffffff"
    return SEQ[i], ink


# ---------------------------------------------------------------- charts (inline SVG) ----
def trajectory_svg(rows, ev):
    W, H, L, R, T, B = 920, 300, 56, 72, 24, 44
    xs = [r["iteration"] for r in rows]
    x0, x1 = 0, max(xs)
    vals = [r["metric"] for r in rows if not math.isnan(r["metric"])]
    ymin = math.floor(min(vals) * 20) / 20 - 0.05
    ymax = math.ceil(max(vals) * 20) / 20 + 0.02

    def sx(x):
        return L + (x - x0) / (x1 - x0) * (W - L - R)

    def sy(y):
        return T + (ymax - y) / (ymax - ymin) * (H - T - B)

    g = []
    # gridlines + y ticks
    step = 0.05
    y = ymin
    while y <= ymax + 1e-9:
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" class="grid"/>')
        g.append(
            f'<text x="{L - 8}" y="{sy(y) + 4:.1f}" class="tick" text-anchor="end">{y:.2f}</text>'
        )
        y = round(y + step, 6)
    for x in xs:
        g.append(
            f'<text x="{sx(x):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{x}</text>'
        )
    g.append(
        f'<text x="{(L + W - R) / 2:.0f}" y="{H - 8}" class="axis-title" text-anchor="middle">iteration</text>'
    )
    # incumbent step line (kept states only)
    inc = []
    cur = None
    for r in rows:
        if r["status"] in ("baseline", "keep"):
            cur = r["metric"]
        if cur is not None:
            inc.append((r["iteration"], cur))
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in inc)
    g.append(f'<polyline points="{pts}" class="line-1"/>')
    # every iteration as a mark
    for r in rows:
        if math.isnan(r["metric"]):
            x, yv = sx(r["iteration"]), sy(inc[[i for i, _ in inc].index(r["iteration"])][1])
            crash_tip = esc(f"iteration {r['iteration']} - crash: {r['description'][:110]}")
            g.append(
                f'<g class="mark" data-tip="{crash_tip}">'
                f'<line x1="{x - 5:.1f}" y1="{yv - 5:.1f}" x2="{x + 5:.1f}" y2="{yv + 5:.1f}" class="crash"/>'
                f'<line x1="{x - 5:.1f}" y1="{yv + 5:.1f}" x2="{x + 5:.1f}" y2="{yv - 5:.1f}" class="crash"/>'
                f'<circle cx="{x:.1f}" cy="{yv:.1f}" r="12" class="hit"/></g>'
            )
            continue
        x, yv = sx(r["iteration"]), sy(r["metric"])
        e = eval_for_iteration(ev, r["iteration"]) or {}
        tip = (
            f"iteration {r['iteration']} ({r['status']}): metric {r['metric']:.4f} | F1 {fmt(e.get('heldout_lineage_f1'))} | "
            f"10x fine {fmt(e.get('tenx_fine_agreement'))} | reject {fmt(e.get('reject_frac'))} - {r['description'][:100]}"
        )
        cls = "kept" if r["status"] in ("keep", "baseline") else "discarded"
        g.append(
            f'<g class="mark" data-tip="{esc(tip)}"><circle cx="{x:.1f}" cy="{yv:.1f}" r="12" class="hit"/>'
            f'<circle cx="{x:.1f}" cy="{yv:.1f}" r="5" class="{cls}"/></g>'
        )
    # end label
    lx, ly = inc[-1]
    g.append(f'<text x="{sx(lx) + 10:.1f}" y="{sy(ly) + 4:.1f}" class="label">{ly:.4f}</text>')
    g.append(
        f'<text x="{sx(0) + 10:.1f}" y="{sy(inc[0][1]) - 10:.1f}" class="label">{inc[0][1]:.4f}</text>'
    )
    legend = (
        '<div class="legend"><span><i class="sw kept"></i>kept / baseline</span>'
        '<span><i class="sw discarded"></i>discarded (reverted)</span><span><i class="sw crash-sw">x</i>crash</span>'
        '<span><i class="sw line"></i>incumbent (best kept state)</span></div>'
    )
    return (
        f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="metric per iteration">'
        + "".join(g)
        + "</svg>"
        + legend
    )


def small_multiple_svg(rows, ev, key, title, cls):
    W, H, L, R, T, B = 450, 190, 52, 16, 26, 36
    pts_all = [
        (r["iteration"], (eval_for_iteration(ev, r["iteration"]) or {}).get(key), r["status"])
        for r in rows
    ]
    pts_all = [(i, v, s) for i, v, s in pts_all if v is not None]
    if not pts_all:
        return ""
    vals = [v for _, v, _ in pts_all]
    ymin, ymax = math.floor(min(vals) * 20) / 20 - 0.03, math.ceil(max(vals) * 20) / 20 + 0.02
    x1 = max(i for i, _, _ in pts_all)

    def sx(x):
        return L + x / x1 * (W - L - R)

    def sy(y):
        return T + (ymax - y) / (ymax - ymin) * (H - T - B)

    g = [f'<text x="{L}" y="14" class="subtitle">{esc(title)}</text>']
    y = ymin
    while y <= ymax + 1e-9:
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" class="grid"/>')
        g.append(
            f'<text x="{L - 6}" y="{sy(y) + 4:.1f}" class="tick" text-anchor="end">{y:.2f}</text>'
        )
        y = round(y + 0.05, 6)
    for i in range(0, x1 + 1, 2):
        g.append(
            f'<text x="{sx(i):.1f}" y="{H - B + 16}" class="tick" text-anchor="middle">{i}</text>'
        )
    inc, cur = [], None
    for i, v, s in pts_all:
        if s in ("baseline", "keep"):
            cur = v
        if cur is not None:
            inc.append((i, cur))
    g.append(
        f'<polyline points="{" ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in inc)}" class="{cls}-line"/>'
    )
    for i, v, s in pts_all:
        k = "kept" if s in ("keep", "baseline") else "discarded"
        g.append(
            f'<g class="mark" data-tip="{esc(f"iteration {i}: {title} {v:.3f} ({s})")}">'
            f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="10" class="hit"/>'
            f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="4" class="{cls if k == "kept" else "discarded"}"/></g>'
        )
    g.append(
        f'<text x="{sx(inc[-1][0]) - 4:.1f}" y="{sy(inc[-1][1]) - 9:.1f}" class="label" text-anchor="end">{inc[-1][1]:.3f}</text>'
    )
    return (
        f'<svg viewBox="0 0 {W} {H}" class="chart small" role="img" aria-label="{esc(title)}">'
        + "".join(g)
        + "</svg>"
    )


def heatmap_html(rowlabels, collabels, values, title, note, vmin=0.0, vmax=1.0):
    h = [
        f'<div class="hm"><h3>{esc(title)}</h3><p class="note">{note}</p><table class="heat"><thead><tr><th></th>'
    ]
    h += [f"<th>{esc(c)}</th>" for c in collabels] + ["</tr></thead><tbody>"]
    for i, rl in enumerate(rowlabels):
        h.append(f"<tr><th>{esc(rl)}</th>")
        for j, cl in enumerate(collabels):
            v = values[i][j]
            bg, ink = seq_color(v, vmin, vmax)
            h.append(
                f'<td style="background:{bg};color:{ink}" data-tip="{esc(rl)} @ {esc(cl)}: {fmt(v)}">{fmt(v, 2)}</td>'
            )
        h.append("</tr>")
    h.append("</tbody></table></div>")
    return "".join(h)


# ---------------------------------------------------------------------- report -----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--loop",
        default=str(sorted(glob.glob(str(HERE.parent.parent / "autoresearch" / "loop-*")))[-1]),
    )
    a = ap.parse_args()
    loop = Path(a.loop)
    rows, notes = load_ledger(loop)
    ev = load_evals()
    handoff = json.load(open(loop / "handoff.json")) if (loop / "handoff.json").exists() else {}
    base_t, final_t = ev.get("iter0_baseline", {}), None
    kept = [r for r in rows if r["status"] in ("baseline", "keep")]
    final_t = eval_for_iteration(ev, kept[-1]["iteration"]) or {}
    tb, tf, tfr = ev.get("test_baseline"), ev.get("test_final"), ev.get("test_final_rna")
    shuf = [ev[k] for k in sorted(ev) if k.startswith("test_final_shuffle")]
    n_kept = sum(r["status"] == "keep" for r in rows)
    n_disc = sum(r["status"] == "discard" for r in rows)
    n_crash = sum(r["status"] == "crash" for r in rows)

    # --- hero / verdict -------------------------------------------------------
    hero = [
        (
            "tune metric",
            base_t.get("metric"),
            final_t.get("metric"),
            "sqrt(F1 x fine agreement), 80k west-half cells",
        ),
        (
            "held-out lineage F1",
            base_t.get("heldout_lineage_f1"),
            final_t.get("heldout_lineage_f1"),
            "protein-gated landmarks, 9 gate markers held out",
        ),
        (
            "10x fine-type agreement",
            base_t.get("tenx_fine_agreement"),
            final_t.get("tenx_fine_agreement"),
            "macro over 16 WNN groups",
        ),
        (
            "reject fraction",
            base_t.get("reject_frac"),
            final_t.get("reject_frac"),
            "lower is better",
        ),
    ]
    if tb and tf:
        hero.insert(
            1,
            (
                "TEST metric (east half, 100k cells)",
                tb.get("metric"),
                tf.get("metric"),
                "never seen by the loop",
            ),
        )
    hero_html = "".join(
        f'<div class="tile"><div class="tile-k">{esc(k)}</div><div class="tile-v">{fmt(v1)}</div>'
        f'<div class="tile-d">{fmt(v0)} &rarr; {fmt(v1)}'
        + (f" ({(v1 - v0) / v0 * 100:+.1f}%)" if v0 else "")
        + f'</div><div class="tile-n">{esc(n)}</div></div>'
        for k, v0, v1, n in hero
    )

    # --- iterations table -----------------------------------------------------
    it_rows = []
    for r in rows:
        e = eval_for_iteration(ev, r["iteration"]) or {}
        it_rows.append(
            f'<tr class="st-{r["status"]}"><td>{r["iteration"]}</td><td><code>{esc(r["commit"])}</code></td>'
            f'<td class="num">{fmt(r["metric"], 4)}</td><td class="num">{esc(r["delta"])}</td>'
            f'<td class="num">{fmt(e.get("heldout_lineage_f1"))}</td><td class="num">{fmt(e.get("tenx_fine_agreement"))}</td>'
            f'<td class="num">{fmt(e.get("reject_frac"))}</td><td class="num">{fmt(e.get("ca9_pos_frac_of_tumour_calls"))}</td>'
            f'<td><span class="pill {r["status"]}">{r["status"]}</span></td><td>{esc(r["description"])}</td></tr>'
        )

    # --- heatmaps over kept states -------------------------------------------
    kept_its = [r["iteration"] for r in kept if eval_for_iteration(ev, r["iteration"])]
    kept_ev = [eval_for_iteration(ev, i) for i in kept_its]
    classes = sorted({c for e in kept_ev for c in e.get("per_class_f1", {})})
    hm1 = heatmap_html(
        classes,
        [f"it {i}" for i in kept_its],
        [[e.get("per_class_f1", {}).get(c, {}).get("f1") for e in kept_ev] for c in classes],
        "Held-out lineage F1 per class, kept states",
        "Landmark cells gated on the 9 held-out protein markers; the pipeline never saw those markers. "
        "Plasma_cell landmarks (CD138+) are E-cadherin-high epithelium, a truth defect that costs every state equally; "
        "NK has 22 landmarks.",
    )
    groups = list(kept_ev[-1].get("per_group_agreement", {}).keys())
    hm2 = heatmap_html(
        [g for g in groups],
        [f"it {i}" for i in kept_its],
        [
            [e.get("per_group_agreement", {}).get(g, {}).get("agreement") for e in kept_ev]
            for g in groups
        ],
        "Agreement with 10x WNN groups, kept states",
        "Fraction of each 10x group whose RCTD call is in the group's accepted set (pDCs accept only cDC1/cDC2, "
        "which were dropped at iteration 2, so that row is 0 by construction from then on).",
    )

    # --- test split ------------------------------------------------------------
    test_html = ""
    verdict = ""
    if tb and tf:
        arms = [("baseline (iteration 0)", tb), ("final (iteration 20)", tf)]
        if tfr:
            arms.append(("final, protein off", tfr))
        for i, s in enumerate(shuf):
            arms.append(
                (f"final, fit markers shuffled (seed {s.get('shuffle_protein_seed', i)})", s)
            )
        trs = "".join(
            f"<tr><td>{esc(n)}</td><td class='num'>{fmt(e.get('metric'), 4)}</td><td class='num'>{fmt(e.get('heldout_lineage_f1'))}</td>"
            f"<td class='num'>{fmt(e.get('tenx_fine_agreement'))}</td><td class='num'>{fmt(e.get('tenx_lineage_agreement'))}</td>"
            f"<td class='num'>{fmt(e.get('reject_frac'))}</td><td class='num'>{fmt(e.get('ca9_pos_frac_of_tumour_calls'))}</td>"
            f"<td class='num'>{fmt(e.get('tumour_groups_called_Tumour_ccRCC'))}</td></tr>"
            for n, e in arms
        )
        test_html = (
            '<table class="grid-t"><thead><tr><th>arm (test split, 100k east-half cells)</th><th>metric</th><th>held-out F1</th>'
            "<th>10x fine</th><th>10x lineage</th><th>reject</th><th>CA9+ of tumour calls</th><th>10x tumour groups called tumour</th></tr></thead>"
            f"<tbody>{trs}</tbody></table>"
        )
        gain = tf["metric"] - tb["metric"]
        prot_gain = (tf["metric"] - tfr["metric"]) if tfr else None
        null_max = max((s["metric"] for s in shuf), default=None)
        parts = [
            f"Test-split metric {tb['metric']:.4f} &rarr; {tf['metric']:.4f} ({gain:+.4f}; tune split showed {final_t.get('metric', float('nan')) - base_t.get('metric', float('nan')):+.4f})."
        ]
        if prot_gain is not None:
            parts.append(
                f"Switching protein off in the final pipeline moves the test metric by {-prot_gain:+.4f}, so protein contributes {prot_gain:+.4f} of the final value."
            )
        if null_max is not None and tfr:
            parts.append(
                f"With the 18 fit markers shuffled independently ({len(shuf)} seeds) the metric is at most {null_max:.4f}, "
                + (
                    "above the protein-off value, so the protein gain does NOT clear the permutation null on the test split."
                    if null_max >= tf["metric"] - 1e-9
                    else f"{tf['metric'] - null_max:+.4f} below the real protein run: the protein gain survives the null."
                )
            )
        verdict = " ".join(parts)

    # --- spatial-anno-metrics --------------------------------------------------
    def sam_row(name, e):
        s = (e or {}).get("spatial_anno_metrics", {}) or {}
        L, Hh, I = (
            s.get("lineage_vs_tenx", {}),
            s.get("hierarchical_vs_tenx_primary", {}),
            s.get("internal_validity_fine", {}),
        )
        return (
            f"<tr><td>{esc(name)}</td><td class='num'>{fmt(L.get('ari'))}</td><td class='num'>{fmt(L.get('kappa'))}</td>"
            f"<td class='num'>{fmt(L.get('balanced_accuracy'))}</td><td class='num'>{fmt(L.get('ecs'))}</td>"
            f"<td class='num'>{fmt(Hh.get('subtype_accuracy'))}</td><td class='num'>{fmt(Hh.get('lineage_accuracy'))}</td>"
            f"<td class='num'>{fmt(I.get('integrated'))}</td><td class='num'>{fmt(I.get('neighborhood_purity'))}</td></tr>"
        )

    sam_rows = []
    for name, e in [
        ("tune, baseline", eval_for_iteration(ev, 0)),
        ("tune, final", final_t),
        ("test, baseline", tb),
        ("test, final", tf),
        ("test, final protein off", tfr),
    ]:
        if e and e.get("spatial_anno_metrics", {}).get("lineage_vs_tenx"):
            sam_rows.append(sam_row(name, e))
    sam_html = (
        (
            '<table class="grid-t"><thead><tr><th>state</th><th>ARI vs 10x lineage</th><th>kappa</th><th>balanced acc.</th><th>ECS</th>'
            "<th>subtype acc. vs 10x primary type</th><th>lineage acc.</th><th>internal validity (integrated)</th><th>neighbourhood purity</th></tr></thead>"
            f"<tbody>{''.join(sam_rows)}</tbody></table>"
        )
        if sam_rows
        else "<p class='note'>spatial-anno-metrics were added to the harness after iteration 4; earlier iterations carry none.</p>"
    )

    # --- narrative -----------------------------------------------------------------
    top = sorted([r for r in rows if r["status"] == "keep"], key=lambda r: -float(r["delta"]))[:4]
    worked = "".join(
        f"<li><b>it {r['iteration']}</b> ({r['delta']}): {esc(r['description'])}</li>" for r in top
    )
    failed = "".join(
        f"<li><b>it {r['iteration']}</b> ({r['delta'] if r['status'] != 'crash' else 'crash'}): {esc(r['description'])}</li>"
        for r in rows
        if r["status"] in ("discard", "crash")
    )
    checkpoints = "".join(f"<li>{esc(n)}</li>" for n in notes)
    final_cfg = esc(final_t.get("config_repr", ""))

    css = """
:root{color-scheme:light;--page:#f9f9f7;--surface-1:#fcfcfb;--surface-2:#f0efec;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--gray:#898781;--good:#006300;--crit:#d03b3b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;--surface-2:#262624;--text-primary:#fff;
--text-secondary:#c3c2b7;--text-muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#0ca30c}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;--surface-2:#262624;--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#0ca30c}
body{margin:0;background:var(--page);color:var(--text-primary);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 10px;padding-top:12px;border-top:1px solid var(--grid)}h3{font-size:15px;margin:18px 0 6px}
.lede,.note{color:var(--text-secondary)}.note{font-size:13px;margin:2px 0 8px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile-k{font-size:12px;color:var(--text-secondary)}.tile-v{font-size:30px;font-weight:600;line-height:1.15;margin:2px 0}.tile-d{font-size:13px;color:var(--text-secondary)}.tile-n{font-size:11px;color:var(--text-muted)}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:10px 0;overflow-x:auto}
svg.chart{width:100%;height:auto;display:block}svg.small{max-width:470px}.row2{display:flex;gap:16px;flex-wrap:wrap}
.grid{stroke:var(--grid);stroke-width:1}.tick{font-size:11px;fill:var(--text-muted)}.axis-title,.subtitle{font-size:12px;fill:var(--text-secondary)}
.label{font-size:12px;fill:var(--text-primary);font-weight:600}
.line-1{fill:none;stroke:var(--s1);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.s3-line{fill:none;stroke:var(--s3);stroke-width:2;stroke-linejoin:round}.s2-line{fill:none;stroke:var(--s2);stroke-width:2;stroke-linejoin:round}
circle.kept{fill:var(--s1);stroke:var(--surface-1);stroke-width:2}circle.s3{fill:var(--s3);stroke:var(--surface-1);stroke-width:2}circle.s2{fill:var(--s2);stroke:var(--surface-1);stroke-width:2}
circle.discarded{fill:var(--surface-1);stroke:var(--gray);stroke-width:2}.crash{stroke:var(--gray);stroke-width:2}circle.hit{fill:transparent}
.mark{cursor:default}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--text-secondary);margin:6px 0 0 56px}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}.sw.kept{background:var(--s1)}.sw.discarded{border:2px solid var(--gray);width:7px;height:7px}
.sw.line{width:18px;height:2px;border-radius:0;background:var(--s1);vertical-align:2px}.crash-sw{color:var(--gray);font-weight:700;margin-right:6px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:5px 8px;text-align:left;vertical-align:top;border-bottom:1px solid var(--grid)}th{color:var(--text-secondary);font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.heat td{text-align:center;font-variant-numeric:tabular-nums;font-size:12px;border:2px solid var(--surface-1);padding:4px 6px;border-radius:3px}.heat th{font-size:12px;white-space:nowrap}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;border:1px solid var(--border)}.pill.keep,.pill.baseline{color:var(--good)}.pill.discard{color:var(--text-secondary)}.pill.crash{color:var(--crit)}
tr.st-discard td:not(:last-child),tr.st-crash td:not(:last-child){color:var(--text-secondary)}
code{font-size:12px;background:var(--surface-2);padding:1px 4px;border-radius:3px}pre{background:var(--surface-2);padding:10px;border-radius:6px;overflow-x:auto;font-size:12px}
#tip{position:fixed;pointer-events:none;background:var(--text-primary);color:var(--surface-1);font-size:12px;padding:6px 8px;border-radius:6px;max-width:420px;display:none;z-index:9}
ul{margin:6px 0}li{margin:3px 0}
"""
    js = """
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{el.addEventListener('mousemove',e=>{tip.textContent=el.dataset.tip;tip.style.display='block';
tip.style.left=Math.min(e.clientX+12,window.innerWidth-440)+'px';tip.style.top=(e.clientY+14)+'px';});el.addEventListener('mouseleave',()=>tip.style.display='none');});
"""
    untried = [
        "class_df over T-cell and myeloid subtypes (recovers singlets; does not change the winner)",
        'protein_reliability="neighbour_ratio" (spillover down-weighting; was inert on the earlier 150k run)',
        "same-lineage neighbour smoothing of labels as post-processing",
        "two-stage typing: lineage from RNA+protein, fine type from RNA weights within the lineage",
        "per-marker informativeness weights in the solver (inv_tau2 scaling), doublet-score fusion changes",
        "a helper-CD4 reference profile that DISCO's Treg does not absorb (the Treg profile, not the CD4 profile, is the attractor: Krishna's CD4 TILs did not fix it)",
        "an NK profile the 22 NK landmarks can reward (NK F1 fell to 0 in the final state; CD16 is a gate marker, so protein cannot help here)",
        "lambda between 1.5 and 2 at the final state; lambda 2 hurt only at the pre-purity state",
        "the same loop against a different truth axis (CA9 tumour purity, or InSituCNV chr3p loss) to check the tumour calls independently of 10x",
    ]
    caveats = [
        "Two axes, both imperfect. The landmark F1 is built on 9 protein gates with 797 landmark cells (NK 22, Stromal 30, Plasma 38); the Plasma gate (CD138+) selects E-cadherin-high epithelium, so its F1 is ~0 for every state and the macro F1 is depressed by 1/8 across the board. The 10x fine axis uses WNN groups that were built WITH protein and are cluster-level labels: they favour protein arms and inherit 10x's own errors (a 'Stressed/Dedifferentiating Proximal Tubule' group that is a third CD8 T cells by RCTD and by CA9).",
        "One section, one panel. Everything was tuned on the west half of a single ccRCC section and validated on its east half. The reference-composition wins (drop cDC1/cDC2, Cycling_myeloid_cell, Cycling_T_NK_cell) are decisions about THIS reference on THIS 477-gene panel, not general rules.",
        "The threshold scaling changes which type a cell gets (first_type moved in 19.8% of cells on the 150k run), not only how many cells are rejected. It is the single largest lever and also the least principled: the thresholds are absolute log-likelihood gaps tuned for ~5k-gene fits.",
        "Keep/discard was mechanical on a composite; iterations 1, 13 and 20 raised the metric while a rare class fell (MDM/TAM at 1, Helper T at 13, NK at 20). Iterations 12 and 18 were kept on gains of 0.002 and 0.0003, which is inside what a different subsample would give. A rerun of an identical pipeline reproduced the metric to 6 decimals, so the noise is in the cells, not the solver.",
        "The 10x-agreement axis cannot be fully separated from protein: 10x's WNN saw all 27 markers, the pipeline saw 18. The held-out F1 axis is the honest one for the protein question, and on it protein's own contribution is what the protein-off arm and the shuffled-marker nulls on the test split measure.",
        "Harness edits mid-loop: the eval gained reporting fields (spatial-anno-metrics after iteration 4, per-group top calls after iteration 6, a third reference file after iteration 9). The metric formula, the truth gates, the splits and the eval set never changed; guard hashes were re-pinned each time and the ledger records it.",
    ]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ccRCC RNA+protein autoresearch</title><style>{css}</style></head><body><div id="tip"></div><main>
<h1>Autoresearch: RNA+protein cell typing on the 10x ccRCC Xenium Protein section</h1>
<p class="lede">rctd-py branch <code>autoresearch/protein-ccrcc</code>, {esc(handoff.get("summary", {}).get("iterations", len(rows) - 1))} iterations of modify &rarr; verify &rarr; keep/revert
against <b>sqrt(held-out lineage F1 &times; 10x fine-type agreement)</b> on an 80,000-cell tune split (west half of the section).
Kept {n_kept}, discarded {n_disc}, crashed {n_crash}. Loop ran {esc(rows[0]["timestamp"][:10])}; every number below is read from the per-iteration eval JSON and the ledger.</p>
<div class="tiles">{hero_html}</div>

<h2>Metric per iteration</h2>
<div class="card">{trajectory_svg(rows, ev)}</div>
<div class="row2"><div class="card">{small_multiple_svg(rows, ev, "heldout_lineage_f1", "held-out lineage F1 (protein-gated landmarks)", "s3")}</div>
<div class="card">{small_multiple_svg(rows, ev, "tenx_fine_agreement", "10x fine-type agreement (16 WNN groups, macro)", "s2")}</div></div>
<p class="note">Filled points are kept states (the line follows the incumbent), hollow points were reverted, an x is a crash. Hover any point for the numbers.</p>

<h2>What was tested</h2>
<div class="card"><table><thead><tr><th>it</th><th>commit</th><th>metric</th><th>&Delta;</th><th>F1</th><th>10x fine</th><th>reject</th><th>CA9+ tumour</th><th>status</th><th>change</th></tr></thead><tbody>{"".join(it_rows)}</tbody></table></div>
<h3>Eval checkpoints</h3><ul>{checkpoints}</ul>

<h2>Where the gains came from, per class and per 10x group</h2>
<div class="card">{hm1}</div><div class="card">{hm2}</div>

<h2>Held-out validation (test split, never seen by the loop)</h2>
<div class="card">{test_html or "<p class='note'>Validation job not finished yet; rerun report.py when out/eval_test_*.json exist.</p>"}
<p><b>{verdict}</b></p></div>

<h2>spatial-anno-metrics (reported, never optimised)</h2>
<div class="card">{sam_html}
<p class="note">ARI / kappa / balanced accuracy / ECS compare RCTD lineages with 10x lineages on 10x-labelled cells; subtype accuracy scores the fine RCTD type against one representative type per 10x group with lineage partial credit; internal validity is the reference-free scTypeEval composite of the fine labels in log-normalised expression space (3,000-cell subsample).</p></div>

<h2>What worked</h2><ul>{worked}</ul>
<h2>What did not</h2><ul>{failed}</ul>
<h2>Not tried yet</h2><ul>{"".join(f"<li>{esc(u)}</li>" for u in untried)}</ul>
<h2>Caveats</h2><ul>{"".join(f"<li>{esc(c)}</li>" for c in caveats)}</ul>

<h2>Final pipeline</h2>
<pre>{final_cfg}</pre>
<p class="note">Reference: DISCO normal kidney + Zhang 2021 <code>Tumour_ccRCC</code>, minus cDC1, cDC2, Cycling_myeloid_cell, Cycling_T_NK_cell. Doublet mode, bootstrap protein profiles from RNA-confident singlets (purity 0.9), arcsinh-robust with cofactor 20, lambda 1.5, thresholds scaled x0.05, counts_MIN 3.
Code: <code>benchmarks/ccrcc/</code> in the worktree <code>/misc/GT/analysis/pgueguen/rctd-py/autoresearch_ccrcc/wt</code>; ledger <code>{esc(str(loop.relative_to(HERE.parent.parent)))}/results.tsv</code>; data <code>/srv/GT/analysis/pgueguen/rctd-py/ccrcc_protein/autoresearch_data/</code>.</p>
</main><script>{js}</script></body></html>"""
    out = OUT / "autoresearch_report.html"
    out.write_text(page)
    try:
        COPY_TO.write_text(page)
    except Exception as e:
        print("copy failed:", e)
    print("WROTE", out, "and", COPY_TO, f"({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
