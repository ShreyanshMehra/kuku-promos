"""
Web front end for kuku-promos (Streamlit).

    streamlit run app.py

Three sources:
  * Real run   - the scores of the full 50-episode run, numbers and timestamps only
                 (the pack is Kuku's, so no script text is published).
  * Sample     - an INVENTED 3-episode show, runnable offline with no key.
  * Upload     - the user's own pack, held in a temporary folder for the session.

The pipeline is run.py, started as a subprocess, so the app and the command line behave
identically.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from sample_data import write_sample_pack  # noqa: E402
from webdata import PERSONAS, build_clips  # noqa: E402

st.set_page_config(page_title="Find the Promos", page_icon="🎬", layout="wide")

ACCENT = "#4f46e5"
WARM = "#f59e0b"
SRC_REAL = "Real run: 50 episodes (scores only)"
SRC_SAMPLE = "Built-in sample (invented show)"
SRC_UPLOAD = "Upload my own pack"

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1250px;}
    div[data-testid="stMetricValue"] {font-size: 1.7rem;}
    .pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem;
           background:#eef2ff; color:#3730a3; margin-right:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- helpers
def _workdir() -> str:
    if "work" not in st.session_state:
        st.session_state.work = tempfile.mkdtemp(prefix="kuku_promos_")
    return st.session_state.work


def _reset_workdir() -> str:
    old = st.session_state.pop("work", None)
    if old:
        shutil.rmtree(old, ignore_errors=True)
    return _workdir()


def _save_upload(files, dest: str) -> str:
    os.makedirs(dest, exist_ok=True)
    for f in files:
        with open(os.path.join(dest, os.path.basename(f.name)), "wb") as fh:
            fh.write(f.getbuffer())
    names = os.listdir(dest)
    if len(names) == 1 and names[0].lower().endswith(".zip"):
        return os.path.join(dest, names[0])
    return dest


def _run(args, env_extra, box) -> int:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **env_extra}
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "run.py"), *args], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip())
        box.code("\n".join(lines[-28:]), language="text")
    proc.wait()
    return proc.returncode


def _zip_dir(path: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _dirs, names in os.walk(path):
            for n in names:
                full = os.path.join(base, n)
                z.write(full, os.path.relpath(full, path))
    return buf.getvalue()


def _load_dataset(source: str):
    """Return the dataset dict for the current source, or None."""
    if source == SRC_REAL:
        with open(os.path.join(ROOT, "demo", "real_run_clips.json"), encoding="utf-8") as fh:
            ds = json.load(fh)
        with open(os.path.join(ROOT, "demo", "real_run_metrics.md"), encoding="utf-8") as fh:
            ds["metrics_md"] = fh.read()
        ds["out"] = None
        return ds
    out = st.session_state.get("out")
    if out and os.path.exists(os.path.join(out, "promos.json")):
        ds = build_clips(os.path.join(out, "promos.json"), with_text=True)
        mpath = os.path.join(out, "metrics.md")
        ds["metrics_md"] = open(mpath, encoding="utf-8").read() if os.path.exists(mpath) else ""
        ds["out"] = out
        return ds
    return None


def _fmt(x, pct=False, nd=2):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x * 100:.0f}%" if pct else f"{x:.{nd}f}"


def _clip_label(c) -> str:
    return f"#{int(c['rank'])}  ·  ep {int(c['episode'])}  ·  {c['start']} to {c['end']}"


# ----------------------------------------------------------------------------- tabs
def tab_overview(ds, df):
    meta = ds["meta"]
    n_eps = int(meta.get("episodes") or df["episode"].max())
    covered = df["episode"].nunique()
    k = st.columns(5)
    cards = [
        ("Promos shortlisted", f"{len(df)}", "20-60s each"),
        ("Episodes covered", f"{covered} of {n_eps}", "gaps = quiet episodes"),
        ("Post-first feed", f"{int((df['feed'] == 'post_first').sum())}", "early, no spoilers"),
        ("Median retention", _fmt(df["retention"].median(), pct=True) if df["retention"].notna().any() else "n/a",
         "simulated viewers"),
        ("Confidence tier", str(meta.get("tier") or "?").upper(), "what actually ran"),
    ]
    for col, (label, value, hint) in zip(k, cards):
        with col.container(border=True):
            st.metric(label, value)
            st.caption(hint)

    st.markdown("##### Post these first")
    top = df.sort_values("rank").head(3)
    cols = st.columns(3)
    for col, (_, c) in zip(cols, top.iterrows()):
        with col.container(border=True):
            st.markdown(f"**#{int(c['rank'])} · Episode {int(c['episode'])}**")
            st.markdown(f"`{c['start']}` to `{c['end']}` · {c['duration_s']:.0f}s · {c['archetype']}")
            st.progress(min(1.0, float(c["final_score"])), text=f"score {c['final_score']:.2f}")
            if ds["with_text"]:
                st.markdown(f"**Hook:** {c.get('hook_line', '')}")
                st.markdown(f"**Ends on:** {c.get('cliffhanger_line', '')}")
            else:
                st.caption("Script text is not published for the real pack.")

    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**Promos per episode**")
        eps = range(int(df["episode"].min()), int(df["episode"].max()) + 1)
        counts = df["episode"].value_counts().reindex(eps, fill_value=0).rename_axis("episode").reset_index(name="promos")
        st.altair_chart(
            alt.Chart(counts).mark_bar(color=ACCENT).encode(
                x=alt.X("episode:O", title="Episode", axis=alt.Axis(labelAngle=0, labelOverlap=True)),
                y=alt.Y("promos:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                tooltip=["episode", "promos"]).properties(height=220),
            use_container_width=True)
    with b.container(border=True):
        st.markdown("**Score by rank** (post-first vs deeper cuts)")
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x=alt.X("rank:O", title="Rank", axis=alt.Axis(labelOverlap=True, labelAngle=0)),
                y=alt.Y("final_score:Q", title=None),
                color=alt.Color("feed:N", title=None, scale=alt.Scale(domain=["post_first", "deeper_cut"], range=[ACCENT, WARM]),
                                legend=alt.Legend(orient="bottom")),
                tooltip=["rank", "episode", "feed", "final_score"]).properties(height=220),
            use_container_width=True)

    c, d = st.columns(2)
    with c.container(border=True):
        st.markdown("**How the simulated viewers rated the clips**")
        rd = df.dropna(subset=["retention"])
        if rd.empty:
            st.caption("No panel scores in this run (offline engine).")
        else:
            st.altair_chart(
                alt.Chart(rd).mark_bar(color=ACCENT).encode(
                    x=alt.X("retention:Q", bin=alt.Bin(maxbins=12), title="Retention score"),
                    y=alt.Y("count()", title=None)).properties(height=200),
                use_container_width=True)
    with d.container(border=True):
        st.markdown("**Blind test: tool vs the last 40 seconds**")
        ev = ds.get("eval")
        if ev:
            rows = []
            for g in ev["blind"]["groups"][:2]:
                for m in ("hook", "cliffhanger", "standalone"):
                    rows.append({"group": g["group"], "metric": m, "score": g[m]})
            st.altair_chart(
                alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
                    x=alt.X("group:N", axis=None, title=None),
                    y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 2]), title="0-2"),
                    color=alt.Color("group:N", title=None,
                                    scale=alt.Scale(domain=["Tool, all 15", "Baseline: last 40s"], range=[ACCENT, WARM]),
                                    legend=alt.Legend(orient="bottom")),
                    column=alt.Column("metric:N", title=None, sort=["hook", "cliffhanger", "standalone"]),
                    tooltip=["group", "metric", "score"]).properties(height=170, width=95),
                use_container_width=False)
            st.caption(f"Rated by {ev['rater']}. Open the Quality tab for the caveats.")
        else:
            st.caption("The blind test is run separately (`python src/evaluate.py sample`, then `score`). "
                       "See the Quality tab.")


def tab_ranked(ds, df):
    f1, f2 = st.columns(2)
    f3, f4 = st.columns(2)
    feeds = f1.multiselect("Feed", sorted(df["feed"].dropna().unique()), default=[])
    arch = f2.multiselect("Archetype", sorted(df["archetype"].dropna().unique()), default=[])
    lo, hi = int(df["episode"].min()), int(df["episode"].max())
    rng = f3.slider("Episodes", lo, hi, (lo, hi)) if hi > lo else (lo, hi)
    min_score = f4.slider("Min score", 0.0, float(max(1.0, df["final_score"].max())), 0.0, 0.05)
    v = df[(df["episode"].between(*rng)) & (df["final_score"] >= min_score)]
    if feeds:
        v = v[v["feed"].isin(feeds)]
    if arch:
        v = v[v["archetype"].isin(arch)]
    st.caption(f"{len(v)} of {len(df)} promos")
    cols = ["rank", "episode", "start", "end", "duration_s", "feed", "archetype"]
    if ds["with_text"]:
        cols += ["hook_line", "cliffhanger_line"]
    cols += ["final_score", "retention", "hook_survival", "watch_through", "tap_next", "evidence_support"]
    cfg = {
        "final_score": st.column_config.ProgressColumn("score", min_value=0, max_value=1.1, format="%.2f"),
        "retention": st.column_config.ProgressColumn("retention", min_value=0, max_value=1, format="%.2f"),
        "hook_survival": st.column_config.ProgressColumn("hook held 3s", min_value=0, max_value=1, format="%.2f"),
        "watch_through": st.column_config.ProgressColumn("watch-through", min_value=0, max_value=1, format="%.2f"),
        "tap_next": st.column_config.ProgressColumn("tap next ep", min_value=0, max_value=1, format="%.2f"),
        "evidence_support": st.column_config.NumberColumn("evidence held up", format="%.2f"),
        "duration_s": st.column_config.NumberColumn("secs", format="%.0f"),
    }
    st.dataframe(v[[c for c in cols if c in v.columns]], use_container_width=True, hide_index=True,
                 column_config=cfg, height=560)


def tab_detail(ds, df):
    clips = {int(c["rank"]): c for c in ds["clips"]}
    pick = st.selectbox("Promo", [int(r) for r in df["rank"]], format_func=lambda r: _clip_label(clips[r]))
    c = clips[pick]
    top = st.columns([2, 1])
    with top[0]:
        st.markdown(f"### Episode {int(c['episode'])}, {c['start']} to {c['end']}")
        st.markdown(f"<span class='pill'>{c['duration_s']:.0f}s</span><span class='pill'>{c['archetype']}</span>"
                    f"<span class='pill'>{c['feed']}</span><span class='pill'>{c['source']}</span>",
                    unsafe_allow_html=True)
        if ds["with_text"]:
            st.markdown(f"**Hook:** {c.get('hook_line', '')}")
            st.markdown(f"**Ends on:** {c.get('cliffhanger_line', '')}")
            st.markdown(f"**Why it works:** {c.get('why', '')}")
        else:
            st.caption("Hook, cliffhanger and script text are not published for the real pack.")
    with top[1]:
        with st.container(border=True):
            st.metric("Final score", _fmt(c["final_score"]))
            st.caption(f"retention x recruitment {_fmt(c.get('recruitment'))}")

    m = st.columns(4)
    m[0].metric("Retention", _fmt(c["retention"], pct=True))
    m[1].metric("Still watching at 3s", _fmt(c["hook_survival"], pct=True))
    m[2].metric("Share of clip watched", _fmt(c["watch_through"], pct=True))
    m[3].metric("Would tap next episode", _fmt(c["tap_next"], pct=True))

    curve = c.get("curve") or []
    with st.container(border=True):
        st.markdown("**Tension through the clip, and where each simulated viewer leaves**")
        if curve:
            cdf = pd.DataFrame({"second": range(len(curve)), "tension": curve})
            line = alt.Chart(cdf).mark_area(opacity=0.25, color=ACCENT, line={"color": ACCENT}).encode(
                x=alt.X("second:Q", title="Seconds into the clip"), y=alt.Y("tension:Q", title="Tension (0-4)"))
            sw = pd.DataFrame([{"second": v["swipe_at_s"], "viewer": PERSONAS.get(v["persona"], str(v["persona"]))}
                               for v in c["viewers"] if v.get("swipe_at_s") is not None])
            chart = line
            if not sw.empty:
                chart = line + alt.Chart(sw).mark_rule(strokeWidth=3).encode(
                    x="second:Q", color=alt.Color("viewer:N", title="Swiped away"))
            st.altair_chart(chart.properties(height=230), use_container_width=True)
            if sw.empty:
                st.caption("All simulated viewers watched to the end.")
        else:
            st.caption("No tension curve stored for this clip.")

    st.markdown("**The three simulated viewers**")
    rows = []
    for v in c["viewers"]:
        row = {
            "viewer": PERSONAS.get(v["persona"], str(v["persona"])),
            "outcome": "Watched to the end" if v.get("swipe_at_s") is None else f"Swiped at {v['swipe_at_s']:.1f}s",
            "would tap next ep": v.get("tap_next"),
            "evidence check": v.get("evidence_status"),
        }
        if ds["with_text"]:
            row["reason"] = v.get("reason")
            row["quoted line"] = v.get("quote")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"would tap next ep": st.column_config.ProgressColumn(
                     "would tap next ep", min_value=0, max_value=1, format="%.2f")})
    st.caption("An LLM role-playing viewers, not audience data. 'Evidence check' = was the quoted line really in the clip.")

    if ds["with_text"] and ds.get("out") and os.path.exists(os.path.join(ds["out"], "report.html")):
        with st.expander("Script text and tension curve for every clip"):
            with open(os.path.join(ds["out"], "report.html"), encoding="utf-8") as fh:
                components.html(fh.read(), height=700, scrolling=True)


def tab_quality(ds, df):
    ev = ds.get("eval")
    st.warning("Judged from screenplay text only. The viewer panel is an LLM role-playing viewers, not audience data, "
               "and the blind rater below is an LLM too, so treat the numbers as indicative.")
    if ev:
        checks = pd.DataFrame(ev["checks"], columns=["check", "passed", "of"])
        checks["pass rate"] = checks["passed"] / checks["of"]
        a, b = st.columns([1, 1])
        with a.container(border=True):
            st.markdown("**Rule checks, re-derived from the final timestamps**")
            st.dataframe(checks, hide_index=True, use_container_width=True,
                         column_config={"pass rate": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                   format="%.0f%%")})
            st.caption("Proves the clips are well-formed. It does not prove they are good.")
        with b.container(border=True):
            st.markdown("**Blind test: 15 tool picks vs 15 'last 40 seconds' clips**")
            st.dataframe(pd.DataFrame(ev["blind"]["groups"]), hide_index=True, use_container_width=True)
            p = ev["blind"]["paired"]
            st.markdown(f"Paired by episode: tool wins **{p['tool_wins']}**, ties **{p['ties']}**, "
                        f"baseline wins **{p['baseline_wins']}**.")
            st.caption(f"Rater: {ev['rater']}. Hidden labels, shuffled. Bottom 5 scoring above random 5 means the "
                       "ranking below the top is not validated.")

        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.markdown("**Where it picks badly** (rater's failure labels)")
            rows = [{"failure": k, "who": w.title(), "clips": n}
                    for w in ("tool", "baseline") for k, n in ev["blind"]["failures"][w].items()]
            st.altair_chart(
                alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
                    y=alt.Y("failure:N", sort="-x", title=None), x=alt.X("clips:Q", title="Clips"),
                    color=alt.Color("who:N", title=None, scale=alt.Scale(domain=["Tool", "Baseline"], range=[ACCENT, WARM]),
                                    legend=alt.Legend(orient="bottom")),
                    yOffset="who:N" if int(alt.__version__.split(".")[0]) >= 5 else alt.value(0),
                    tooltip=["failure", "who", "clips"]).properties(height=270),
                use_container_width=True)
        with c2.container(border=True):
            pn = ev["panel"]
            st.markdown("**Measured limits**")
            st.markdown(
                f"- The curve proposed **{pn['windows']}** windows; the model dropped **{pn['model_dropped_share']:.0%}**.\n"
                f"- Curve score vs final ranking: Spearman **{pn['spearman_curve_vs_panel']}**, so the curve is a "
                "finder, not a ranker.\n"
                f"- In **{pn['all_survive_3s']:.0%}** of clips all three viewers survive 3s: the panel is generous on hooks.\n"
                "- Watch to the end: " + ", ".join(f"{k} {v:.0%}" for k, v in pn["watch_to_end"].items()) + ".\n"
                f"- Panel evidence that held up: **{pn['evidence_supported']:.0%}**; "
                f"quoted a line not in the clip: **{pn['evidence_unsupported']:.0%}**.")
    else:
        rows = {}
        for c in ds["clips"]:
            for v in c["viewers"]:
                r = rows.setdefault(PERSONAS.get(v["persona"], str(v["persona"])), [0, 0])
                r[0] += 1
                r[1] += v.get("swipe_at_s") is None
        if rows:
            st.markdown("**Share of clips each simulated viewer watched to the end**")
            st.dataframe(pd.DataFrame([{"viewer": k, "watch to end": n[1] / n[0]} for k, n in rows.items()]),
                         hide_index=True, use_container_width=True,
                         column_config={"watch to end": st.column_config.ProgressColumn(
                             min_value=0, max_value=1, format="%.0f%%")})
        st.info("The blind test (tool vs baseline) is run separately: `python src/evaluate.py sample`, "
                "rate the sheet, then `python src/evaluate.py score`.")
    with st.expander("Full quality report (metrics.md)"):
        st.markdown(ds.get("metrics_md") or "No report for this run.")


def tab_downloads(ds, df):
    flat = df.drop(columns=[c for c in ("curve", "viewers") if c in df.columns])
    st.download_button("Shortlist (.csv)", flat.to_csv(index=False).encode("utf-8"), "promos.csv", "text/csv")
    if ds.get("out"):
        st.download_button("All outputs (.zip)", _zip_dir(ds["out"]), "promos_output.zip", "application/zip")
        st.caption("Contains screenplay text if you uploaded a pack: share it privately.")
    else:
        st.caption("Timestamps and scores only. The raw model outputs and script text for the real pack "
                   "are shared privately with Kuku, not published here.")


# ----------------------------------------------------------------------------- sidebar
run_btn = estimate_btn = False
engine, key, confirm_model, judge_model, episodes, uploads = "", "", "", "", "", None
with st.sidebar:
    st.header("Source")
    source = st.radio("What to look at", [SRC_REAL, SRC_SAMPLE, SRC_UPLOAD], label_visibility="collapsed")
    if source == SRC_REAL:
        st.caption("The real 50-episode run. Timestamps and scores only: the script is Kuku's and is not published.")
    else:
        if source == SRC_UPLOAD:
            uploads = st.file_uploader("A .zip of episode .txt files, or several .txt files",
                                       type=["zip", "txt"], accept_multiple_files=True)
            st.caption("Held in a temporary folder for this session only.")
        episodes = st.text_input("Only these episodes (optional, e.g. 1-10)", "")
        st.header("Engine")
        engine = st.radio("Model", ["Offline (no key, curve only)", "Gemini (use my free key)"],
                          label_visibility="collapsed")
        if engine.startswith("Gemini"):
            key = st.text_input("Gemini API key", type="password",
                                help="Free at aistudio.google.com/apikey. Used only for this run, never stored.")
            with st.expander("Swap the models"):
                confirm_model = st.text_input("Confirm model (blank = default)", "")
                judge_model = st.text_input("Panel model (blank = default)", "")
            st.caption("Free tier: about 20 requests per day per model. A 50-episode pack needs about 20 calls, "
                       "so Estimate first.")
        estimate_btn = st.button("Estimate cost (no API calls)", disabled=not engine.startswith("Gemini"))
        run_btn = st.button("Run", type="primary")

# ----------------------------------------------------------------------------- header + run
st.title("🎬 Find the Promos")
st.caption("Reads microdrama screenplays and shortlists 20-60s promo clips: hook in the first 3s, ends on a "
           "cliffhanger, makes sense to a stranger. An editor approves; it does not replace one.")

if (run_btn or estimate_btn) and engine.startswith("Gemini") and not key.strip():
    st.warning("Paste a Gemini API key, or switch to the offline engine.")
    run_btn = estimate_btn = False
if (run_btn or estimate_btn) and source == SRC_UPLOAD and not uploads:
    st.warning("Upload a pack first, or switch to the built-in sample.")
    run_btn = estimate_btn = False

if run_btn or estimate_btn:
    work = _reset_workdir()
    inp = (_save_upload(uploads, os.path.join(work, "input")) if source == SRC_UPLOAD
           else write_sample_pack(os.path.join(work, "input")))
    out_dir = os.path.join(work, "out")
    args = ["--input", inp, "--out", out_dir]
    if episodes.strip():
        args += ["--episodes", episodes.strip()]
    env_extra = {"LLM_CACHE_DIR": os.path.join(work, "cache")}
    if engine.startswith("Gemini"):
        env_extra.update({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": key.strip()})
        if confirm_model.strip():
            env_extra["LLM_MODEL"] = confirm_model.strip()
        if judge_model.strip():
            env_extra["JUDGE_MODEL"] = judge_model.strip()
    else:
        args.append("--no-llm")
    if estimate_btn:
        args.append("--dry-run")
    box = st.empty()
    with st.spinner("Estimating..." if estimate_btn else "Running the pipeline (a full real run takes minutes)..."):
        rc = _run(args, env_extra, box)
    st.session_state.out = None if (estimate_btn or rc != 0) else out_dir
    if rc != 0:
        st.error("The run failed. The log above shows why (common causes: bad key, daily quota used up).")
    elif estimate_btn:
        st.info("Estimate only: nothing was sent to the model.")

# ----------------------------------------------------------------------------- results
ds = _load_dataset(source)
if ds is None:
    st.info("Press **Run** in the sidebar (the offline engine needs no key), or switch the source to "
            "**Real run** to see the full 50-episode result. The sample is an invented 3-episode show, "
            "so its quiet episode correctly yields no promo.")
else:
    df = pd.DataFrame(ds["clips"])
    meta = ds["meta"]
    tier = str(meta.get("tier") or "?")
    where = "Real 50-episode run" if source == SRC_REAL else "This run"
    models = f" · confirm `{meta.get('model')}` · panel `{meta.get('judge_model')}`" if meta.get("model") else ""
    (st.success if tier == "full" else st.warning)(f"**{where}: {tier.upper()}.** {meta.get('tier_message') or ''}{models}")
    if df.empty:
        st.info("No promos were found in this input.")
    else:
        t1, t2, t3, t4, t5 = st.tabs(["Overview", "Ranked", "Clip detail", "Quality", "Downloads"])
        with t1:
            tab_overview(ds, df)
        with t2:
            tab_ranked(ds, df)
        with t3:
            tab_detail(ds, df)
        with t4:
            tab_quality(ds, df)
        with t5:
            tab_downloads(ds, df)

with st.expander("How it works and what it cannot do"):
    st.markdown(
        "1. **Code** builds a per-second tension curve from the delivery notes and finds candidate windows.\n"
        "2. **An LLM** confirms or trims those windows, speaking in beat indices, never timestamps.\n"
        "3. **Code** resolves times, snaps cuts so no dialogue is chopped, and verifies every quoted line.\n"
        "4. **A simulated viewer panel** (an LLM role-playing three viewers) ranks the clips.\n\n"
        "Judged from screenplay text only: acting, music and visuals are not seen. The offline engine skips "
        "steps 2 and 4 and labels every clip curve-only."
    )
