"""
Web front end for kuku-promos (Streamlit).

    streamlit run app.py

Nothing is hosted with the app except the code: the screenplay pack is uploaded by the
person using it, written to a temporary folder for that session, and never committed.
The built-in sample is an INVENTED show, so the app can be tried without any upload.

The pipeline itself is run.py, started as a subprocess, so the app and the command
line behave identically.
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

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from sample_data import write_sample_pack  # noqa: E402

st.set_page_config(page_title="Find the Promos", page_icon="🎬", layout="wide")

SHOW_COLS = ["rank", "feed", "episode", "start", "end", "duration_s", "archetype",
             "hook_line", "cliffhanger_line", "why_it_works", "final_score"]


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
        name = os.path.basename(f.name)
        with open(os.path.join(dest, name), "wb") as fh:
            fh.write(f.getbuffer())
    zips = [n for n in os.listdir(dest) if n.lower().endswith(".zip")]
    return os.path.join(dest, zips[0]) if len(zips) == 1 and len(os.listdir(dest)) == 1 else dest


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


st.title("Find the Promos")
st.caption("Reads microdrama screenplays and shortlists 20-60s promo clips: hook in the first 3s, "
           "ends on a cliffhanger, makes sense to a stranger. An editor approves; it does not replace one.")

with st.sidebar:
    st.header("1. Input")
    source = st.radio("Screenplays", ["Built-in sample (invented show)", "Upload my own pack"])
    uploads = None
    if source.startswith("Upload"):
        uploads = st.file_uploader("A .zip of episode .txt files, or several .txt files",
                                   type=["zip", "txt"], accept_multiple_files=True)
        st.caption("Held in a temporary folder for this session only. Do not upload anything you may not share.")
    episodes = st.text_input("Only these episodes (optional, e.g. 1-10)", "")

    st.header("2. Engine")
    engine = st.radio("Model", ["Offline (no key, curve only)", "Gemini (use my free key)"])
    key = ""
    judge_model = confirm_model = ""
    if engine.startswith("Gemini"):
        key = st.text_input("Gemini API key", type="password",
                            help="Free at aistudio.google.com/apikey. Used only for this run, never stored.")
        with st.expander("Swap the models"):
            confirm_model = st.text_input("Confirm model (blank = default)", "")
            judge_model = st.text_input("Panel model (blank = default)", "")
        st.caption("Free tier allows about 20 requests per day per model. A 50-episode pack needs about 20 calls: "
                   "use Estimate first.")
    st.header("3. Go")
    estimate_btn = st.button("Estimate cost (no API calls)", disabled=not engine.startswith("Gemini"))
    run_btn = st.button("Run", type="primary")

needs_key = engine.startswith("Gemini") and not key.strip()
if engine.startswith("Gemini") and needs_key and (run_btn or estimate_btn):
    st.warning("Paste a Gemini API key, or switch to the offline engine.")
    run_btn = estimate_btn = False
if source.startswith("Upload") and not uploads and (run_btn or estimate_btn):
    st.warning("Upload a pack first, or switch to the built-in sample.")
    run_btn = estimate_btn = False

if run_btn or estimate_btn:
    work = _reset_workdir()
    if source.startswith("Upload"):
        inp = _save_upload(uploads, os.path.join(work, "input"))
    else:
        inp = write_sample_pack(os.path.join(work, "input"))
    out = os.path.join(work, "out")
    args = ["--input", inp, "--out", out]
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
    st.session_state.ran = None if estimate_btn else (rc == 0)
    st.session_state.out = None if estimate_btn else out
    if rc != 0:
        st.error("The run failed. The log above shows why (common causes: bad key, daily quota used up).")
    elif estimate_btn:
        st.info("Estimate only: nothing was sent to the model.")

out = st.session_state.get("out")
if out and os.path.exists(os.path.join(out, "promos.csv")):
    meta = {}
    try:
        with open(os.path.join(out, "promos.json"), encoding="utf-8") as fh:
            meta = json.load(fh).get("meta", {})
    except (OSError, ValueError):
        pass
    tier = meta.get("tier", "?")
    msg = f"**Confidence tier: {str(tier).upper()}.** {meta.get('tier_message', '')}"
    (st.success if tier == "full" else st.warning)(msg)

    df = pd.read_csv(os.path.join(out, "promos.csv"), encoding="utf-8-sig")
    st.subheader(f"{len(df)} promos, best first")
    st.dataframe(df[[c for c in SHOW_COLS if c in df.columns]], use_container_width=True, hide_index=True)

    tab_report, tab_metrics, tab_raw = st.tabs(["Clip report", "Quality checks", "Downloads"])
    with tab_report:
        st.caption("Each clip's script text, its tension curve and where the simulated viewers swipe.")
        with open(os.path.join(out, "report.html"), encoding="utf-8") as fh:
            components.html(fh.read(), height=760, scrolling=True)
    with tab_metrics:
        mpath = os.path.join(out, "metrics.md")
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as fh:
                st.markdown(fh.read())
    with tab_raw:
        st.download_button("Download all outputs (.zip)", _zip_dir(out), "promos_output.zip", "application/zip")
        st.caption("Contains screenplay text if you uploaded a pack: share it privately.")
elif not (run_btn or estimate_btn):
    st.info("Pick the built-in sample and press Run (offline, no key needed) to see the output. "
            "The sample is an invented 3-episode show, so a quiet episode correctly yields no promo.")

with st.expander("How it works and what it cannot do"):
    st.markdown(
        "1. **Code** builds a per-second tension curve from the delivery notes and finds candidate windows.\n"
        "2. **An LLM** confirms or trims those windows, speaking in beat indices, never timestamps.\n"
        "3. **Code** resolves times, snaps cuts so no dialogue is chopped, and verifies every quoted line.\n"
        "4. **A simulated viewer panel** (an LLM role-playing three viewers) ranks the clips. It is not audience data.\n\n"
        "Judged from screenplay text only: acting, music and visuals are not seen. "
        "The offline engine skips steps 2 and 4 and labels every clip curve-only."
    )
