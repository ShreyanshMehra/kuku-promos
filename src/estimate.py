"""
Dry-run estimator: how many API calls and tokens will this run need, and how many of
those calls are ALREADY CACHED (free)? Zero API calls.

Why it exists: the binding constraint on the free tier is the DAILY request limit per
model (measured: 20/day on gemini-3.6-flash), not speed. A reviewer or an interview
question like "will a full run fit in one day?" should have a number before you spend
anything.

What is exact and what is estimated
  exact     : the lexicon call and every confirm call. Their prompts are built from the
              parsed episodes and the curve, using the same functions and the same cache
              key (llm.prompt_fingerprint) as a real run, so "cached" is a fact.
  estimated : the panel stage. Its prompts depend on which windows the model keeps, which
              cannot be known without calling it. We assume a survival fraction (default
              0.5, measured 70/143 on the one full run) and report a range.
  not given : a dollar figure. Prices change and model IDs churn; calls and tokens do not.

Token counts use a chars-per-token ratio, and the output ratio and per-call latencies,
all measured on the one full run; they will differ on other shows and models.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import confirm as confirm_mod
import lexicon as lexicon_mod
import llm
import panel as panel_mod
from rank import clip_lines
from tension import find_all

OUTPUT_PER_PROMPT = 0.157   # output tokens / prompt tokens, measured on the full run
SECONDS_PER_CONFIRM = 48    # measured average
SECONDS_PER_PANEL = 25      # measured on trials (smaller model)
FREE_RPD = 20               # per model per key, measured on gemini-3.6-flash free tier


# Calibrated on the one full run: 240,331 measured prompt tokens vs a chars/4 estimate of
# 157,235 (this text is token-dense: timestamps, ids, punctuation). Other shows or models
# will differ, so treat token figures as +/-20%.
CHARS_PER_TOKEN = 2.62


def _tok(*texts: str) -> int:
    return int(sum(len(t) for t in texts) / CHARS_PER_TOKEN)


def estimate(
    episodes: List[Dict[str, Any]],
    cfg: llm.LLMConfig,
    judge_model: str,
    lex_scores: Dict[str, float],
    k: Optional[int] = None,
    survive: float = 0.5,
    free_rpd: int = FREE_RPD,
    n_keys: int = 1,
) -> Dict[str, Any]:
    by_ep = {e["episode_no"]: e for e in episodes}
    windows = find_all(episodes, lex_scores, k=k) if k else find_all(episodes, lex_scores)

    # --- lexicon (exact) ---------------------------------------------------------
    words = lexicon_mod.collect_words(episodes)
    lex_user = f"RATE_WORDS: {', '.join(sorted(words))}"
    lex_cached = llm.is_cached(cfg, lexicon_mod.SYSTEM, lex_user, lexicon_mod.SCHEMA)
    lex_tokens = _tok(lexicon_mod.SYSTEM, lex_user)

    # --- confirm (exact) ---------------------------------------------------------
    batches = [windows[i:i + confirm_mod.CONFIRM_BATCH]
               for i in range(0, len(windows), confirm_mod.CONFIRM_BATCH)]
    confirm_calls = confirm_cached = confirm_tokens = 0
    for b in batches:
        user = confirm_mod._prompt(b, by_ep)
        confirm_calls += 1
        confirm_tokens += _tok(confirm_mod.SYSTEM, user)
        if llm.is_cached(cfg, confirm_mod.SYSTEM, user, confirm_mod.SCHEMA):
            confirm_cached += 1

    # --- panel: EXACT when confirm is fully cached, otherwise estimated ---------------
    # Panel prompts depend on which windows survive confirm. If every confirm call is
    # cached we can replay confirm for free (the provider is swapped for one that refuses
    # to call out, so this cannot spend anything) and build the real panel prompts.
    exact_panel = None
    if batches and confirm_cached == confirm_calls:
        orig = llm.PROVIDERS[cfg.provider]

        def _no_api(*a, **k):
            raise llm.LLMError("dry run: refusing to call the API")

        llm.PROVIDERS[cfg.provider] = _no_api
        try:
            replay = confirm_mod.confirm_windows(windows, episodes, cfg=cfg, log=lambda *_: None)
        finally:
            llm.PROVIDERS[cfg.provider] = orig
        pcfg = cfg.with_model(judge_model)
        cands = replay["candidates"]
        pb = [cands[i:i + panel_mod.PANEL_BATCH] for i in range(0, len(cands), panel_mod.PANEL_BATCH)]
        p_calls = p_cached = p_tokens = 0
        for b in pb:
            user = panel_mod._prompt(b, by_ep)
            p_calls += 1
            p_tokens += _tok(panel_mod.SYSTEM, user)
            if llm.is_cached(pcfg, panel_mod.SYSTEM, user, panel_mod.SCHEMA):
                p_cached += 1
        exact_panel = {"clips": len(cands), "calls": p_calls, "cached": p_cached, "prompt_tokens": p_tokens}

    clip_chars = 0
    for w in windows:
        cand = {"start_s": w["start_s"], "end_s": w["end_s"]}
        clip_chars += sum(len(x) for x in clip_lines(by_ep[w["episode"]], cand)) + 40

    def panel_for(frac: float) -> Dict[str, int]:
        n = int(round(len(windows) * frac))
        return {
            "clips": n,
            "calls": math.ceil(n / panel_mod.PANEL_BATCH) if n else 0,
            "prompt_tokens": int(clip_chars / CHARS_PER_TOKEN * frac)
            + int(len(panel_mod.SYSTEM) / CHARS_PER_TOKEN) * (math.ceil(n / panel_mod.PANEL_BATCH) if n else 0),
        }

    lo, mid, hi = panel_for(max(0.0, survive - 0.15)), panel_for(survive), panel_for(min(1.0, survive + 0.15))

    new_confirm = confirm_calls - confirm_cached
    new_lex = 0 if lex_cached else 1
    if exact_panel:
        new_panel = exact_panel["calls"] - exact_panel["cached"]
        panel_prompt = exact_panel["prompt_tokens"]
    else:
        new_panel = mid["calls"]
        panel_prompt = mid["prompt_tokens"]
    prompt_total = lex_tokens + confirm_tokens + panel_prompt
    new_calls: Dict[str, int] = {}
    for model, n in ((cfg.model, new_lex + new_confirm), (judge_model, new_panel)):
        new_calls[model] = new_calls.get(model, 0) + n        # sum if both stages share a model
    return {
        "episodes": len(episodes),
        "beats": sum(e["n_beats"] for e in episodes),
        "windows": len(windows),
        "lexicon": {"calls": 1, "cached": lex_cached, "model": cfg.model, "prompt_tokens": lex_tokens},
        "confirm": {"calls": confirm_calls, "cached": confirm_cached, "model": cfg.model,
                    "prompt_tokens": confirm_tokens},
        "panel": {"model": judge_model, "assumed_survival": survive, "low": lo, "mid": mid, "high": hi,
                  "exact": exact_panel},
        "new_calls": new_calls,
        "tokens": {
            "prompt": prompt_total,
            "output": int(prompt_total * OUTPUT_PER_PROMPT),
        },
        "wall_minutes": round((new_lex * 30 + new_confirm * SECONDS_PER_CONFIRM
                               + new_panel * SECONDS_PER_PANEL) / 60, 1),
        "quota": {"free_rpd": free_rpd, "keys": n_keys, "per_model_limit": free_rpd * n_keys},
        "same_model": cfg.model == judge_model,
    }


def render(e: Dict[str, Any]) -> str:
    q = e["quota"]
    lim = q["per_model_limit"]
    lx, cf, pn = e["lexicon"], e["confirm"], e["panel"]
    lines = [
        f"DRY RUN: {e['episodes']} episodes, {e['beats']} beats -> {e['windows']} candidate windows (curve, no API)",
        "",
        "API calls                          total  cached  NEW      model",
        f"  lexicon                          {lx['calls']:>5}  {int(lx['cached']):>6}  {lx['calls'] - int(lx['cached']):>3}      {lx['model']}",
        f"  confirm (exact)                  {cf['calls']:>5}  {cf['cached']:>6}  {cf['calls'] - cf['cached']:>3}      {cf['model']}",
        (f"  panel (exact: confirm replayed)  {pn['exact']['calls']:>5}  {pn['exact']['cached']:>6}  "
         f"{pn['exact']['calls'] - pn['exact']['cached']:>3}      {pn['model']}   ({pn['exact']['clips']} clips)"
         if pn.get("exact") else
         f"  panel (estimate, {pn['assumed_survival']:.0%} survive)  {pn['mid']['calls']:>5}  {'?':>6}  {pn['mid']['calls']:>3}      "
         f"{pn['model']}   (range {pn['low']['calls']}-{pn['high']['calls']}; depends on what confirm keeps)"),
        "",
    ]
    for model, n in e["new_calls"].items():
        fits = "fits" if n <= lim else "DOES NOT FIT"
        lines.append(f"  NEW calls on {model}: {n}  vs free daily limit {lim} ({q['free_rpd']}/key x {q['keys']} key(s)) -> {fits} in one day")
    if e["same_model"]:
        lines.append("  (confirm and panel share one model, so their calls add up against one limit)")
    t = e["tokens"]
    lines += [
        "",
        f"Tokens if nothing were cached: ~{t['prompt']:,} prompt + ~{t['output']:,} output (calibrated on one full run; +/-20%)",
        f"Wall time for the NEW calls only: ~{e['wall_minutes']} min (sequential; per-call latency measured on one run)",
        "Cached calls are free: re-running costs nothing for anything already answered.",
        "Not estimated: dollars (prices and model IDs change). Panel figures depend on how many windows the model keeps.",
    ]
    return "\n".join(lines)
