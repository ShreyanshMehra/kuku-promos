"""
One command, any screenplay folder / zip:

    python run.py --input <folder|zip|file.txt> --out out/

Default pipeline ("curve" finder):
    parse -> lexicon (1 LLM call per show) -> tension curve finds candidate windows
    (code, 0 calls) -> LLM confirms/trims the windows -> validate/repair (code)
    -> simulated viewer panel scores retention -> rank + dedup (code) -> report + checks

Legacy pipeline (--finder llm): the LLM reads whole episodes and proposes promos,
then a 1-5 judge scores them. Kept as an ablation baseline.

Zero-API pipeline (--no-llm): hand lexicon + curve only. Every promo is labelled
curve-only. This is also the automatic fallback if the model is unavailable.

Model is chosen by environment, never by code:
    LLM_PROVIDER = gemini | openai_compatible | mock
    LLM_MODEL    = confirm model         (default gemini-3.6-flash)
    JUDGE_MODEL  = panel / judge model   (default gemini-3.5-flash-lite)
`LLM_PROVIDER=mock` runs everything offline. Its output is placeholder, NOT model judgement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import llm  # noqa: E402
from confirm import all_curve_only, confirm_windows  # noqa: E402
from evaluate import write_metrics  # noqa: E402
from lexicon import build_lexicon  # noqa: E402
from parse import load_aliases, load_episodes, write_beats_jsonl  # noqa: E402
from rank import confidence_tier, rank_candidates  # noqa: E402
from report import write_all  # noqa: E402
from select_promos import BATCH_EPISODES, select_candidates  # noqa: E402
from tension import find_all  # noqa: E402


def _parse_episode_filter(spec: str) -> set:
    want = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            want |= set(range(int(a), int(b) + 1))
        else:
            want.add(int(part))
    return want


def main() -> int:
    ap = argparse.ArgumentParser(description="Find promo clips in microdrama screenplays.")
    ap.add_argument("--input", required=True, help="folder of .txt screenplays, a .zip, or one .txt")
    ap.add_argument("--out", default="out", help="output folder (contains screenplay text; do not publish)")
    ap.add_argument("--episodes", default=None, help="subset, e.g. 1,2,3 or 1-10")
    ap.add_argument("--finder", choices=["curve", "llm"], default="curve",
                    help="curve = tension-curve windows (default); llm = LLM reads whole episodes")
    ap.add_argument("--scorer", choices=["panel", "judge", "curve"], default=None,
                    help="panel (default for curve finder), judge (1-5, default for llm finder), curve (no model)")
    ap.add_argument("--no-llm", action="store_true", help="zero API calls: hand lexicon + curve only")
    ap.add_argument("--dry-run", action="store_true",
                    help="estimate API calls/tokens and how many are already cached, then exit (zero API calls)")
    ap.add_argument("--model", default=None, help="confirm/selector model (overrides LLM_MODEL)")
    ap.add_argument("--judge-model", default=None, help="panel/judge model (overrides JUDGE_MODEL)")
    ap.add_argument("--batch", type=int, default=BATCH_EPISODES, help="episodes per call (llm finder)")
    ap.add_argument("--k", type=int, default=None, help="curve windows kept per episode (default 3)")
    ap.add_argument("--top", type=int, default=0, help="limit CSV/HTML to the top N (0 = all)")
    ap.add_argument("--aliases", default=os.getenv("PROMO_ALIASES"),
                    help="optional per-show JSON {\"LADY X\": \"X\"} merging character tag variants")
    args = ap.parse_args()

    t0 = time.monotonic()
    out = args.out
    os.makedirs(out, exist_ok=True)

    cfg = llm.LLMConfig(raw_dir=os.path.join(out, "raw"))
    if args.model:
        cfg.model = args.model
    judge_model = args.judge_model or os.getenv("JUDGE_MODEL") or "gemini-3.5-flash-lite"
    if cfg.provider == "mock":
        judge_model = cfg.model
    no_llm = args.no_llm
    scorer = args.scorer or ("curve" if no_llm else ("panel" if args.finder == "curve" else "judge"))
    if no_llm and args.finder == "llm":
        print("--no-llm needs the curve finder", file=sys.stderr)
        return 2

    print(f"[1/6] parsing {args.input}")
    if args.aliases:
        print(f"      loaded {load_aliases(args.aliases)} character aliases from {args.aliases}")
    episodes = load_episodes(args.input)
    if args.episodes:
        want = _parse_episode_filter(args.episodes)
        episodes = [e for e in episodes if e["episode_no"] in want]
    if not episodes:
        print("no episodes to process", file=sys.stderr)
        return 2
    if args.dry_run:
        from estimate import estimate, render
        lex_path = os.path.join(out, "lexicon.json")
        if os.path.exists(lex_path):          # reuse a previous run's lexicon: windows then match exactly
            with open(lex_path, encoding="utf-8") as fh:
                scores = json.load(fh)["scores"]
            print(f"      using {lex_path} so candidate windows match the earlier run")
        else:
            scores = build_lexicon(episodes, use_llm=False, log=lambda *_: None)["scores"]
            print("      no earlier lexicon.json: using the hand lexicon, so window counts are approximate")
        print()
        print(render(estimate(episodes, cfg, judge_model, scores, k=args.k,
                              n_keys=max(1, len(llm._gemini_keys())))))
        return 0

    write_beats_jsonl(episodes, os.path.join(out, "beats.jsonl"))
    print(f"      {len(episodes)} episodes, {sum(e['n_beats'] for e in episodes)} beats")

    meta = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": os.path.basename(args.input.rstrip("/\\")),
        "provider": "none" if no_llm else cfg.provider,
        "model": None if no_llm else cfg.model,
        "judge_model": None if no_llm else judge_model,
        "finder": args.finder,
        "scorer": scorer,
        "episodes": len(episodes),
        "mock": (cfg.provider == "mock") and not no_llm,
    }

    if args.finder == "curve":
        print(f"[2/6] lexicon  ({'hand lexicon, no model' if no_llm else cfg.model})")
        lex = build_lexicon(episodes, cfg=cfg, model=None, use_llm=not no_llm)
        with open(os.path.join(out, "lexicon.json"), "w", encoding="utf-8") as fh:
            json.dump(lex, fh, indent=2, ensure_ascii=False)
        meta["lexicon_source"] = lex["source"]

        print("[3/6] tension curve -> candidate windows  (code, 0 API calls)")
        windows = find_all(episodes, lex["scores"], k=args.k) if args.k else find_all(episodes, lex["scores"])
        meta["windows_found"] = len(windows)
        print(f"      {len(windows)} windows across {len(episodes)} episodes")

        if no_llm:
            print("[4/6] confirm: skipped (--no-llm); every promo is curve-only")
            selection = all_curve_only(windows, episodes)
        else:
            print(f"[4/6] confirm: LLM sees only the windows  ({cfg.model})")
            selection = confirm_windows(windows, episodes, cfg=cfg)
        meta["dropped_by_model"] = len(selection["dropped"])
        meta["curve_only_fallbacks"] = len(selection["failures"])
    else:
        print(f"[2/6] lexicon + curve: skipped (--finder llm)")
        print(f"[3/6] Pass 1: LLM selects from whole episodes  ({cfg.model}, {args.batch} eps/call)")
        selection = select_candidates(episodes, cfg=cfg, batch_size=args.batch)
        selection.setdefault("dropped", [])
        print("[4/6] (validation happens inside selection)")

    print(f"      {len(selection['candidates'])} valid, {len(selection['rejected'])} rejected, "
          f"{len(selection['dropped'])} dropped by model, {len(selection['failures'])} fell back")

    print(f"[5/6] scoring + ranking  (scorer={scorer}"
          f"{'' if scorer == 'curve' else ', ' + judge_model})")
    ranking = rank_candidates(selection["candidates"], episodes, cfg=cfg,
                              judge_model=judge_model, scorer=scorer)
    if scorer == "panel" and selection["candidates"] and not ranking["ranked"] and not ranking["duplicates"]:
        print("      panel returned nothing usable; falling back to curve score")
        ranking = rank_candidates(selection["candidates"], episodes, cfg=cfg, scorer="curve")
        meta["scorer"] = "curve (fallback: panel unavailable)"
    print(f"      {len(ranking['ranked'])} ranked, {len(ranking['duplicates'])} duplicates folded, "
          f"{len(ranking['unjudged'])} unjudged")

    if not no_llm and cfg.provider == "gemini":
        meta["api_keys_configured"] = len(llm._gemini_keys())
        meta["key_switches"] = len(llm.KEY_SWITCHES)
    tier = confidence_tier(ranking["ranked"], mock=meta["mock"], offline=no_llm,
                           unscored=len(ranking["unjudged"]))
    meta["tier"], meta["tier_message"], meta["tier_counts"] = tier["tier"], tier["message"], tier["counts"]

    exhausted = llm.quota_exhausted()
    if exhausted:
        meta["degraded_quota"] = exhausted
        print("\n!! WARNING: API quota ran out mid-run; this output is DEGRADED:")
        for m, why in exhausted.items():
            print(f"!!   {m}: {why}")
        print("!! Windows the model could not confirm are labelled source=curve-only; clips the panel could")
        print("!! not score are listed as unjudged. Completed calls are cached: re-run the same command")
        print("!! after the quota resets and only the missing calls will be made.\n")

    print("[6/6] writing report + checks")
    paths = write_all(out, ranking, selection, episodes, meta, top=args.top)

    usage = llm.usage_summary()
    usage["pipeline_wall_seconds"] = round(time.monotonic() - t0, 1)
    with open(os.path.join(out, "usage.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "usage": usage}, fh, indent=2)

    if ranking["ranked"]:
        paths["metrics"] = write_metrics(out, episodes)

    print("\ndone.")
    for k, v in paths.items():
        print(f"  {k:9s} {v}")
    print(f"  usage     {os.path.join(out, 'usage.json')}")
    print(f"  confidence tier: {meta['tier'].upper()} -- {meta['tier_message']}")
    print(f"  api_calls={usage['api_calls']} cache_hits={usage['cache_hits']} "
          f"tokens(incl cache)={usage['prompt_tokens_incl_cache'] + usage['output_tokens_incl_cache']:,} "
          f"wall={usage['pipeline_wall_seconds']}s")
    if meta["mock"]:
        print("\nNOTE: mock provider -- model-stage output is placeholder, not model judgement.")
    if no_llm:
        print("\nNOTE: --no-llm -- every promo is curve-only (not model-confirmed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
