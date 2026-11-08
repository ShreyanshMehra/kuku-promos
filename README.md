# kuku-promos: find promo clips in microdrama screenplays

Reads timestamped screenplays and shortlists 20-60s promo clips (hook in the first
3s, ends on a cliffhanger, makes sense to a stranger) for an editor to approve.

## The idea

1. **Code finds candidates.** The scripts annotate intensity: ~99% of dialogue lines carry
   a delivery note ("Cold, Commanding"). A per-second *tension curve* built from those notes
   (plus `!`, `?`, actions, sound cues) is searched for stretches that open on a spike and
   end still tense. No API calls.
2. **The LLM only confirms and trims those windows**, speaking in *beat indices* (`ep3:41`),
   never timestamps. Code resolves indices to times, snaps cuts so no dialogue line is chopped,
   repairs duration, opens on the hook if it came late, and verifies every quoted line against
   the source. A hallucinated timestamp is structurally impossible; a hallucinated quote is rejected.
3. **A simulated viewer panel ranks them.** Three personas "watch" each clip beat by beat and
   say when they would swipe away and whether they would tap through to the next episode.
   Code turns that into a retention score and applies a recruitment factor (earlier episodes,
   no spoilers, shows the premise). It is an LLM role-playing viewers, **not audience data**.

```
parse -> lexicon (1 call) -> tension curve (code) -> confirm (LLM) -> validate/repair (code)
      -> viewer panel (LLM) -> rank + dedup (code) -> report + independent checks
```

## Run

```bash
pip install -r requirements.txt
cp .env.example .env          # add GEMINI_API_KEY (free: https://aistudio.google.com/apikey)

python run.py --input <folder | .zip | one .txt> --out out/
python run.py --input <pack> --out out/ --aliases data/aliases.json   # optional per-show name merges
python run.py --input <pack> --out out/ --dry-run                     # zero API calls: how many calls/tokens, how many already cached
python run.py --input <pack> --out out/ --no-llm                      # zero API calls, curve-only
python run.py --input <pack> --out out/ --finder llm                  # old design: LLM reads whole episodes
```

`LLM_PROVIDER=mock` runs the whole pipeline offline; its model-stage output is a placeholder,
**not a quality result**. **New input at interview time:** point `--input` at any folder, zip
or single `.txt` in the same screenplay format. Nothing is show-specific.

**Swap the model** (environment only, no code edits):

| want | set |
|---|---|
| another Gemini model | `LLM_MODEL=...` (confirm) and `JUDGE_MODEL=...` (panel) |
| OpenRouter / Groq / vLLM / any OpenAI-shaped API | `LLM_PROVIDER=openai_compatible` `LLM_MODEL=...` `OPENAI_API_KEY=...` `OPENAI_BASE_URL=...` |
| offline | `LLM_PROVIDER=mock` |

## Free-tier reality

Measured: Gemini's free tier allows **20 requests per day, per model** on the model I used.
A full 50-episode run is **~20 calls** (1 lexicon + 12 confirm + ~7 panel), roughly
270k tokens, so it fits one free day only if nothing else is spent. **Check before you spend:**
`--dry-run` builds the real prompts without sending them and reports calls, tokens and, exactly,
which calls are already cached (cached calls are free). It is exact for the lexicon and confirm
stages, and exact for the panel too when confirm is fully cached; otherwise the panel figure is a
range. Token counts matched a held-out 4-episode run within ~7% (other shows untested). The code:
caches every response (re-runs are free), fails fast on a daily-quota error instead of retrying,
and degrades to curve-only output with a loud warning rather than crashing. Extra
`GEMINI_API_KEY_2`, `_3` ... are used as failover when a model's daily quota is gone. Google's
terms discourage using multiple accounts to get around free limits; fine for finishing an
assignment, not a basis for a product.

## Outputs (`out/`)

| file | what |
|---|---|
| `promos.csv` | ranked shortlist: episode, start/end, hook line, cliffhanger line, why, archetype, scores, feed |
| `promos.json` | everything, incl. repairs, curve and panel data, run metadata |
| `rejected.json` | validator rejections AND windows the model dropped, with reasons |
| `report.html` | each clip's script text with its tension curve and the viewers' swipe points |
| `metrics.md` | programmatic checks re-derived from final timestamps, panel stats, blind-rating results |
| `lexicon.json` | the emotion-word scores (auditable, hand-editable) |
| `usage.json` | measured API calls and tokens |
| `raw/` | unedited model responses, one file per call |

## Evaluate honestly

```bash
python src/evaluate.py sample --input <pack> --out out/    # blind sheet: top5 + random5 + bottom5 vs a last-40s baseline
#   fill out/eval/ratings.csv from out/eval/blind_sample.md (labels hidden), then:
python src/evaluate.py score  --input <pack> --out out/
```

## Status: what is and is not verified

**Full-pack live run** (Gemini, `gemini-3.6-flash` confirm + `gemini-3.5-flash-lite` panel):
143 curve windows -> the model dropped 71 -> 68 promos ranked across 46 of 50 episodes. All ten
independent programmatic checks pass on 68/68 (duration, beat boundaries, no chopped dialogue,
hook within 3s, cliffhanger in last 6s, verbatim quotes). The validator rejected 2 of 72 proposals.

Measured limits, stated plainly:
- The curve **over-proposes**: the model dropped 50% of its windows, so curve-only output (`--no-llm`) is
  noticeably weaker. Kept windows scored 0.67 on the curve vs 0.57 for dropped ones.
- Among surviving clips the curve score does **not** predict the panel's ranking (Spearman ~0.08, n=68;
  range-restricted). The curve is a finder, not a ranker.
- The panel's ordering is driven mostly by one persona (the impatient scroller). The fan almost always
  watches to the end and the skeptic almost always swipes, so they add close to a constant.
- Using the model for confirm + panel does **not** save tokens versus an LLM-reads-everything design
  (measured: 23.0k vs 23.2k prompt tokens on 4 episodes). The benefits are the zero-API fallback and
  an inspectable first stage, not cost.
- Everything is judged from screenplay text: visual impact, performance, pacing and music are not assessed.

**Not yet done:** the blind human rating (the independent quality signal) has not been filled in, so
there is no human-validated quality claim yet. Multi-segment and cross-episode promos are not built
(the data model carries `segments` as a list, but only single clips are proposed).
Model IDs were checked against Google's docs on 2026-11-08 and churn quickly.

Every run ends with a **confidence tier** (full / partial / degraded / offline / mock) computed from what
actually happened to each ranked clip, shown in the console and as a banner in `report.html`.

`python -m unittest discover -s tests -v` -> 51 tests, on synthetic screenplays (no Kuku text).

## Confidentiality

The input pack is Kuku's and must not be published. `data/`, `out/`, `raw/`, `.cache/`, `*.zip`
and `*.txt` are gitignored; tests build synthetic screenplays in code; the code contains no
show-specific names (aliases are a per-show config file). `out/` holds screenplay text: share it
through a link-only folder, never a public repo.
