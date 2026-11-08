# Quality report

## 1. Programmatic checks (re-derived from final timestamps, 100% of ranked promos)

- ranked promos: **68** across **46** episodes
- proposals rejected at selection time: **2** (of 72 model proposals)
- batches that failed outright: **0**
- candidates the scorer could not score (left out of the ranking): **0**
- near-duplicates folded into a better clip: **2**
- ranked promos that needed a boundary repair: **5**

| check | passed | rate |
|---|---|---|
| duration_in_range | 68/68 | 100% |
| within_episode | 68/68 | 100% |
| start_on_beat_boundary | 68/68 | 100% |
| end_on_beat_boundary | 68/68 | 100% |
| no_dialogue_chopped | 68/68 | 100% |
| hook_within_3s | 68/68 | 100% |
| cliffhanger_in_last_6s | 68/68 | 100% |
| hook_quote_verbatim | 68/68 | 100% |
| cliffhanger_quote_verbatim | 68/68 | 100% |
| segments_ordered_disjoint | 68/68 | 100% |

**Why the validator rejected proposals** (the model's raw misses):

- ungrounded quote: 1
- hook not within 3s: 1

**Final-output failures:** 0

### Where the candidates came from, and what the model added

- finder `curve`, scorer `panel`, lexicon `llm`, sources of ranked promos: `{"llm-confirmed": 68}`
- the curve proposed **143** windows; the model dropped **71** (50%) and trimmed **42** of the ranked ones
- mean curve score: kept 0.673 vs dropped 0.572 (if close, the model is rejecting windows the curve cannot tell apart)
- **Spearman correlation, curve score vs final scorer: 0.058** (n=68). High = the free curve alone already orders clips much like the model; low = the model is adding independent signal. Neither is 'good'; it tells you what the API spend buys.

### Viewer-panel discrimination (does it separate clips, or rate everything alike?)

- retention mean 0.623, sd 0.175 over 68 clips (sd near 0 = the panel is not discriminating)
- mean hook survival 0.941, watch-through 0.602, tap-next 0.357
- share of clips where all viewers survive 3s: 85% (very high = the simulated viewers are too generous)

| persona | tap-next mean | tap-next sd | watches to end |
|---|---|---|---|
| 1 | 0.72 | 0.27 | 88% |
| 2 | 0.17 | 0.31 | 23% |
| 3 | 0.19 | 0.3 | 28% |

A persona with sd near 0, or that watches to the end every time, adds a constant, not signal. Treat the panel's ordering as driven by the personas that move.

### Is the panel's reasoning grounded in the clip? (deterministic; flags only, never changes a score)

- 207 of 207 simulated viewers cited a quote. **94% held up** (the quote is a real clip line and, for swipers, on screen near the swipe); **2% cited a line that is not in the clip**.
- breakdown: `{"supported": 195, "quote_ok_timing_off": 7, "unsupported": 5}`
- clips flagged for human review: **1** (**0** of the top 10)
- viewers whose evidence failed, by persona: `{"1": "2/69", "2": "7/69", "3": "3/69"}`
- Reasons that fail this check are role-play, not reading: treat that persona's verdicts with extra suspicion.

**Selector vs judge disagreement** on flags: `{"spoils_arc": 0, "carries_premise": 0}`
**Share of top-10 from early episodes (<=15):** 80%

## 2. Blind human rating (small N -- indicative, not statistical)

Rated clips: **30** (labels hidden at rating time, shuffled).

| group | n | hook | cliffhanger | standalone | total /6 |
|---|---|---|---|---|---|
| tool (all) | 15 | 1.27 | 1.6 | 1.53 | 4.4 |
| baseline: last 40s | 15 | 0.8 | 1.13 | 1.47 | 3.4 |
| tool / top | 5 | 1.2 | 2.0 | 2.0 | 5.2 |
| tool / random | 5 | 1.4 | 1.2 | 1.2 | 3.8 |
| tool / bottom | 5 | 1.2 | 1.6 | 1.4 | 4.2 |

**Paired by episode, tool vs last-40s baseline:** tool wins 7, ties 5, baseline wins 2 (n=14). In 0 of these the tool picked the baseline window itself (counted as a tie).

**Failure taxonomy** (what went wrong, by who picked it):

| failure type | tool | baseline |
|---|---|---|
| none | 5 | 4 |
| needs_backstory | 2 | 1 |
| weak_hook | 4 | 6 |
| cut_mid_reaction | 3 | 1 |
| resolved_ending | 1 | 3 |

## 3. Not judgeable from screenplay text

Visual impact of the rendered footage, actor/voice performance, pacing, music and sound, and whether the AI-generated video at a given timestamp is clean. A line that reads shocking can land flat on screen. Treat every ranking as a shortlist for an editor, not a verdict.
