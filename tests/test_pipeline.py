"""
Tests on SYNTHETIC screenplays built in code (same format as the real pack, but
invented content). No Kuku text is ever stored in this repo.

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import llm  # noqa: E402
import parse as parse_module  # noqa: E402
from evaluate import check_promo  # noqa: E402
from parse import load_episodes, parse_episode, render_compact  # noqa: E402
from rank import mark_duplicates, recruitment_factor  # noqa: E402
from select_promos import (  # noqa: E402
    quote_is_grounded, snap_window, validate_candidate,
)

# (speaker, start, end, text)
DIALOGUE = [
    ("ALICE", 0.5, 2.5, "You lied to me."),
    ("BOB", 3.0, 5.0, "I had no choice."),
    ("ALICE", 6.0, 8.0, "Who is she?"),
    ("BOB", 9.0, 11.0, "She is my wife."),
    ("CARA", 12.0, 14.0, "Surprise, darling."),
    ("ALICE", 15.0, 17.0, "Get out of my house."),
    ("BOB", 18.0, 20.0, "Alice, wait."),
    ("CARA", 22.0, 24.0, "She will never forgive you."),
    ("ALICE", 26.0, 28.0, "Watch me."),
    ("BOB", 29.0, 33.0, "There is something you do not know about your father."),
    ("ALICE", 36.0, 38.0, "What about my father?"),
    ("BOB", 40.0, 42.0, "He is not dead."),
    ("CARA", 45.0, 47.0, "Tell her everything."),
    ("ALICE", 50.0, 52.0, "Where is he?"),
    ("BOB", 55.0, 57.0, "Behind you."),
    ("ALICE", 60.0, 62.0, "No..."),
    ("CARA", 64.0, 66.0, "Too late."),
]
CAMERA = [
    (0.0, 3.0, "Wide shot of the foyer."),
    (28.0, 31.0, "Close-up on Bob's face."),
    (66.0, 68.0, "Cut to black."),
]


def ts(x: float) -> str:
    return f"{int(x // 60):02d}:{x - 60 * int(x // 60):05.2f}"


def make_screenplay(ep_no: int, upper_labels: bool = False, shuffle: bool = False) -> str:
    beats = [("CAMERA", a, b, t) for a, b, t in CAMERA]
    beats += [(s.replace("ALICE", "LADY ALICE") if i % 5 == 0 else s, a, b, f'"{t}"')
              for i, (s, a, b, t) in enumerate(DIALOGUE)]
    if shuffle:
        beats = beats[::-1]
    heading, sstart, send, adl = (
        ("SCENE HEADING", "SCENE START", "SCENE END", "ACTIONS AND DIALOGUE")
        if upper_labels else
        ("Scene heading", "Scene start", "Scene end", "Actions and dialogue")
    )
    out = [
        f"EPISODE NUMBER: Episode {ep_no}",
        "EPISODE LOGLINE: A synthetic test episode.",
        "CONTINUITY NOTES: Synthetic.",
        "",
        "--- SCRIPT START ---",
        "",
        "SCENE 01",
        f"{heading}: INT. FOYER - DAY",
        f"{sstart}: 00:00.00",
        f"{send}: 01:08.00",
        "",
        f"{adl}:",
        "",
    ]
    for tag, a, b, text in beats:
        out += [f"**{tag}:**", f"Start: ({ts(a)}) | End: ({ts(b)})", text, ""]
    out.append("--- SCRIPT END ---")
    return "\n".join(out)


def find(ep: dict, contains: str) -> int:
    return next(b["idx"] for b in ep["beats"] if contains in b["text"])


def ref(ep: dict, contains: str) -> str:
    return f"ep{ep['episode_no']}:{find(ep, contains)}"


def cand(ep, start, end, hook, cliff, hq=None, cq=None):
    return {
        "episode": ep["episode_no"],
        "start_idx": ref(ep, start), "end_idx": ref(ep, end),
        "hook_idx": ref(ep, hook), "cliffhanger_idx": ref(ep, cliff),
        "hook_quote": hq if hq is not None else hook,
        "cliffhanger_quote": cq if cq is not None else cliff,
        "why": "synthetic", "standalone_notes": "none",
        "spoils_arc": False, "carries_premise": True,
    }


class ParseTests(unittest.TestCase):
    def test_title_and_upper_case_labels_parse_identically(self):
        a = parse_episode(make_screenplay(1))
        b = parse_episode(make_screenplay(1, upper_labels=True))
        self.assertEqual(a["n_beats"], b["n_beats"])
        self.assertEqual([x["start_s"] for x in a["beats"]], [x["start_s"] for x in b["beats"]])

    def test_out_of_order_file_is_sorted_and_indexed_after_sorting(self):
        ep = parse_episode(make_screenplay(1, shuffle=True))
        starts = [b["start_s"] for b in ep["beats"]]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual([b["idx"] for b in ep["beats"]], list(range(ep["n_beats"])))

    def tearDown(self):
        parse_module.set_aliases({})

    def test_no_aliases_by_default_so_no_name_is_ever_guessed(self):
        # Auto-merging "LADY X" into "X" would wrongly collapse LORD/LADY X.
        parse_module.set_aliases({})
        ep = parse_episode(make_screenplay(1))
        speakers = {b["speaker"] for b in ep["beats"] if b["is_dialogue"]}
        self.assertIn("LADY ALICE", speakers)
        self.assertIn("ALICE", speakers)

    def test_configured_alias_merges_and_keeps_flags_and_emotion(self):
        text = make_screenplay(1).replace("**LADY ALICE:**", "**LADY ALICE (V.O.) (Tender):**")
        parse_module.set_aliases({"lady alice": "alice"})   # case-insensitive config
        ep = parse_episode(text)
        speakers = {b["speaker"] for b in ep["beats"] if b["is_dialogue"]}
        self.assertNotIn("LADY ALICE", speakers)
        merged = [b for b in ep["beats"] if b["speaker_raw"] == "LADY ALICE"]
        self.assertTrue(merged)
        self.assertTrue(all(b["speaker"] == "ALICE" for b in merged))
        self.assertTrue(all("V.O." in b["flags"] and b["emotion"] == "Tender" for b in merged))

    def test_alias_file_loads_from_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "aliases.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"LADY ALICE": "ALICE"}, fh)
            self.assertEqual(parse_module.load_aliases(path), 1)
            ep = parse_episode(make_screenplay(1))
            self.assertNotIn("LADY ALICE", {b["speaker"] for b in ep["beats"] if b["is_dialogue"]})

    def test_production_beats_have_no_speaker(self):
        ep = parse_episode(make_screenplay(1))
        self.assertTrue(all(b["speaker"] is None for b in ep["beats"] if not b["is_dialogue"]))
        self.assertTrue(any(b["type"] == "CAMERA" for b in ep["beats"]))

    def test_compact_render_namespaces_indices(self):
        ep = parse_episode(make_screenplay(7))
        self.assertIn("[ep7:0]", render_compact(ep))


class GroundingTests(unittest.TestCase):
    def test_verbatim_passes_paraphrase_and_invention_fail(self):
        text = "There is something you do not know about your father."
        self.assertTrue(quote_is_grounded("something you do not know", text))
        self.assertTrue(quote_is_grounded('"something you do not know..."', text))
        self.assertFalse(quote_is_grounded("something you never knew", text))
        self.assertFalse(quote_is_grounded("a line that does not exist", text))
        self.assertFalse(quote_is_grounded("", text))


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.ep = parse_episode(make_screenplay(1))

    def test_snap_extends_end_past_a_line_still_being_spoken(self):
        s, e = snap_window(self.ep["beats"], 0.5, 31.0)  # 29.0-33.0 line straddles 31.0
        self.assertEqual(e, 33.0)

    def test_snap_pulls_start_back_to_start_of_straddling_line(self):
        s, e = snap_window(self.ep["beats"], 31.0, 50.0)
        self.assertEqual(s, 29.0)


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.ep = parse_episode(make_screenplay(1))

    def v(self, raw):
        return validate_candidate(raw, self.ep, 1)

    def test_valid_candidate_is_ok_and_survives_independent_checks(self):
        c = self.v(cand(self.ep, "You lied", "Watch me", "You lied", "Watch me"))
        self.assertEqual(c["status"], "ok", c["reasons"])
        self.assertTrue(all(check_promo(c, self.ep).values()))

    def test_late_hook_is_repaired_by_opening_on_the_hook_when_clip_stays_long_enough(self):
        # hook ("Who is she?") is 5.5s in; opening on it leaves a 22s clip, so it is kept
        c = self.v(cand(self.ep, "You lied", "Watch me", "Who is she", "Watch me"))
        self.assertEqual(c["status"], "ok", c["reasons"])
        self.assertEqual(c["start_s"], 6.0)
        self.assertTrue(c["hook_within_3s"])
        self.assertTrue(any("opened on the hook" in r for r in c["repairs"]))
        self.assertTrue(all(check_promo(c, self.ep).values()))

    def test_late_hook_is_still_rejected_when_opening_on_it_would_make_the_clip_too_short(self):
        # hook ("Get out of my house.") is 14.5s in; opening on it leaves 13s < 20s -> reject, never pad
        c = self.v(cand(self.ep, "You lied", "Watch me", "Get out of my house", "Watch me"))
        self.assertEqual(c["status"], "rejected")
        self.assertTrue(any("hook beat starts" in r for r in c["reasons"]))

    def test_too_long_is_repaired_by_reopening_on_the_hook(self):
        c = self.v(cand(self.ep, "You lied", "Too late", "Who is she", "Too late"))
        self.assertEqual(c["status"], "ok", c["reasons"])
        self.assertLessEqual(c["duration_s"], 60.0)
        self.assertTrue(any("reopened on hook" in r for r in c["repairs"]))

    def test_too_long_and_unrepairable_is_rejected_not_silently_kept(self):
        c = self.v(cand(self.ep, "You lied", "Too late", "You lied", "Too late"))
        self.assertEqual(c["status"], "rejected")
        self.assertTrue(any("duration" in r for r in c["reasons"]))

    def test_hook_is_rechecked_after_repair_extends_the_start(self):
        # 11s clip -> repair pulls the start back -> the hook is no longer in the first 3s
        c = self.v(cand(self.ep, "Behind you", "Too late", "Behind you", "Too late"))
        self.assertEqual(c["status"], "rejected")
        self.assertTrue(any("hook beat starts" in r for r in c["reasons"]), c["reasons"])

    def test_cut_never_chops_dialogue(self):
        c = self.v(cand(self.ep, "You lied", "Close-up on Bob", "You lied", "There is something"))
        self.assertEqual(c["status"], "ok", c["reasons"])
        self.assertEqual(c["end_s"], 33.0)
        self.assertEqual(c["dialogue_straddles"], [])
        self.assertTrue(any("snapped" in r for r in c["repairs"]))

    def test_cross_episode_index_is_rejected(self):
        raw = cand(self.ep, "You lied", "Watch me", "You lied", "Watch me")
        raw["start_idx"] = "ep9:3"
        c = self.v(raw)
        self.assertEqual(c["status"], "rejected")
        self.assertTrue(any("points at episode 9" in r for r in c["reasons"]))

    def test_out_of_range_and_malformed_indices_are_rejected(self):
        raw = cand(self.ep, "You lied", "Watch me", "You lied", "Watch me")
        raw["end_idx"] = "ep1:99999"
        self.assertEqual(self.v(raw)["status"], "rejected")
        raw["end_idx"] = "line 42"
        self.assertEqual(self.v(raw)["status"], "rejected")

    def test_fabricated_quote_is_rejected_as_ungrounded(self):
        c = self.v(cand(self.ep, "You lied", "Watch me", "You lied", "Watch me",
                        hq="I never loved you"))
        self.assertEqual(c["status"], "rejected")
        self.assertTrue(any("ungrounded" in r for r in c["reasons"]))

    def test_validator_never_raises_on_garbage(self):
        for raw in ({}, {"start_idx": None}, {"start_idx": 5, "end_idx": [], "hook_idx": {}}):
            self.assertEqual(self.v(raw)["status"], "rejected")


class RankTests(unittest.TestCase):
    def test_dedup_ignores_character_names(self):
        names = {"alice", "bob"}
        mk = lambda i, ep, hq, s: {  # noqa: E731
            "id": i, "episode": ep, "start_s": s, "end_s": s + 30,
            "hook_quote": hq, "cliffhanger_quote": "", "why": "",
        }
        a = mk("a", 1, "Alice you betrayed the crown", 0)
        b = mk("b", 5, "Bob you betrayed the crown", 0)       # same event, different name
        c = mk("c", 6, "Alice you are the rightful heir", 0)  # same lead, different event
        mark_duplicates([a, b, c], names)
        self.assertEqual(b["duplicate_of"], "a")
        self.assertIsNone(c["duplicate_of"])

    def test_same_episode_overlap_is_duplicate_but_disjoint_is_not(self):
        base = {"episode": 2, "hook_quote": "x", "cliffhanger_quote": "", "why": ""}
        a = {**base, "id": "a", "start_s": 0, "end_s": 30}
        b = {**base, "id": "b", "start_s": 5, "end_s": 35}
        c = {**base, "id": "c", "start_s": 60, "end_s": 90}
        mark_duplicates([a, b, c], set())
        self.assertEqual(b["duplicate_of"], "a")
        self.assertIsNone(c["duplicate_of"])

    def test_recruitment_prefers_early_unspoiled_premise_clips(self):
        self.assertGreater(recruitment_factor(1, False, False), recruitment_factor(40, False, False))
        self.assertLess(recruitment_factor(5, True, False), recruitment_factor(5, False, False))
        self.assertGreater(recruitment_factor(5, False, True), recruitment_factor(5, False, False))


class LLMLayerTests(unittest.TestCase):
    def test_extract_json_tolerates_fences_and_prose(self):
        self.assertEqual(llm.extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(llm.extract_json('Sure! Here you go: {"a": [1, 2]} hope that helps'), {"a": [1, 2]})
        with self.assertRaises(llm.LLMError):
            llm.extract_json("no json here")

    def test_unknown_provider_fails_loudly(self):
        with self.assertRaises(llm.LLMError):
            llm.call_llm("s", "u", cfg=llm.LLMConfig(provider="nope"), use_cache=False)

    def test_provider_swap_is_configuration_only(self):
        self.assertEqual(set(llm.PROVIDERS), {"gemini", "openai_compatible", "mock"})


class EndToEndMockTest(unittest.TestCase):
    def test_new_input_folder_runs_end_to_end_with_no_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "newshow")
            os.makedirs(src)
            for n in (1, 2, 3):
                with open(os.path.join(src, f"episode-00{n}-Episode {n}.txt"), "w", encoding="utf-8") as fh:
                    fh.write(make_screenplay(n, upper_labels=(n == 2), shuffle=(n == 3)))
            self.assertEqual(len(load_episodes(src)), 3)
            out = os.path.join(tmp, "out")
            env = {**os.environ, "LLM_PROVIDER": "mock", "LLM_CACHE_DIR": os.path.join(tmp, "cache"),
                   "PYTHONIOENCODING": "utf-8"}
            env.pop("GEMINI_API_KEY", None)
            r = subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), "--input", src, "--out", out],
                               capture_output=True, text=True, env=env, encoding="utf-8")
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            for name in ("promos.json", "promos.csv", "rejected.json", "report.html", "usage.json", "beats.jsonl"):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)
            with open(os.path.join(out, "promos.json"), encoding="utf-8") as fh:
                data = json.load(fh)
            with open(os.path.join(out, "rejected.json"), encoding="utf-8") as fh:
                rej = json.load(fh)
            accounted = (len(data["ranked"]) + len(data["duplicates"]) + len(data["unjudged"])
                         + len(rej["rejected"]) + len(rej["dropped_by_model"]))
            self.assertGreater(data["meta"]["windows_found"], 0)
            self.assertEqual(accounted, data["meta"]["windows_found"],
                             "every curve window must be accounted for, none silently lost")
            self.assertTrue(data["meta"]["mock"])


def tagged_episode(ep_no: int = 1) -> dict:
    """70s synthetic episode: first half calm, second half screaming, ends on a question."""
    beats = []
    for i, t in enumerate(range(1, 67, 3)):
        calm = t < 31
        last = t >= 64
        text = "Is this the end?" if last else ("We should talk." if calm else "Get out now!")
        beats.append((f"ALICE ({'Calm, Gentle' if calm else 'Screaming, Furious'})", float(t), t + 2.0, f'"{text}"'))
    lines = [f"EPISODE NUMBER: Episode {ep_no}", "EPISODE LOGLINE: x", "CONTINUITY NOTES: x", "",
             "--- SCRIPT START ---", "", "SCENE 01", "Scene heading: INT. HALL - DAY",
             "Scene start: 00:00.00", "Scene end: 01:10.00", "", "Actions and dialogue:", ""]
    for tag, a, b, text in beats:
        lines += [f"**{tag}:**", f"Start: ({ts(a)}) | End: ({ts(b)})", text, ""]
    lines.append("--- SCRIPT END ---")
    return parse_episode("\n".join(lines))


class TensionTests(unittest.TestCase):
    def setUp(self):
        from lexicon import FALLBACK
        self.ep = tagged_episode()
        self.lex = dict(FALLBACK)

    def test_screaming_question_ending_window_beats_calm_window(self):
        from tension import candidate_windows
        wins = candidate_windows(self.ep, self.lex, k=3)
        self.assertTrue(wins)
        best = wins[0]
        self.assertGreaterEqual(best["start_s"], 30.0, "best window should open in the screaming half")
        self.assertGreaterEqual(best["curve"]["hook"], 0.7)

    def test_windows_are_legal_and_do_not_overlap_much(self):
        from rank import _iou
        from select_promos import MAX_S, MIN_S
        from tension import candidate_windows
        wins = candidate_windows(self.ep, self.lex, k=3)
        for w in wins:
            self.assertTrue(MIN_S - 1e-6 <= w["duration_s"] <= MAX_S + 1e-6)
        for i in range(len(wins)):
            for j in range(i + 1, len(wins)):
                self.assertLess(_iou(wins[i], wins[j]), 0.30)

    def test_curve_only_candidates_pass_validator_and_independent_checks(self):
        from confirm import curve_only_candidate
        from tension import candidate_windows
        for w in candidate_windows(self.ep, self.lex, k=3):
            c = curve_only_candidate(w, self.ep)
            self.assertEqual(c["status"], "ok", c["reasons"])
            self.assertEqual(c["source"], "curve-only")
            self.assertTrue(all(check_promo(c, self.ep).values()))

    def test_untagged_lines_are_neutral_not_zero(self):
        from lexicon import emotion_score
        self.assertEqual(emotion_score(None, {}), 1.0)
        self.assertEqual(emotion_score("Screaming", {"screaming": 3.0}), 3.0)
        self.assertEqual(emotion_score("Calm, Screaming", {"calm": 0.0, "screaming": 3.0}), 3.0)


class PanelMathTests(unittest.TestCase):
    def test_none_swipe_means_watched_to_end_and_values_are_derived(self):
        from panel import score_panel
        s = score_panel(40.0, [
            {"persona": 1, "swipe_at_s": None, "tap_next": 0.9},
            {"persona": 2, "swipe_at_s": 2.0, "tap_next": 0.1},
            {"persona": 3, "swipe_at_s": 20.0, "tap_next": 0.5},
        ])
        self.assertAlmostEqual(s["hook_survival"], 2 / 3, places=2)     # swipe at 2s fails the 3s test
        self.assertAlmostEqual(s["watch_through"], (1.0 + 0.05 + 0.5) / 3, places=2)
        self.assertAlmostEqual(s["tap_next"], 0.5, places=2)
        self.assertAlmostEqual(s["swipe_spread"], 0.95, places=2)
        self.assertNotIn("disagreement", s)   # personas differ by design; a flag carried no information

    def test_out_of_range_values_are_clamped_or_discarded(self):
        from panel import score_panel
        s = score_panel(30.0, [
            {"swipe_at_s": 99, "tap_next": 1.7},          # past the end == watched all; tap clamped to 1
            {"swipe_at_s": "null", "tap_next": -3},       # string null == watched all; tap clamped to 0
            {"swipe_at_s": -5, "tap_next": 0.5},          # negative swipe: unusable
            {"swipe_at_s": "soon", "tap_next": 0.5},      # junk: unusable
            "not a dict",
        ])
        self.assertEqual(s["n_viewers"], 2)
        self.assertEqual(s["watch_through"], 1.0)
        self.assertEqual(s["tap_next"], 0.5)

    def test_early_swiper_without_tap_next_counts_as_zero_not_discarded(self):
        # Regression: the model omits tap_next for viewers who swipe away. Discarding
        # them dropped the worst clips from the ranking (survivorship bias).
        from panel import score_panel
        s = score_panel(40.0, [
            {"persona": 1, "swipe_at_s": None, "tap_next": 0.9},
            {"persona": 2, "swipe_at_s": 10.2},
            {"persona": 3, "swipe_at_s": 6.9},
        ])
        self.assertIsNotNone(s)
        self.assertEqual(s["n_viewers"], 3)
        self.assertAlmostEqual(s["tap_next"], 0.3, places=2)         # (0.9 + 0 + 0) / 3
        # but a viewer who watched to the end and gave no tap_next is incomplete
        self.assertIsNone(score_panel(40.0, [
            {"swipe_at_s": None}, {"swipe_at_s": None, "tap_next": 0.5},
        ]))

    def test_too_few_usable_viewers_means_unscored_not_invented(self):
        from panel import score_panel
        self.assertIsNone(score_panel(30.0, [{"swipe_at_s": 5, "tap_next": 0.4}]))
        self.assertIsNone(score_panel(30.0, []))

    def test_spearman(self):
        from evaluate import spearman
        self.assertEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))        # no variance -> undefined, not 0
        self.assertIsNone(spearman([1, 2], [1, 2]))               # too few points


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        def boom(system, user, schema, cfg):
            raise llm.LLMError("simulated outage")

        llm.PROVIDERS["failing"] = boom
        self.addCleanup(llm.PROVIDERS.pop, "failing", None)
        self.cfg = llm.LLMConfig(provider="failing", cache_dir=os.path.join(self.tmp.name, "c"),
                                 raw_dir=os.path.join(self.tmp.name, "r"))
        self.ep = tagged_episode()

    def test_lexicon_falls_back_to_hand_lexicon_when_model_is_down(self):
        from lexicon import build_lexicon
        lex = build_lexicon([self.ep], cfg=self.cfg, log=lambda *_: None)
        self.assertEqual(lex["source"], "fallback")
        self.assertEqual(lex["scores"]["screaming"], 3.0)

    def test_confirm_falls_back_to_curve_only_when_model_is_down(self):
        from confirm import confirm_windows
        from lexicon import FALLBACK
        from tension import candidate_windows
        wins = candidate_windows(self.ep, dict(FALLBACK), k=3)
        res = confirm_windows(wins, [self.ep], cfg=self.cfg, log=lambda *_: None)
        self.assertEqual(len(res["failures"]), len(wins))
        self.assertTrue(res["candidates"])
        self.assertTrue(all(c["source"] == "curve-only" for c in res["candidates"]))

    def test_curve_scorer_ranks_without_any_model(self):
        from confirm import all_curve_only
        from lexicon import FALLBACK
        from rank import rank_candidates
        from tension import candidate_windows
        wins = candidate_windows(self.ep, dict(FALLBACK), k=3)
        sel = all_curve_only(wins, [self.ep])
        out = rank_candidates(sel["candidates"], [self.ep], cfg=self.cfg, scorer="curve")
        self.assertTrue(out["ranked"])
        self.assertTrue(all(c["scorer"] == "curve" for c in out["ranked"]))
        self.assertEqual([c["rank"] for c in out["ranked"]], list(range(1, len(out["ranked"]) + 1)))


class QuotaTests(unittest.TestCase):
    ONE_KEY = {"GEMINI_API_KEY": "key-one", "GEMINI_API_KEY_2": "", "GOOGLE_API_KEY": ""}

    def setUp(self):
        for s in (llm.EXHAUSTED, llm.KEY_EXHAUSTED, llm.KEY_SWITCHES):
            s.clear()
            self.addCleanup(s.clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = llm.LLMConfig(provider="gemini", model="fake-model", rpm=600,
                                 cache_dir=os.path.join(self.tmp.name, "c"),
                                 raw_dir=os.path.join(self.tmp.name, "r"))

    def test_daily_quota_429_fails_fast_with_one_http_attempt_and_is_remembered(self):
        from unittest import mock

        class Resp:
            status_code = 429
            headers = {}
            text = ('{"error":{"message":"limit: 20, model: fake-model","details":[{"quotaId":'
                    '"GenerateRequestsPerDayPerProjectPerModel-FreeTier"},{"retryDelay":"4771s"}]}}'
                    ' "retryDelay": "4771s"')

        with mock.patch.object(llm.requests, "post", return_value=Resp()) as post, \
                mock.patch.dict(os.environ, self.ONE_KEY):
            with self.assertRaises(llm.QuotaExhausted):
                llm.call_llm("s", "u", cfg=self.cfg, use_cache=False)
            self.assertEqual(post.call_count, 1, "a daily quota must not be retried")
            self.assertIn("fake-model", llm.quota_exhausted())
            # every later call to that model now fails instantly, with NO http request
            with self.assertRaises(llm.QuotaExhausted):
                llm.call_llm("s2", "u2", cfg=self.cfg, use_cache=False)
            self.assertEqual(post.call_count, 1)

    def _two_key_post(self):
        """Fake endpoint: key-one is out of daily quota, key-two works."""
        calls = []

        class R:
            def __init__(self, ok):
                self.status_code = 200 if ok else 429
                self.headers = {}
                self.text = "" if ok else ('limit: 20 GenerateRequestsPerDayPerProjectPerModel '
                                           '"retryDelay": "4771s"')

            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]},
                                        "finishReason": "STOP"}],
                        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}}

        def post(url, **kw):
            calls.append(url)
            return R(ok="key=key-two" in url)

        return post, calls

    def test_second_key_takes_over_when_first_keys_daily_quota_is_gone(self):
        from unittest import mock
        post, calls = self._two_key_post()
        env = {"GEMINI_API_KEY": "key-one", "GEMINI_API_KEY_2": "key-two", "GOOGLE_API_KEY": ""}
        with mock.patch.object(llm.requests, "post", side_effect=post), mock.patch.dict(os.environ, env):
            self.assertEqual(llm.call_llm("s", "u", cfg=self.cfg, use_cache=False), {"ok": True})
            self.assertEqual(len(calls), 2)                      # key one tried once, then key two
            # the exhausted key is remembered: the next call goes straight to key two
            self.assertEqual(llm.call_llm("s2", "u2", cfg=self.cfg, use_cache=False), {"ok": True})
            self.assertEqual(len(calls), 3)
            self.assertIn("key=key-two", calls[-1])
            self.assertEqual(llm.quota_exhausted(), {}, "model is NOT exhausted while a key remains")
            self.assertTrue(llm.KEY_SWITCHES)

    def test_all_keys_exhausted_raises_and_marks_model(self):
        from unittest import mock

        class Dead:
            status_code = 429
            headers = {}
            text = 'GenerateRequestsPerDayPerProjectPerModel "retryDelay": "4771s"'

        env = {"GEMINI_API_KEY": "key-one", "GEMINI_API_KEY_2": "key-two", "GOOGLE_API_KEY": ""}
        with mock.patch.object(llm.requests, "post", return_value=Dead()) as post, \
                mock.patch.dict(os.environ, env):
            with self.assertRaises(llm.QuotaExhausted) as cm:
                llm.call_llm("s", "u", cfg=self.cfg, use_cache=False)
            self.assertEqual(post.call_count, 2)                 # one attempt per key, no retries
            self.assertIn("all 2 keys", str(cm.exception))
            self.assertIn("fake-model", llm.quota_exhausted())

    def test_key_helper_ignores_blanks_and_duplicates(self):
        from unittest import mock
        env = {"GEMINI_API_KEY": "a", "GEMINI_API_KEY_2": "a", "GEMINI_API_KEY_3": " b ", "GOOGLE_API_KEY": ""}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(llm._gemini_keys(), ["a", "b"])

    def test_cached_answers_still_work_after_quota_is_exhausted(self):
        cfg = llm.LLMConfig(provider="mock", model="m1", cache_dir=os.path.join(self.tmp.name, "c2"),
                            raw_dir=os.path.join(self.tmp.name, "r2"))
        first = llm.call_llm("sys", "RATE_WORDS: calm, furious", cfg=cfg)
        llm.EXHAUSTED["m1"] = "daily quota exhausted for model m1"
        self.assertEqual(llm.call_llm("sys", "RATE_WORDS: calm, furious", cfg=cfg), first)
        with self.assertRaises(llm.QuotaExhausted):          # uncached prompt: fails fast
            llm.call_llm("sys", "RATE_WORDS: something else", cfg=cfg)

    def test_pipeline_degrades_to_curve_only_instead_of_crashing_or_retrying(self):
        from confirm import confirm_windows
        from lexicon import FALLBACK
        from tension import candidate_windows
        llm.EXHAUSTED["fake-model"] = "daily quota exhausted for model fake-model"
        ep = tagged_episode()
        wins = candidate_windows(ep, dict(FALLBACK), k=3)
        res = confirm_windows(wins, [ep], cfg=self.cfg, log=lambda *_: None)
        self.assertTrue(res["candidates"])
        self.assertTrue(all(c["source"] == "curve-only" for c in res["candidates"]))


class ConfirmMockTests(unittest.TestCase):
    def test_confirmed_windows_still_pass_validator_and_checks(self):
        from confirm import confirm_windows
        from lexicon import FALLBACK
        from tension import candidate_windows
        with tempfile.TemporaryDirectory() as tmp:
            cfg = llm.LLMConfig(provider="mock", cache_dir=os.path.join(tmp, "c"), raw_dir=os.path.join(tmp, "r"))
            ep = tagged_episode()
            wins = candidate_windows(ep, dict(FALLBACK), k=3)
            res = confirm_windows(wins, [ep], cfg=cfg, log=lambda *_: None)
            self.assertEqual(len(res["candidates"]) + len(res["rejected"]) + len(res["dropped"]), len(wins))
            for c in res["candidates"]:
                self.assertEqual(c["source"], "llm-confirmed")
                self.assertTrue(all(check_promo(c, ep).values()))


class ConfidenceTierTests(unittest.TestCase):
    def clip(self, source="llm-confirmed", scored=True):
        return {"source": source, "panel": {"retention": 0.5} if scored else None}

    def test_tiers(self):
        from rank import confidence_tier as ct
        self.assertEqual(ct([self.clip() for _ in range(4)])["tier"], "full")
        self.assertEqual(ct([self.clip(), self.clip(), self.clip("curve-only")])["tier"], "partial")
        self.assertEqual(ct([self.clip(scored=False)])["tier"], "partial")       # confirmed but never scored
        self.assertEqual(ct([self.clip("curve-only", False), self.clip("curve-only", False),
                             self.clip()])["tier"], "degraded")
        self.assertEqual(ct([self.clip("curve-only", False)])["tier"], "degraded")
        self.assertEqual(ct([self.clip("curve-only", False)], offline=True)["tier"], "offline")
        self.assertEqual(ct([self.clip()], mock=True)["tier"], "mock")
        self.assertEqual(ct([])["tier"], "empty")

    def test_counts_and_message_are_reported(self):
        from rank import confidence_tier as ct
        t = ct([self.clip(), self.clip("curve-only", False)])
        self.assertEqual(t["counts"], {"clips": 2, "model_selected": 1, "curve_only": 1, "model_scored": 1,
                                       "unscored_missing_from_ranking": 0})
        self.assertTrue(t["message"])

    def test_candidates_lost_before_ranking_stop_a_run_from_reading_as_full(self):
        # Regression: a run that silently lost 3 unscored candidates was reported as FULL
        from rank import confidence_tier as ct
        clips = [self.clip() for _ in range(4)]
        self.assertEqual(ct(clips, unscored=0)["tier"], "full")
        t = ct(clips, unscored=3)
        self.assertEqual(t["tier"], "partial")
        self.assertIn("3 candidate(s)", t["message"])
        self.assertEqual(t["counts"]["unscored_missing_from_ranking"], 3)


class JsonSalvageTests(unittest.TestCase):
    VIEW = '{"id": "%s", "viewers": [{"persona": 1, "swipe_at_s": null, "tap_next": 0.8}]}'

    def damaged(self, tail):
        return '{"views": [' + self.VIEW % "a" + ", " + self.VIEW % "b" + tail

    def test_complete_objects_survive_a_damaged_tail(self):
        for tail in ("].}", "] DX]}", "]}  extra words", ",]}"):
            got = llm.salvage_list(self.damaged(tail), "views")
            self.assertEqual([v["id"] for v in got], ["a", "b"], repr(tail))

    def test_a_trailing_comma_inside_the_last_object_is_forgiven(self):
        text = '{"views": [' + self.VIEW % "a" + ', {"id": "b", "viewers": [{"persona": 1, "tap_next": 0.5},]}]}'
        self.assertEqual([v["id"] for v in llm.salvage_list(text, "views")], ["a", "b"])

    def test_a_truncated_object_is_dropped_not_guessed(self):
        text = '{"views": [' + self.VIEW % "a" + ', {"id": "b", "viewers": [{"persona": 1, "tap_ne'
        self.assertEqual([v["id"] for v in llm.salvage_list(text, "views")], ["a"])

    def test_braces_and_commas_inside_strings_are_not_structure(self):
        text = '{"views": [{"id": "a, ]", "note": "has } and { and ,] inside"}, {"id": "b"}]}'
        self.assertEqual([v["id"] for v in llm.salvage_list(text, "views")], ["a, ]", "b"])
        self.assertEqual(llm.salvage_list("no such key here", "views"), [])

    def test_extract_json_alone_now_forgives_trailing_commas(self):
        self.assertEqual(llm.extract_json('{"a": [1, 2,], "b": {"c": 1,},}'), {"a": [1, 2], "b": {"c": 1}})

    # object "b" is missing a comma INSIDE it: no whole-document fallback can parse this
    BROKEN_B = '{"id": "b", "viewers": [{"persona": 1 "tap_next": 0.8}]}'

    def test_call_llm_salvages_instead_of_paying_for_a_repair_call(self):
        calls = []
        text = '{"views": [' + self.VIEW % "a" + ", " + self.BROKEN_B + ", " + self.VIEW % "c" + "]}"

        def flaky(system, user, schema, cfg):
            calls.append(1)
            return text

        llm.PROVIDERS["flaky"] = flaky
        self.addCleanup(llm.PROVIDERS.pop, "flaky", None)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = llm.LLMConfig(provider="flaky", cache_dir=os.path.join(tmp, "c"), raw_dir=os.path.join(tmp, "r"))
            before = llm.USAGE.salvaged
            got = llm.call_llm("s", "u", cfg=cfg, salvage_key="views")
            self.assertEqual([v["id"] for v in got["views"]], ["a", "c"])   # b lost, a and c kept
            self.assertEqual(len(calls), 1, "no repair call when objects can be salvaged")
            self.assertEqual(llm.USAGE.salvaged, before + 1)
            # without a salvage key the old behaviour stands: repair attempt, then a clear failure
            with self.assertRaises(llm.LLMError):
                llm.call_llm("s2", "u2", cfg=cfg)
            self.assertEqual(len(calls), 3)

    def test_a_clip_that_keeps_failing_is_retried_alone_then_left_unscored_never_invented(self):
        import re
        import panel
        ep = tagged_episode()
        cands = [{"id": f"ep1-w{i}", "episode": 1, "start_s": 31.0, "end_s": 62.0, "duration_s": 31.0}
                 for i in (1, 2, 3)]
        asked = []
        good = ('{"id": "%s", "viewers": [{"persona": 1, "swipe_at_s": null, "tap_next": 0.9},'
                ' {"persona": 2, "swipe_at_s": 4.0}, {"persona": 3, "swipe_at_s": 2.0}]}')

        def stubborn(system, user, schema, cfg):
            ids = re.findall(r"^CANDIDATE (\S+)", user, re.M)
            asked.append(ids)
            ok = [good % i for i in ids if i == "ep1-w1"]       # w2 and w3 are never answered
            return '{"views": [' + ", ".join(ok) + "]}"

        llm.PROVIDERS["stubborn"] = stubborn
        self.addCleanup(llm.PROVIDERS.pop, "stubborn", None)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = llm.LLMConfig(provider="stubborn", cache_dir=os.path.join(tmp, "c"), raw_dir=os.path.join(tmp, "r"))
            panel.run_panel(cands, {1: ep}, cfg, None, log=lambda *_: None)
        self.assertIsNotNone(cands[0]["panel"])
        self.assertIsNone(cands[1]["panel"], "an unanswered clip stays unscored; no score is invented")
        self.assertIsNone(cands[2]["panel"])
        # full batch, then the two stragglers together, then each ALONE as the last resort
        self.assertEqual(asked, [["ep1-w1", "ep1-w2", "ep1-w3"], ["ep1-w2", "ep1-w3"], ["ep1-w2"], ["ep1-w3"]])

    def test_panel_keeps_the_good_views_and_retries_only_the_missing_one(self):
        import re
        import panel
        ep = tagged_episode()
        cands = [{"id": f"ep1-w{i}", "episode": 1, "start_s": 31.0, "end_s": 62.0, "duration_s": 31.0}
                 for i in (1, 2, 3)]
        asked = []

        def view(cid):
            return ('{"id": "%s", "viewers": [{"persona": 1, "swipe_at_s": null, "tap_next": 0.9},'
                    ' {"persona": 2, "swipe_at_s": 4.0, "tap_next": 0}, {"persona": 3, "swipe_at_s": 2.0}]}' % cid)

        def flaky(system, user, schema, cfg):
            ids = re.findall(r"^CANDIDATE (\S+)", user, re.M)
            asked.append(ids)
            if len(asked) == 1:                       # first reply: the middle view is corrupted inside
                broken = view(ids[1]).replace('"persona": 1, "swipe_at_s"', '"persona": 1 "swipe_at_s"')
                return '{"views": [' + view(ids[0]) + ", " + broken + ", " + view(ids[2]) + "]}"
            return '{"views": [' + ", ".join(view(i) for i in ids) + "]}"

        llm.PROVIDERS["flaky2"] = flaky
        self.addCleanup(llm.PROVIDERS.pop, "flaky2", None)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = llm.LLMConfig(provider="flaky2", cache_dir=os.path.join(tmp, "c"), raw_dir=os.path.join(tmp, "r"))
            panel.run_panel(cands, {1: ep}, cfg, None, log=lambda *_: None)
        self.assertTrue(all(c["panel"] is not None for c in cands), "every clip must end up scored")
        self.assertEqual(len(asked), 2, "one damaged batch + one retry of ONLY what was missing")
        self.assertEqual(asked[1], ["ep1-w2"])


class PanelIsolationTests(unittest.TestCase):
    def test_panel_prompt_contains_none_of_the_selectors_reasoning(self):
        import panel
        ep = tagged_episode()
        ep["logline"] = "ZZ-LOGLINE-LEAK"
        ep["continuity"] = "ZZ-CONTINUITY-LEAK"
        cand = {"id": "ep1-w1", "episode": 1, "start_s": 31.0, "end_s": 62.0, "duration_s": 31.0,
                "why": "ZZ-WHY-LEAK", "standalone_notes": "ZZ-NOTES-LEAK", "archetype": "ZZ-ARCH-LEAK",
                "source": "ZZ-SRC-LEAK", "curve": {"score": 0.987654}, "spoils_arc": True}
        prompt = panel._prompt([cand], {1: ep})
        for leak in ("ZZ-WHY-LEAK", "ZZ-NOTES-LEAK", "ZZ-ARCH-LEAK", "ZZ-SRC-LEAK", "0.987654",
                     "ZZ-LOGLINE-LEAK", "ZZ-CONTINUITY-LEAK"):
            self.assertNotIn(leak, prompt)
        self.assertIn("CANDIDATE ep1-w1", prompt)
        self.assertIn("Get out now!", prompt)        # ...but the clip's own script text IS there


class EstimatorTests(unittest.TestCase):
    def test_dry_run_makes_no_http_call_and_cached_count_matches_reality(self):
        from unittest import mock
        from confirm import confirm_windows
        from estimate import estimate, render
        from lexicon import FALLBACK
        from tension import find_all
        eps = [tagged_episode(1), tagged_episode(2)]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = llm.LLMConfig(provider="mock", cache_dir=os.path.join(tmp, "c"), raw_dir=os.path.join(tmp, "r"))
            lex = dict(FALLBACK)
            with mock.patch.object(llm.requests, "post", side_effect=AssertionError("dry run called the API")):
                before = estimate(eps, cfg, cfg.model, lex)
            self.assertGreater(before["windows"], 0)
            self.assertEqual(before["confirm"]["cached"], 0)
            self.assertFalse(before["lexicon"]["cached"])
            self.assertEqual(before["confirm"]["calls"], -(-before["windows"] // 12))

            self.assertIsNone(before["panel"]["exact"], "panel cannot be exact while confirm is uncached")

            # now really run confirm on the mock, populating the cache ...
            res = confirm_windows(find_all(eps, lex), eps, cfg=cfg, log=lambda *_: None)
            mid = estimate(eps, cfg, cfg.model, lex)
            # ... and the estimator must report exactly those calls as cached
            self.assertEqual(mid["confirm"]["cached"], mid["confirm"]["calls"])
            # confirm is fully cached, so the panel stage is now computed EXACTLY by replaying confirm
            self.assertIsNotNone(mid["panel"]["exact"])
            self.assertEqual(mid["panel"]["exact"]["clips"], len(res["candidates"]))
            self.assertEqual(mid["panel"]["exact"]["cached"], 0)       # panel not run yet
            # mock model serves both stages, so their NEW calls must add, not overwrite
            self.assertEqual(mid["new_calls"][cfg.model], 1 + mid["panel"]["exact"]["calls"])

            # run the panel for real (mock) on those same candidates ...
            from rank import rank_candidates
            rank_candidates(res["candidates"], eps, cfg=cfg, judge_model=cfg.model, scorer="panel",
                            log=lambda *_: None)
            after = estimate(eps, cfg, cfg.model, lex)
            self.assertEqual(after["panel"]["exact"]["cached"], after["panel"]["exact"]["calls"])
            self.assertEqual(after["new_calls"][cfg.model], 1)        # only the lexicon is still new
            text = render(after)
            self.assertIn("DRY RUN", text)
            self.assertIn("free daily limit", text)
            self.assertNotIn("$", text)                               # deliberately no dollar figure


class ClipTextTests(unittest.TestCase):
    def setUp(self):
        self.ep = parse_episode(make_screenplay(1))      # CAMERA beat 28.0-31.0 "Close-up on Bob's face."

    def test_a_beat_already_playing_when_the_clip_opens_is_shown_to_the_panel(self):
        from rank import clip_lines
        lines = clip_lines(self.ep, {"start_s": 29.0, "end_s": 55.0})
        carried = [ln for ln in lines if "Close-up on Bob" in ln]
        self.assertEqual(len(carried), 1)
        self.assertTrue(carried[0].startswith("+00.0s"))
        self.assertIn("already on screen", carried[0])

    def test_a_beat_that_is_about_to_end_is_not_dragged_in(self):
        from rank import clip_lines
        # camera beat ends at 31.0; a clip opening at 30.8 sees only 0.2s of it
        self.assertFalse(any("Close-up on Bob" in ln for ln in clip_lines(self.ep, {"start_s": 30.8, "end_s": 56.0})))

    def test_dialogue_is_never_carried_in_when_the_start_is_snapped(self):
        from rank import clip_beat_list
        from select_promos import snap_window
        s, e = snap_window(self.ep["beats"], 31.0, 56.0)           # 29-33s line straddles 31.0 -> pulled back
        carried_dialogue = [b for b, carried in clip_beat_list(self.ep, {"start_s": s, "end_s": e})
                            if carried and b["is_dialogue"]]
        self.assertEqual(carried_dialogue, [])

    def test_evidence_can_cite_a_carried_in_beat(self):
        from panel import clip_beats, score_panel
        beats = clip_beats(self.ep, {"start_s": 29.0, "end_s": 55.0})
        v = [{"persona": 1, "swipe_at_s": 0.5, "tap_next": 0, "reason": "r",
              "evidence": {"quote": "Close-up on Bob's face"}},
             {"persona": 2, "swipe_at_s": None, "tap_next": 0.5, "reason": "r"}]
        self.assertEqual(score_panel(26.0, v, beats)["viewers"][0]["evidence_status"], "supported")

    def test_skeptic_persona_can_be_won_over(self):
        import panel
        self.assertIn("CAN be won over", panel.SYSTEM)


class ProvenanceTests(unittest.TestCase):
    def test_every_ranked_clip_carries_a_consistent_trail(self):
        from confirm import confirm_windows
        from lexicon import FALLBACK
        from rank import rank_candidates
        from report import write_html
        from tension import find_all
        eps = [tagged_episode(n) for n in range(1, 4)]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = llm.LLMConfig(provider="mock", cache_dir=os.path.join(tmp, "c"), raw_dir=os.path.join(tmp, "r"))
            sel = confirm_windows(find_all(eps, dict(FALLBACK)), eps, cfg=cfg, log=lambda *_: None)
            out = rank_candidates(sel["candidates"], eps, cfg=cfg, judge_model=cfg.model, scorer="panel",
                                  log=lambda *_: None)
            self.assertTrue(out["ranked"])
            for c in out["ranked"]:
                stages = [s["stage"] for s in c["provenance"]]
                self.assertEqual(stages[0], "curve")
                self.assertEqual(stages[-1], "rank")
                self.assertIn("panel", stages)
                curve = c["provenance"][0]
                self.assertEqual([curve["start_s"], curve["end_s"]], c["curve_window"])
                rank = c["provenance"][-1]
                self.assertEqual((rank["rank"], rank["final_score"]), (c["rank"], c["final_score"]))
                confirm = next(s for s in c["provenance"] if s["stage"] == "confirm")
                self.assertEqual(confirm["verdict"], "trim" if c.get("trimmed_by_model") else "keep")
            path = os.path.join(tmp, "r.html")
            write_html(path, out["ranked"], {e["episode_no"]: e for e in eps}, {"tier": "mock", "tier_message": "m"})
            with open(path, encoding="utf-8") as fh:
                self.assertIn("how this clip was chosen", fh.read())


class PanelEvidenceTests(unittest.TestCase):
    BEATS = [
        {"start_off": 0.0, "end_off": 4.0, "text": "Close-up on Alice's shaking hands."},
        {"start_off": 5.0, "end_off": 8.0, "text": '"You will never leave this house."'},
        {"start_off": 12.0, "end_off": 14.0, "text": '"Then watch me burn it down."'},
    ]

    def viewer(self, swipe, quote, persona=1, tap=0.0):
        v = {"persona": persona, "swipe_at_s": swipe, "tap_next": tap, "reason": "r"}
        if quote is not None:
            v["evidence"] = {"at_s": swipe, "quote": quote}
        return v

    def score(self, viewers, beats="default"):
        from panel import score_panel
        return score_panel(30.0, viewers, self.BEATS if beats == "default" else beats)

    def status(self, v):
        return self.score([v, self.viewer(None, None, 2, 0.5)])["viewers"][0]["evidence_status"]

    def test_real_quote_on_screen_at_the_swipe_is_supported(self):
        self.assertEqual(self.status(self.viewer(6.0, "never leave this house")), "supported")

    def test_wrapping_quotes_and_ellipsis_are_tolerated(self):
        self.assertEqual(self.status(self.viewer(6.0, '"never leave this..."')), "supported")

    def test_invented_or_paraphrased_quote_is_unsupported(self):
        self.assertEqual(self.status(self.viewer(6.0, "I will never let you go")), "unsupported")
        self.assertEqual(self.status(self.viewer(6.0, "you shall never depart this home")), "unsupported")
        self.assertEqual(self.status(self.viewer(6.0, "ab")), "unsupported")          # too short to mean anything

    def test_real_quote_but_not_playing_near_the_swipe_is_flagged_as_timing_off(self):
        # the line exists (5-8s) but this viewer swiped at 25s
        self.assertEqual(self.status(self.viewer(25.0, "never leave this house")), "quote_ok_timing_off")

    def test_a_viewer_who_watched_to_the_end_is_checked_on_the_quote_only(self):
        self.assertEqual(self.status(self.viewer(None, "watch me burn it down", tap=0.8)), "supported")
        self.assertEqual(self.status(self.viewer(None, "a line nobody says", tap=0.8)), "unsupported")

    def test_missing_evidence_is_none_not_unsupported(self):
        self.assertEqual(self.status(self.viewer(6.0, None)), "none")

    def test_evidence_can_arrive_as_a_bare_string(self):
        v = self.viewer(6.0, None)
        v["evidence"] = "never leave this house"
        self.assertEqual(self.status(v), "supported")

    def test_verification_never_changes_a_score(self):
        views = [self.viewer(None, "made up words here", 1, 0.9), self.viewer(6.0, "never leave this", 2),
                 self.viewer(25.0, "burn it down", 3)]
        with_check = self.score(views)
        without = self.score(views, beats=None)
        for k in ("hook_survival", "watch_through", "tap_next", "retention", "swipe_spread"):
            self.assertEqual(with_check[k], without[k], k)
        self.assertEqual(without["viewers"][0]["evidence_status"], "unchecked")

    def test_review_flag_needs_two_cited_and_fewer_than_half_supported(self):
        bad = lambda p: self.viewer(6.0, "totally invented words", p)           # noqa: E731
        good = lambda p: self.viewer(6.0, "never leave this house", p)          # noqa: E731
        self.assertTrue(self.score([bad(1), bad(2), bad(3)])["needs_review"])
        self.assertTrue(self.score([good(1), bad(2), bad(3)])["needs_review"])          # 1/3 supported
        self.assertFalse(self.score([good(1), good(2), bad(3)])["needs_review"])        # 2/3 supported
        self.assertFalse(self.score([bad(1), self.viewer(6.0, None, 2), self.viewer(7.0, None, 3)])["needs_review"])
        s = self.score([good(1), bad(2), bad(3)])
        self.assertEqual(s["evidence"], {"supported": 1, "quote_ok_timing_off": 0, "unsupported": 2, "none": 0})
        self.assertAlmostEqual(s["evidence_support"], 1 / 3, places=2)

    def test_panel_prompt_demands_evidence_and_mock_run_verifies_it_end_to_end(self):
        import panel
        from confirm import confirm_windows
        from lexicon import FALLBACK
        from rank import rank_candidates
        from tension import find_all
        self.assertIn("evidence", panel.SYSTEM)
        eps = [tagged_episode(n) for n in range(1, 6)]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = llm.LLMConfig(provider="mock", cache_dir=os.path.join(tmp, "c"), raw_dir=os.path.join(tmp, "r"))
            sel = confirm_windows(find_all(eps, dict(FALLBACK)), eps, cfg=cfg, log=lambda *_: None)
            out = rank_candidates(sel["candidates"], eps, cfg=cfg, judge_model=cfg.model, scorer="panel",
                                  log=lambda *_: None)
            sts = [v["evidence_status"] for c in out["ranked"] for v in c["panel"]["viewers"]]
            self.assertTrue(sts)
            self.assertIn("supported", sts)                       # honest mock evidence verifies
            self.assertTrue(all(s != "unchecked" for s in sts))   # the pipeline always runs the check
            self.assertIn("evidence_support", out["ranked"][0]["panel"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
