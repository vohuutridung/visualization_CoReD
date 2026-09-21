from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from visualization.cache import JsonlCache
from visualization.data import TrajectoryParseError, deterministic_subset, parse_longcot
from visualization.metrics.answer_gain import answer_gains
from visualization.metrics.branching import expert_pair_count, semantic_branching
from visualization.metrics.removal import removal_damages
from visualization.metrics.repetition import max_previous_bleu, max_previous_semantic
from visualization.prompts import answer_gain_user_prompt, removal_user_prompt, render_chat_prompt
from visualization.statistics import describe
from visualization.weights import (
    StandardizationStats,
    WeightParameters,
    assign_high_weight,
    compute_weight,
    render_teacher_forcing_text,
)


def row(longcot: str) -> dict[str, str]:
    return {"uuid": "u1", "problem": "What is 1+1?", "answer": "2", "longcot": longcot}


class ParsingTests(unittest.TestCase):
    def test_exact_split_empty_removal_and_last_step_exclusion(self) -> None:
        parsed = parse_longcot(
            row("polished before<think>  first  \n\n\n\n second $x$ \n\n final answer  </think>polished after")
        )
        self.assertEqual(parsed.eligible_steps, ("first", "second $x$"))
        self.assertEqual(parsed.excluded_last_step_text, "final answer")
        self.assertEqual(parsed.original_num_steps, 3)
        self.assertEqual(parsed.outside_before_think, "polished before")
        self.assertEqual(parsed.outside_after_think, "polished after")

    def test_missing_tags_are_categorized(self) -> None:
        with self.assertRaises(TrajectoryParseError) as context:
            parse_longcot(row("no tags"))
        self.assertEqual(context.exception.reason, "missing_think_open")
        with self.assertRaises(TrajectoryParseError) as context:
            parse_longcot(row("<think>unfinished"))
        self.assertEqual(context.exception.reason, "missing_think_close")

    def test_only_answer_step_is_invalid(self) -> None:
        with self.assertRaises(TrajectoryParseError) as context:
            parse_longcot(row("<think>answer only</think>"))
        self.assertEqual(context.exception.reason, "no_eligible_step")

    def test_deterministic_subset_ignores_input_order(self) -> None:
        first = deterministic_subset(["c", "a", "b", "d"], 3, seed=42)
        second = deterministic_subset(["d", "b", "c", "a"], 3, seed=42)
        self.assertEqual(first, second)


class WeightTests(unittest.TestCase):
    def test_ceil_quarter_and_stable_ties(self) -> None:
        labels = assign_high_weight([1.0] * 5)
        self.assertEqual(sum(labels), 2)
        self.assertEqual(labels, (True, True, False, False, False))
        labels = assign_high_weight([0.1, 0.9, 0.8, 0.2])
        self.assertEqual(labels, (False, True, False, False))

    def test_weight_formula_and_floor(self) -> None:
        stats = StandardizationStats(1.0, 2.0, 2.0, 4.0, "test")
        parameters = WeightParameters(0.4, 0.3, weight_floor=0.05)
        weight, u_hat, rho_hat = compute_weight(3.0, 6.0, stats, parameters)
        self.assertEqual((u_hat, rho_hat), (1.0, 1.0))
        self.assertAlmostEqual(weight, 1 + 0.7 * math.tanh(1.0))


class MetricTests(unittest.TestCase):
    def test_answer_gain_arithmetic(self) -> None:
        self.assertEqual(answer_gains([0.25, 0.5, 0.375]), [0.25, -0.125])

    def test_removal_damage_arithmetic(self) -> None:
        self.assertEqual(removal_damages(0.75, [0.5, 1.0]), [0.25, -0.25])

    def test_max_previous_bleu_indexing(self) -> None:
        scores = {
            ("b", "a"): 0.2,
            ("c", "a"): 0.8,
            ("c", "b"): 0.3,
        }
        with mock.patch(
            "visualization.metrics.repetition.sentence_bleu4",
            side_effect=lambda current, previous: scores[(current, previous)],
        ):
            self.assertEqual(max_previous_bleu(["a", "b", "c"]), [None, 0.2, 0.8])

    def test_max_previous_semantic_indexing(self) -> None:
        embeddings = [(1.0, 0.0), (0.0, 1.0), (0.8, 0.6)]
        values = max_previous_semantic(embeddings)
        self.assertIsNone(values[0])
        self.assertAlmostEqual(values[1], 0.0)
        self.assertAlmostEqual(values[2], 0.8)

    def test_pair_count_and_branching(self) -> None:
        self.assertEqual(expert_pair_count(4), 6)
        score = semantic_branching([(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)])
        self.assertAlmostEqual(score, (1.0 + 2.0 + 1.0) / 3)


class PromptTests(unittest.TestCase):
    def test_answer_gain_does_not_request_immediate_answer(self) -> None:
        prompt = answer_gain_user_prompt("P", ["S1", "S2"])
        self.assertIn("Continue solving", prompt)
        self.assertIn("S1\n\nS2", prompt)
        self.assertNotIn("High-W", prompt)

    def test_removal_simply_omits_step(self) -> None:
        prompt = removal_user_prompt("P", ["S1", "S3"])
        self.assertIn("S1\n\nS3", prompt)
        self.assertNotIn("REMOVED", prompt)
        self.assertNotIn("S2", prompt)

    def test_qwen3_hard_switch_is_explicit(self) -> None:
        class Tokenizer:
            kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "rendered"

        tokenizer = Tokenizer()
        self.assertEqual(
            render_chat_prompt(tokenizer, "hello", is_qwen3=True, enable_thinking=False),
            "rendered",
        )
        self.assertIs(tokenizer.kwargs["enable_thinking"], False)

    def test_teacher_forcing_excludes_polished_outside_text(self) -> None:
        parsed = parse_longcot(
            row("polished-before<think>step one\n\nanswer step</think>polished-after")
        )

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return messages[0]["content"] + "|" + messages[1]["content"]

        tokenizer = Tokenizer()
        rendered, assistant = render_teacher_forcing_text(
            tokenizer, parsed, enable_thinking=None
        )
        self.assertEqual(assistant, "step one\n\nanswer step")
        self.assertNotIn("polished-before", rendered)
        self.assertNotIn("polished-after", rendered)


class CacheTests(unittest.TestCase):
    def test_resume_and_duplicate_skip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            cache = JsonlCache(path, key=lambda value: value["id"])
            self.assertTrue(cache.append({"id": 1, "value": "a"}))
            self.assertFalse(cache.append({"id": 1, "value": "duplicate"}))
            resumed = JsonlCache(path, key=lambda value: value["id"])
            self.assertEqual(resumed.get(1)["value"], "a")

    def test_truncated_final_record_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            path.write_text(json.dumps({"id": 1}) + "\n{\"id\":", encoding="utf-8")
            resumed = JsonlCache(path, key=lambda value: value["id"])
            self.assertIn(1, resumed)
            self.assertTrue(resumed.append({"id": 2}))
            final = JsonlCache(path, key=lambda value: value["id"])
            self.assertEqual(set(final.records), {1, 2})


class StatisticsTests(unittest.TestCase):
    def test_describe(self) -> None:
        result = describe([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["median"], 2.5)
        self.assertEqual(result["q25"], 1.75)
        self.assertEqual(result["q75"], 3.25)


if __name__ == "__main__":
    unittest.main()
