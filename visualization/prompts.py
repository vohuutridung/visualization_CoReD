from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .data import ParsedTrajectory, STEP_SEPARATOR


@dataclass(frozen=True)
class LeakageAudit:
    gold_answer_supplied: bool
    excluded_step_supplied: bool
    outside_think_supplied: bool
    label_terms_supplied: bool

    @property
    def passed(self) -> bool:
        return not any(
            (
                self.gold_answer_supplied,
                self.excluded_step_supplied,
                self.outside_think_supplied,
                self.label_terms_supplied,
            )
        )


def answer_gain_user_prompt(problem: str, prefix_steps: Sequence[str]) -> str:
    parts = [f"Problem:\n{problem.strip()}"]
    if prefix_steps:
        parts.append("Provided reasoning prefix:\n" + STEP_SEPARATOR.join(prefix_steps))
    parts.append(
        "Continue solving the problem step by step from the provided context. "
        "Conclude with the final answer in \\boxed{}."
    )
    return STEP_SEPARATOR.join(parts)


def removal_user_prompt(problem: str, reasoning_steps: Sequence[str]) -> str:
    reasoning = STEP_SEPARATOR.join(reasoning_steps)
    return (
        f"Problem:\n{problem.strip()}\n\n"
        f"Reasoning:\n{reasoning}\n\n"
        "Based on the reasoning above, provide only the final answer in \\boxed{}."
    )


def branching_user_prompt(problem: str, prefix_steps: Sequence[str]) -> str:
    parts = [f"Problem:\n{problem.strip()}"]
    if prefix_steps:
        parts.append("Reasoning so far:\n" + STEP_SEPARATOR.join(prefix_steps))
    parts.append("Write the next single reasoning step.")
    return STEP_SEPARATOR.join(parts)


def render_chat_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    is_qwen3: bool,
    enable_thinking: bool = False,
) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if is_qwen3:
        # Qwen's official hard switch; omission defaults to thinking mode.
        kwargs["enable_thinking"] = enable_thinking
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}], **kwargs
    )
    if is_qwen3 and not enable_thinking and "<think>" in rendered:
        raise RuntimeError(
            "Qwen3 non-thinking chat template unexpectedly emitted an open <think> block"
        )
    return rendered


def audit_prompt_sources(
    trajectory: ParsedTrajectory,
    supplied_steps: Sequence[str],
    user_prompt: str,
) -> LeakageAudit:
    """Audit construction inputs without false alarms for coincidental substrings.

    A forbidden string that already occurs in the problem or an explicitly supplied
    eligible step is not evidence of accidental field leakage.
    """

    allowed = trajectory.problem + "\n" + "\n".join(supplied_steps)

    def unexpectedly_present(value: str) -> bool:
        cleaned = value.strip()
        return bool(cleaned and cleaned not in allowed and cleaned in user_prompt)

    lowered = user_prompt.lower()
    return LeakageAudit(
        gold_answer_supplied=unexpectedly_present(trajectory.answer),
        excluded_step_supplied=unexpectedly_present(trajectory.excluded_last_step_text),
        outside_think_supplied=(
            unexpectedly_present(trajectory.outside_before_think)
            or unexpectedly_present(trajectory.outside_after_think)
        ),
        label_terms_supplied=any(
            term in lowered
            for term in (
                "high-weight",
                "low-weight",
                "high weight",
                "low weight",
                "cored",
                "expert disagreement",
            )
        ),
    )


def require_clean_audit(audit: LeakageAudit) -> None:
    if not audit.passed:
        raise ValueError(f"prompt leakage audit failed: {audit}")
