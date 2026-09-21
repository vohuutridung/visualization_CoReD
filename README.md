# CoReD functional-step analysis

This repository implements the held-out analysis pipeline for **Council-Weighted Reasoning Distillation (CoReD)**. It asks whether the top 25% of CoReD-weighted reasoning steps within each trajectory are independently more useful, less redundant, and more likely to produce expert branching than the remaining 75%.

The implementation is measurement-only. It never trains or updates the base model, council experts, embedding model, or Phase-2 student.

## Scientific contract

- Dataset: `vohuutridung/data_ablation_phase1_loss`; only `problem`, `answer`, `longcot`, and `uuid` are loaded.
- Reasoning source: only the first `<think>...</think>` region. It is split exactly on `"\n\n"`; chunks are stripped, empty chunks removed, and only then the final non-empty answer step is excluded.
- Council teacher forcing renders only that parsed think content as the assistant trajectory; polished text outside the tags is never supplied. The assistant-content wrapper is configurable as `council.teacher_forcing_content_template` if the original Phase-2 run used an additional structural wrapper.
- No adaptive trajectory truncation is performed.
- Groups: exactly `max(1, ceil(0.25*K))` steps per sample are High-W, with original step index breaking exact weight ties. Grouping is never global.
- Answer Gain and Removal use the original frozen Hugging Face base model, with no LoRA or student adapter.
- Qwen3 prompts use the official hard switch `tokenizer.apply_chat_template(..., enable_thinking=False)`.
- Mathematical correctness is centralized through `math-verify`; parse and runtime failures are reported separately. Empirical correctness remains `# correct / configured rollouts`, so malformed model answers cannot silently disappear from the denominator, while infrastructure failures prevent that condition from being summarized.
- Semantic repetition and branching use the fixed `Qwen/Qwen3-Embedding-0.6B` model with symmetric, instruction-free inputs, explicit end-of-document tokens, last-token pooling, and L2 normalization.
- Every expensive generation is written to durable JSONL immediately and keyed at `uuid + condition + rollout` (or `uuid + step + expert`) for safe resume.

## Important Phase-2 configuration requirement

The target repository was initially empty, so there was no Phase-2 implementation to import. The local paper source supplies the exact signal and weight equations, but its `lambda_U` and `lambda_D` values are still marked TODO. This project therefore refuses to guess them.

Final runs must provide:

1. the actual Phase-1 expert checkpoint directory;
2. the exact `lambda_U` and `lambda_D` used by Phase 2; and
3. the Phase-2 **training-corpus** standardization moments:

```json
{
  "mean_u": 0.0,
  "std_u": 1.0,
  "mean_rho": 0.0,
  "std_rho": 1.0,
  "source": "description or path of the Phase-2 training-statistics artifact"
}
```

The numeric values above illustrate the file schema only; they are not experimental defaults. For a smoke test, `--fit-standardization-on-analysis` is available and explicitly records that methodological deviation. It must not be used for the paper result when the original training statistics exist.

Weights are computed from full-vocabulary teacher-forced distributions:

```text
U_i   = mean token entropy across experts
V_i   = entropy(mean expert distribution) - mean expert entropy
rho_i = V_i / (U_i + epsilon)
w_i   = max(w_min, 1 + lambda_U*tanh(U_hat_i) + lambda_D*tanh(rho_hat_i))
```

The default checkpoint layout is configurable and expects:

```text
COUNCIL_DIR/
  expert_0/
  expert_1/
  expert_2/
```

Use `--expert-pattern` if the actual Phase-1 convention differs.

Token-to-step assignment uses the fast tokenizer's character offsets: a predicted token belongs to a step when its rendered span overlaps that stripped raw step. The `"\n\n"` separator itself is not assigned to either step. Standard deviations use population moments, matching a corpus-wide running mean/variance. Both conventions are saved in run provenance; if the original Phase-2 implementation used different boundary ownership, update this adapter rather than silently mixing definitions.

## Installation

Python 3.10+ and CUDA are required for the model-backed stages. Qwen3 requires `transformers>=4.51.0`; generation uses `vllm>=0.8.5`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m pytest
```

## Qwen3-8B run order

Set convenient paths first:

```bash
CONFIG=visualization/config/qwen3_8b.yaml
COUNCIL_DIR=/path/to/qwen3_phase1_council
TRAINING_STATS=/path/to/qwen3_phase2_standardization.json
```

### Phase A — parsing, council signals, and weights

```bash
python scripts/viz_extract_weights.py \
  --config "$CONFIG" \
  --council-checkpoint-dir "$COUNCIL_DIR" \
  --lambda-u ACTUAL_LAMBDA_U \
  --lambda-d ACTUAL_LAMBDA_D \
  --standardization-path "$TRAINING_STATS" \
  --resume
```

This also creates deterministic `answer_gain_500.json` and `expert_branching_1000.json` UUID manifests with seed 42. The manifests live outside the backbone-specific result directory so the Qwen2.5 follow-up reuses the same samples.

### Phase B — repetition over all valid samples

```bash
python scripts/viz_repetition.py \
  --config "$CONFIG" \
  --batch-size 32 \
  --embedding-batch-size 64 \
  --resume
```

BLEU is sentence BLEU-4 over `nltk.wordpunct_tokenize`, with `SmoothingFunction.method1`. Each step is compared separately with every earlier step and the maximum is retained. The first step is excluded.

### Phase C — deterministic expert branching

```bash
python scripts/viz_expert_branching.py \
  --config "$CONFIG" \
  --council-checkpoint-dir "$COUNCIL_DIR" \
  --batch-size 64 \
  --embedding-batch-size 64 \
  --resume
```

The generator holds one base model and switches vLLM LoRA requests. Decoding is greedy, the maximum is 256 new tokens, and `"\n\n"` is the stop delimiter. Semantic mean pairwise cosine distance is primary; symmetric mean pairwise BLEU is auxiliary.

### Phase D — Answer Gain

Validate 5 samples and 2 rollouts before the full run:

```bash
python scripts/viz_answer_gain.py \
  --config "$CONFIG" \
  --limit 5 \
  --num-rollouts 2 \
  --save-prompts \
  --debug \
  --resume
```

Inspect `artifacts/results/qwen3_8b/answer_gain_prompts.jsonl`, generation text, and verifier decisions. Increasing only the rollout count safely extends the cache because every rollout has an independent deterministic seed. Changing prompts, decoding settings, checkpoints, or embedding models is rejected by cache-provenance checks; use a separate output directory for such changes.

Then run 50 samples, inspect again, and run the fixed 500-sample protocol:

```bash
python scripts/viz_answer_gain.py --config "$CONFIG" --limit 50 --resume
python scripts/viz_answer_gain.py --config "$CONFIG" --resume
```

The full configuration uses 16 rollouts per prefix and 32,768 maximum new tokens. Each prefix state `C_0...C_K` is evaluated once; `AG_i = C_i - C_{i-1}` is derived from the cached states.

### Phase E — leave-one-step-out Removal Damage

```bash
python scripts/viz_removal.py \
  --config "$CONFIG" \
  --resume
```

This uses the same 500 UUIDs. Full-context correctness is computed once per sample, and each ablation simply omits one raw step without a marker or repair. The answer-only maximum defaults to 512 tokens and is independently configurable with `--max-new-tokens-removal`.

### Phase F — pooled summaries and figures

```bash
python scripts/viz_make_plots.py --config "$CONFIG"
```

The script reads saved raw results only. It does not rerun inference. Summaries contain sample count, step count, mean, median, population standard deviation, Q25, and Q75 for High-W and Low-W. It reports the observed values even when a hypothesis fails.

For a validated full run, the sequence can be launched with:

```bash
bash scripts/viz_run_all.sh \
  "$CONFIG" "$COUNCIL_DIR" ACTUAL_LAMBDA_U ACTUAL_LAMBDA_D "$TRAINING_STATS"
```

## Outputs

```text
artifacts/
  configs/                     # complete run settings, UUIDs, git commit, timestamps
  subsets/                     # deterministic UUID manifests
  results/qwen3_8b/
    weights_raw.jsonl          # U, V, rho before corpus standardization
    weights.jsonl              # finalized weights and High-W/Low-W labels
    answer_gain_generations.jsonl
    answer_gain_prefixes.jsonl # cached C_i
    answer_gain.jsonl
    removal_generations.jsonl
    removal_conditions.jsonl   # cached C_full and C_minus_i conditions
    removal_damage.jsonl
    repetition.jsonl
    expert_branching_generations.jsonl
    expert_branching.jsonl
  tables/qwen3_8b/
    metric_summary.csv
    metric_summary.json
  figures/
    answer_gain_qwen3_8b.{pdf,png}
    removal_damage_qwen3_8b.{pdf,png}
    bleu_repetition_qwen3_8b.{pdf,png}
    semantic_repetition_qwen3_8b.{pdf,png}
    expert_branching_qwen3_8b.{pdf,png}
    four_panel_qwen3_8b.{pdf,png}
  reports/
    exclusions.json
```

Generated text is kept only in generation-level files, not compact step summaries. Reports count malformed `<think>` regions, too-short trajectories, incomplete caches, generation failures, OOMs, verifier parse failures, embedding failures, empty expert continuations, and non-finite metrics.

## Qwen2.5 follow-up

Use `visualization/config/qwen2_5_7b.yaml` and the matching Phase-1 council/training statistics. Parsing, UUID manifests, group construction, metrics, aggregation, and plotting are unchanged. The Qwen2.5 tokenizer does not receive the Qwen3-only `enable_thinking` template argument.

## Tests

The synthetic suite covers strict `<think>` parsing, exact double-newline splitting, empty-step removal, final-step exclusion, top-25% ceiling and stable ties, Answer Gain and Removal arithmetic, previous-step BLEU/semantic indexing, expert pair count and branching score, explicit Qwen3 non-thinking rendering, deterministic subsets, summary statistics, and interrupted-cache resume behavior.

```bash
python -m unittest discover -s tests -v
```
