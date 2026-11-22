# Kaggle distillation pilot

Upload `notebooks/Sentiment_Distillation_Train_Kaggle.ipynb` to Kaggle, enable GPU + Internet,
and run cells in order. The notebook embeds both Python modules and requirements; no repository
clone or GitHub token is needed. It installs the train libraries on Kaggle after a pip dry-run
that blocks replacing platform PyTorch/CUDA. Nothing is installed or trained locally.

Default: Qwen3-0.6B, up to 5,000 eligible training records, 200 steps per mode, batch 1,
accumulation 8, sequence length 2048, LoRA rank 16, LR 1e-4. Change MODEL to
`Qwen/Qwen3-1.7B` for the second student. These are pilot starting values, not paper hyperparameters
or validated optimal settings. Measure throughput before committing to a longer run.

## Corpus and mapping

The notebook downloads ONLY `Qwen3-32B_50k.jsonl` from
`zhang-yice/sentiment-distillation-v2`, with a pinned dataset revision. Model/tokenizer revision
is resolved to a SHA at prepare time. Each record ID is source-file SHA-256 plus line number.
No teacher inference, soft logits, rewritten answers or generated labels are required.

After audit, fill PROMPT_FIELDS in the notebook and set MAPPING_CONFIRMED=True.
This is intentionally not guessed: `instruction`, `input`, and `text` can have different roles;
check full records to avoid omitting or duplicating the final input/demonstrations. Field contents
and response are preserved; fields are joined with two newlines. Preview the final text and split
manifest before starting train. `selected_cluster_id` is only a candidate grouping field; inspect it.

The prompt uses the Qwen chat template with `enable_thinking=False`. Only response and terminal EOS
are supervised. This is offline response imitation, not KL distillation with teacher probabilities.
Empty/invalid mapped records, exact prompt-response-group duplicates and overlength records are
excluded with IDs/reasons. Invalid JSON stops the run. Long answers are never silently truncated.
No benchmark contamination audit is implemented yet; this pilot is not a paper-ready data audit.

The common pool is split by normalized source input into train, adaptive probe, independent holdout.
The default dev fraction is 10% of unique inputs, divided between probe and holdout. Training subset
selection is seeded random after splitting. The max-train cap is applied before grouping; train-only
counts determine the most frequent groups and collapse the remainder into `__rare__` (up to 16 groups).
Probe and holdout retain up to four records per represented group. Missing group coverage is recorded.
A holdout changes the author's training budget and must be disclosed. A 200-step pilot uses only
1,600 sample presentations at the default effective batch, not all 5,000 examples or all 50k.

## Method and controls

All modes share LoRA modules, initialization seed, batch order, step count and response-only CE
(mean tokens per example, then mean weighted examples). Sample order is independent of group weights.
Every mode measures probe/holdout on the same schedule to control measurement cost.

- `uniform`: weight one per example; this is our controlled LoRA baseline, not an exact reimplementation
  of the author's full fine-tuning pipeline.
- `balanced`: inverse train group frequency, bounded and normalized.
- `loss`: dynamic weighting from group probe CE relative to its initial value.
- `progress`: dynamic weighting from positive relative decrease in EMA probe CE.

For progress, with current probe CE L_g(t), smoothed CE S_g(t), initial CE L_g(0):

    S_g(t) = ema * S_g(t-1) + (1-ema) * L_g(t)
    score_g = max(0, (S_g(t-1) - S_g(t)) / max(L_g(0), epsilon))
    q = softmax(score / temperature)
    raw_g = (1-mix) + mix * G * q_g
    weight_g = clip(scale * raw_g, floor, ceiling)

Solve scale so the train-frequency-weighted average weight is one. Defaults: EMA .5,
temperature .05, mix .5, bounds [.5, 2]. Groups without probe observations receive score zero;
they keep a nonzero floor, not necessarily weight one. The first probe initializes the controller;
adaptive updates start after the configured warmup. The holdout never updates group weights.
Final checkpoint selection is fixed-step, not selected by holdout/test score.

The current warmup/probe interval is step-based; because all modes traverse exactly the same batches,
these are also identical train-token milestones within a student. Record actual input, response,
probe and holdout tokens. Across different students, verify retained IDs/token counts before claiming
identical token budgets. Adaptive weighting does not guarantee improvement and is not certified novel.

## Outputs and resume

Each mode produces adapter/, metrics.jsonl, summary.json, identity.json, latest.json and full checkpoints.
Checkpoints save adapter plus optimizer, scheduler, GradScaler, Python/NumPy/CPU/CUDA RNG, epoch/cursor,
controller history and token/time counters. Save only at accumulation boundaries. Resume requires the
same original output path, prepared splits, source, total planned steps, dependencies and GPU model.
Restore the entire WORK directory and set the mode's RESUME path to checkpoint-N. Source/config hashes
reject incompatible resumes. Use a checkpoint you produced yourself; state.pt uses trusted PyTorch pickle.
If a write was interrupted, preserve/move the incomplete `.tmp` directory before retrying. Old complete
checkpoints are retained, so monitor disk use or save less frequently on longer runs.

FP16 overflow may skip optimizer updates; logs report successful updates separately from steps.
Report a mismatch between modes and rerun with suitable settings rather than hiding it. The method
is a single-process, single-GPU pilot, including on Kaggle T4 x2. It does not combine their VRAM.

The compact artifacts ZIP includes adapters, metadata, logs, source and split IDs, but not raw/tokenized
corpus, base weights, merged models or optimizer states. For resume, save full Kaggle outputs (Save Version).
Export optionally merges a completed adapter with the pinned base on CPU into a standard model directory.

## Interpreting results

comparison.json shows initial/final held-out teacher-response CE, total tokens, runtime and allocated
peak VRAM. CE is not Accuracy, sentiment F1, or evidence of benchmark improvement. Runtime includes
probes/holdout/checkpoints before the final save; final adapter/archive/export are outside the timed run.
Peak VRAM is torch allocated memory, not full device utilization. Probe overhead must be counted.

SentiBench v2 evaluation is not included in this pilot. Existing repo evaluators use raw prompts and
first-label-token scoring for classification, and have not been audited for Qwen3 chat/non-thinking or
CompEffDist's revised FSA prompts. Do not tune on their test outputs or call these paper-equivalent scores.
After auditing evaluation, compare original and trained checkpoints on the same 12 datasets × 3 demo seeds.
Use uniform/balanced/loss/progress ablations and training seeds before claiming a method contribution.

Rebuild notebook after changes:

    python3 tools/build_train_notebook.py

Local dependency-free checks (no model execution):

    python3 -m unittest discover -s tests -v

Sources: https://arxiv.org/html/2510.24425v1,
https://huggingface.co/datasets/zhang-yice/sentiment-distillation-v2,
https://huggingface.co/Qwen/Qwen3-0.6B,
https://proceedings.mlr.press/v80/chen18a.html.
