# Gemma / Olympic mathematics with Ngoc's PI configuration

The user selected Ngoc's configuration on 17 October 2026. Training and decoding follow Ngoc, but the model is Gemma 3-1B-IT and data remains the frozen NuminaMath-to-Omni-MATH Olympic experiment. Following an initial shortage of 66/128 eligible anchors, the user authorized expansion within the previously frozen candidate bank. This preparation adjustment is recorded below; it is not an identical 800-candidate replication.

## Upstream reference

Repository: https://github.com/HoangLeminh17/DATN-FPTU

Main commit: `786291e`, merged on 16 October 2026. References are notebooks 6a/6b, 7a-7e and `results/20261016-pi-arms/run_info/`. Actual replay/KL/FlipGuard arguments are captured in `tests/fixtures/ngoc_pi_args.json`.

Seven original scripts retain their source text (LF line endings): common, constraints, train, flipcon, evaluate, build_anchors and fixflip. prepare_buckets adds only `olympic` to its dataset CLI choices; its sampling and scoring code is unchanged. run_upstream supplies the frozen Olympic train/test/candidate rows. SHA-256 hashes are checked before execution. The same TRL trainer supplies the optimizer, loss, KL definition, anchor auditing, FlipGuard NLL and on-policy replay behavior.

## Configuration

| Item | Matched value |
|---|---|
| Train / test | Original frozen 2,000 NuminaMath-1.5 Olympiad / 2,000 Omni-MATH numeric question-answer pairs |
| Seeds / arms | 1, 2, 3; monitor, NAFC, fixed replay, KL, FlipGuard |
| Training | 300 steps; 8 prompts x 8 responses; micro-batch 4; Dr. GRPO |
| Response cap | 1,536 throughout; no extra long-cap run |
| PI | B=358; eta=0.05; kp=4; lambda max=3, signed |
| LoRA | r32; alpha 64; dropout 0; all linear layers |
| Optimizer | lr 1e-5; constant with 10-step warmup; BF16 |
| Anchors | Initial pool 800; fixed maximum 2,048 if insufficient; select 128 at >=6/8; one 4-draw recheck |
| Split / audits | Half repair / half held; split seed 0; every 25 steps x 4 samples |
| NAFC | eta 10; lambda max 1; margin 0.02 above Gemma's measured noise; max replacement fraction 0.5 |
| Fixed replay | gamma 0.5 x max fraction 0.5 = 2/8 prompt groups; rotate through repair anchors |
| KL / FlipGuard | beta 0.04 / raw single-sample flags with gamma 0.1 and reference-response NLL |
| Evaluation | Steps 50, 100, 150, 200, 300; 4 draws per question, seed 1 |
| Reference | 8 draws plus independent 4-draw test noise check at one cap |
| Checkpoints | Every 25 steps; retain last 3 full states; milestone adapters at 50-200 |
| Software | vLLM 0.19.1; TRL 1.13.0; Transformers 5.18.0; PEFT 0.21.1; Torch 2.10.0+cu128 |

Math-Verify was not version-pinned in Ngoc's installation cell. This notebook follows that installation and records its resolved version, together with datasets, before preparation. Resume rejects a different recorded environment.

Training data retains the `olympiads_ref` and `olympiads` sources. Test data retains the numeric-answer Omni-MATH subset; it is not the full official benchmark or an exclusively IMO dataset. The exact approved 2,000 train/test partitions are unchanged. The initial pool is the first 800 of the already frozen 2,048 Numina candidates. If insufficient, all remaining candidates are evaluated; selection does not stop midway through expansion as soon as 128 become available. All partitions remain disjoint, and the original freeze excluded exact overlaps against the full Omni-MATH corpus.

The original train-bucket preparation also samples 2,000 x 8 responses. B=358 buckets are computed on CPU by reusing those samples. This is separate from the 2,000 x (8+4) test-reference responses. Both prepare once for all 15 runs. No new validation split or 70%-of-reference budget is introduced.

## Historical output

The preserved ZIP `snapshots/nafc_gemma_math_3seeds_output.zip` passed CRC checking. It contains:

- Complete 64-question validation, 4 responses per question.
- 336 tested anchor candidates, 8 responses each, yielding 128 stable anchors.
- Three completed 64-anchor noise checks, 4 responses each.
- Reference test completed for 208 of 2,000 questions, 8 responses each.
- No prepared bundle and no optimizer training checkpoint.

Those responses use a 2,048-token cap, Hugging Face generation and the old data partition. They are not combined with the 1,536-token vLLM/BF16 experiment. The historical Gemma output used MATH, so it also has the wrong data for this Olympic experiment. The source ZIP is preserved unchanged; its SHA and inspection are recorded in `GEMMA_SAVED_PROGRESS.json`. The new notebook starts matching preparation afresh.

## Notebook behavior

Fill the token cell, select A100 and Run all. A fresh run needs no file upload. A later output ZIP from this new notebook can be restored by enabling `RESTORE_MATCHED_OUTPUT`; old protocol ZIPs are rejected. The final download button also works after an interrupted command once setup has completed.

Bucket and base-evaluation scripts save completed stages, so an interrupted stage may restart. Anchor selection now saves every 128-candidate batch. Training resumes only from a complete model/optimizer/scheduler/RNG/PI/NAFC checkpoint. No fallback to CPU, smaller counts or shorter evaluation is automatic.

The full experiment has 15 training runs. No claim is made that it fits one Colab session or a specified CU budget. All planned seeds and milestones must be reported. Qwen-specific empirical variance constants and numerical success thresholds are not reused for Gemma; descriptive question-bootstrap comparisons are explicitly conditional on each seed. The report uses upstream fixflip definitions rather than selecting favorable runs.

Only Gemma is changed. Phi's notebook and saved preparation remain as before. GPU compatibility and speed still require a real A100 execution.

## Process supervision fix (18 October 2026)

The old runner waited for pipe EOF before checking the upstream process status. An orphan vLLM worker could keep
that pipe open after its parent exited, blocking the next stage indefinitely. The updated runner redirects output
to a regular log file, polls the parent status, and cleans up the job's process group immediately after exit. On
interruption or timeout it also collects descendant sessions before stopping them, so nested jobs are not left
running. It prints a heartbeat every minute and stops a stage after 15 minutes without new output. That silence
limit can stop legitimate long silent work; completed files and checkpoints are preserved for inspection/resume.

Setup performs a CPU-only Linux check with a synthetic worker holding stdout after parent exit, before any model
preparation/training. This check does not download a model or perform an optimizer update. It cannot guarantee
against unrelated CUDA, driver, resource, or platform failures.

ZIPs and identities from the previous matched Olympic runner are accepted only when all experiment settings,
data hashes, upstream source hashes and recorded software/GPU still match. The sole permitted source difference
is the known previous runner hash. Historical MATH/2,048-token output remains incompatible.

For an already-running old notebook, upload `src/colab_process_guard.py` into `/content` and run
`python -u /content/colab_process_guard.py` in the Colab Terminal. Keep that terminal command running. It watches
only child sessions owned by `/content/gemma_ngoc/matched_run.py`, releases orphan groups after their leaders exit,
and stops owned jobs after 15 minutes without log progress. It never deletes output. There is no need to interrupt
an active evaluation just to start the helper. The helper exits after 60 seconds without any Gemma job; restart it
if using the old notebook again later. It does not modify the old notebook or its experiment fingerprints.

## Recovering the 66-anchor preparation

`src/install_anchor_extension.py` upgrades an idle existing runtime from the two known 800-pool runners. It verifies
all sources before writing, rejects active jobs or pool changes after training has started, and preserves bucket,
base-evaluation and anchor files. The new frozen-data hash records the complete pre-existing candidate bank;
train/test questions are unchanged. A saved old Olympic ZIP can also be restored into the updated notebook.

The extension reuses every qualified item when the old file contains all eligible items from its initial pool.
These items become part of the candidate set; final selection still uses the upstream seeded shuffle, so not all
66 must appear in the final 128. Only the remaining 1,248 candidates require new selection samples. Each completed
128-candidate batch is saved under `results/anchor_extension/selection.json`. Final selected anchors are calibrated
again with four independent responses. Initial metadata and source plan are retained under that results folder.

If the full 2,048-candidate bank is still insufficient, the run stops with the complete selection cache preserved.
It does not lower the success threshold, use train/test questions as anchors, or silently reduce the denominator.
Updated output ZIPs require the updated notebook; do not rerun the old embedded source cells after upgrading.
