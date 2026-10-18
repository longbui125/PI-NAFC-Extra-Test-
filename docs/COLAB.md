# Colab notebooks

- [Gemma / Olympic mathematics](../notebooks/01_Gemma_Olympiad_3Seeds.ipynb)
- [Phi-4-mini-instruct / Olympic mathematics](../notebooks/02_Phi_Olympiad_3Seeds_Resume.ipynb)

Both standalone notebooks use Ngoc's PI protocol from DATN-FPTU commit `786291e`: five arms, seeds 1/2/3,
300 steps, BF16, LoRA r32, PI budget 358 and one 1,536-token response cap. The five arms are PI monitor,
PI + NAFC, fixed replay, KL and FlipGuard. Each model runs 15 training jobs sequentially.
After the A100-40GB OOM and the user's request to speed up the restart, Gemma uses micro-batch 2 x accumulation
32; Phi stays at 1 x 64. Both retain the effective 8 x 8 response batch. Physical batching differs from Ngoc's
4 x 16; numerical and anchor-rotation order can differ. Micro-batch 2 still needs real-GPU memory validation.

The same frozen 2,000 NuminaMath Olympic training questions, 2,000 numeric Omni-MATH test questions and
2,048 disjoint anchor candidates are embedded. These are question lists, not model predictions. No ZIP upload
or repository checkout is needed for a fresh run. See [protocol details](GEMMA_NGOC_MATCH.md).

## Run

Import the corresponding notebook into Colab and select A100. For Gemma, accept its Hugging Face license
and fill the first token cell. Gemma also requests permission to mount the running account's Google Drive.
For the current Gemma restart, enable `RESTORE_MATCHED_OUTPUT` and upload the saved matched preparation-only ZIP
when prompted; it contains 128 anchors and complete train/test/base preparation, but no trained checkpoints.
Phi is public and needs no token. Leave `RESTORE_MATCHED_OUTPUT=False` for a genuinely fresh run, then Run all. Setup removes unused torchao and tests a tiny CPU LoRA forward/backward pass.
Both notebooks include the text-tokenizer and process-supervision fixes. Each notebook is independent.

Phi model revision is `cfbefacb99257ffa30c83adab238a50856ac3083`. Its pinned config declares Phi3ForCausalLM,
bfloat16 weights and no quantization. Optimizer, loss and sampling flags match Gemma. Phi additionally sets vLLM's training context to the longest
actual task/anchor prompt plus 1,536 completion tokens, rather than the native 131,072-token window. Prompts
are not truncated, and the 1,536-token response cap is unchanged. Architecture and tokenizer differ.
No A100 training run, peak VRAM or CU consumption has been measured for the updated Phi notebook.

## What can be reused

Only the question lists, answers and split boundaries are shared between models. Each model must generate its
own bucket samples and difficulty groups, base evaluations, anchors, reference responses/token IDs and noise floor.
A Gemma output ZIP cannot resume Phi, and the notebook rejects it before importing results. The previous Phi
4,096-token experiment also cannot resume the new 1,536-token protocol. Historical ZIPs remain in snapshots.
Neither model reuses the other's adapters, checkpoints, scores or learned controller state.

The optional restore switch accepts only a compatible output ZIP from the same matched model notebook.
Previous micro-batch-1/4 preparation-only ZIPs remain valid for Gemma. Training state is not mixed across batching
configurations. The OOM recovery installer can preserve scratch metadata and restart a failed uncheckpointed arm.
The current filename still contains `Resume`, but a fresh Phi run does not embed or resume historical preparation.

## Output and interruptions

Gemma writes experiment sources, preparation, checkpoints, evaluation results and logs directly under
`/content/drive/MyDrive/NAFC_Gemma_Olympiad_micro2`. This is My Drive of the account authorized in the running
Colab, not an externally shared folder. It needs sufficient free Drive storage. Model downloads stay under
`/content/gemma_ngoc_model`, the Hugging Face cache under `/content/hf_cache`, and temporary merged models
under `/content/merged`; they are redownloaded/rebuilt after a VM replacement and are not backed up to Drive.
Direct Drive writes can add I/O overhead, especially at checkpoint saves.

After a VM replacement, reopen this same notebook, provide the token, authorize the same Drive and Run all.
Existing Drive preparation/results are kept; an old ZIP is not imported over existing experiment identity.
The training loop skips completed jobs and resumes from the latest complete checkpoint. If the VM dies while
saving a checkpoint, the runner skips incomplete checkpoints and tries earlier complete ones. It cannot recover
steps since the last valid save. Pinned software/GPU and experiment identity still must match; unexpected changes
stop rather than silently restart. Do not run two Gemma runtimes writing to this same Drive folder concurrently.

The final button creates an additional local ZIP including partial preparation, logs, response texts and training
checkpoints; model downloads/cache, token and temporary merged models are excluded. Phi remains local-only and
still needs manual ZIP downloads before its runtime is destroyed. No Secrets or rclone is used.

Completed bucket/base-evaluation stages are reused. An interrupted stage may restart. Anchor selection saves
each 128-candidate batch; if the first 800 candidates are insufficient it checks the remaining pre-frozen bank.
The target remains 128 anchors with at least 6/8 correct. Insufficient eligible anchors stop training without
lowering the target or changing test data. Training checkpoints are saved every 25 steps; milestone adapters
alone cannot resume optimizer state. The supervisor stops a command after 15 minutes without new output.

## Build and checks

`python src/build_colab.py` rebuilds both notebooks from the shared matched sources in `src/gemma_ngoc/`.
Phi changes the pinned model, runtime/model-folder names and the measured prompt-based vLLM context.
Training algorithms, controllers, evaluation scripts and frozen question bytes remain shared. The builder computes and embeds the resulting source fingerprints.
Legacy engine sources and saved ZIPs remain available for provenance, but are not used by the updated notebooks.

Run CPU checks with `python -m pytest tests -q`. Successful CPU checks do not establish GPU memory fit,
completion time, an affordable CU budget or NAFC superiority.
