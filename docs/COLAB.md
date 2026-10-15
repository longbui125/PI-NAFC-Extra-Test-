# Two Colab notebooks: resume saved preparation

| Notebook | Model / task | Training seeds | Embedded saved validation |
|---|---|---|---|
| [`01_Gemma_MATH_3Seeds_Resume.ipynb`](../notebooks/01_Gemma_MATH_3Seeds_Resume.ipynb) | Gemma 3-1B-IT / MATH | 42, 43, 44 | 16/64 questions |
| [`02_Phi_Olympiad_3Seeds_Resume.ipynb`](../notebooks/02_Phi_Olympiad_3Seeds_Resume.ipynb) | Phi-4-mini-instruct / Numina to Omni-MATH | 42, 43, 44 | 32/64 questions |

These replace the earlier three-notebook layout. Both are standalone: experiment sources, frozen data and saved preparation are embedded. They can start independently on two Colab runtimes. No repository, ZIP picker, Drive login, Secrets or intermediate file transfer is needed.

## Run on fresh Colab

1. Import the two new notebooks into separate Colab notebooks and select **NVIDIA A100-SXM4-40GB**.
2. In Gemma only, accept the [model license](https://huggingface.co/google/gemma-3-1b-it) and paste an authorized [Hugging Face read token](https://huggingface.co/settings/tokens) into `HF_TOKEN = ""` in section 1 **before Run all**. Phi needs no token. This field is notebook source: remove your real token before sharing the notebook. Output ZIPs do not include it.
3. Select **Run all** in each notebook. Sources are written, exact saved dependencies are installed, progress is restored and cell 5 continues preparation automatically. All three paired seeds then train sequentially, followed by the complete report.
4. At the end, click **Download all output (.zip)** and wait for the browser download to finish. There are no automatic remote uploads.

The embedded saved outputs are read from `snapshots/` when `src/build_colab.py` builds the notebooks. The two old Phi ZIPs contain identical files and response records. One copy is embedded; 32 saved validation questions remain 32, not 64. No anchor selection, noise calibration, full reference test or optimizer training had finished in these snapshots.

On a fresh runtime, validation continues from the saved chunks. Model loading, prompt checks and the backward preflight repeat; completed validation responses are reused. Re-running setup in a runtime that has made newer progress does not overwrite it with the initial embedded snapshot.

## Exact resume compatibility

Source fingerprints are checked against each saved archive. Gemma retains the generation-compilation fix. Phi embeds its original, pre-fix loader exactly: the later Gemma-only source edit is not silently substituted into a frozen Phi run. Model weights/configuration remain pinned, and PI/NAFC equations are unchanged.

The recorded environment is PyTorch **2.11.0+cu130**, Transformers **4.52.4**, PEFT **0.15.2**, Accelerate **1.8.1**, Math-Verify **0.8.0**, NumPy **2.1.3**, Matplotlib **3.10.0**, and **NVIDIA A100-SXM4-40GB**. Non-PyTorch libraries are installed at these versions automatically. A mismatched CUDA PyTorch build stops before preparation rather than silently changing the experiment. Existing compatible CUDA PyTorch is retained; no untested GPU-library downgrade is attempted.

All embedded source hashes, software versions, configuration and archive SHA-256 are checked. An incompatible environment or damaged snapshot stops execution with an error. The notebooks contain no private model weights; those must download again in a new runtime. Gemma still requires valid access credentials.

## Preserved experimental design

Both models retain seeds 42-44 and two arms per seed: **GRPO + PI + monitor** versus **GRPO + PI + NAFC**. There are twelve training runs overall. Each model prepares one reference/anchor/noise bundle shared across its three training seeds. No seed starts from another seed's adapter.

| Setting | Gemma / MATH | Phi / Numina to Omni-MATH |
|---|---|---|
| Train / validation / test questions | 2,000 / 64 / 2,000 | Same |
| Steps / rollouts | 300 per arm; 8 prompts x 8 responses/step | Same |
| Primary / secondary response cap | 2,048 / 4,096 | 4,096 / 8,192 |
| Generation batch | 16 | 8 |
| Prompt cap | 4,096; reject overflow | Same |
| Primary tests | Steps 50, 100, 150, 200, 300; four responses/question | Same |
| Secondary test | Final step only; four responses/question | Same |
| Reference tests | Eight responses plus independent four-response noise check, at both caps | Same |
| Anchors | Exactly 128 from 2,048 candidates; initial success >=6/8 | Same |
| Repair / held controller split | 64 / 64 | Same |
| Calibration | Three independent four-response held-anchor audits | Same |
| Online audits | Four responses on every anchor every 25 steps | Same |
| Training | LoRA r32, all linear layers; lr 1e-5; fp16 | Same |

PI targets 70% of the frozen reference validation mean length. NAFC retains the original noise allowance, adaptive coefficient and fresh on-policy replay of flagged repair anchors. Replay replaces task groups within the same planned rollout budget; task/anchor exposure is reported. The control receives the same audit schedule. No supervised anchor NLL, distillation loss or extra KL is added. There is no separate fixed-replay comparator in these notebooks.

All prespecified seeds, milestones and capped/unboxed responses remain in the report. If the candidate pool cannot supply all 128 anchors, preparation is marked inconclusive, training is skipped and the diagnostic remains downloadable. No denominator is silently reduced. Reports require all three paired seeds.

Gemma uses all seven MATH topics and answer types; two source training rows with empty gold are logged/excluded before sampling. Phi uses frozen Numina `olympiads_ref`/`olympiads` numeric-answer problems and a random numeric-answer Omni-MATH subset. This is not the full official Omni-MATH score or exclusively IMO-level material. Normalized exact deduplication prevents direct partition overlap; paraphrase and pretraining contamination remain unverified.

Compare each model's matched arms separately. Because models, data and caps differ, cross-experiment differences do not isolate architecture effects. Three training seeds provide limited evidence about randomness. Question-bootstrap intervals are conditional on these seeds, pointwise and not corrected for multiple comparisons. More compute does not guarantee NAFC will win.

## Output and interruptions

Final archives are `nafc_gemma_math_3seeds_output.zip` and `nafc_phi_olympic_3seeds_output.zip`. They include all available shared preparation, seed/arm results, milestone adapters, optimizer, scheduler, scaler, PI/NAFC state, evaluation records and aggregate reports. Model cache and credentials are excluded.

The runtime is the only location of new progress until you download it. Unexpected runtime deletion loses unexported progress. For a planned interruption, stop the running process, execute the last cell and download partial output before ending the runtime. The final button works even when preparation or training stopped early. A downloaded later ZIP is newer than the initial snapshot embedded in these notebooks: retain it for subsequent resume rather than assuming this initial notebook has updated itself.

Creating the final full ZIP needs extra local disk space and can take several minutes. Browser download and live GPU execution have not been validated locally.

## CU budget

Merging Phi into one notebook eliminates the duplicate Phi preparation. It does **not** halve the total training work: both models still have six arms, and Phi's six arms now run sequentially on one GPU. Wall time can increase relative to two Phi GPUs even as redundant preparation cost falls.

The full two-model protocol has at most **1,010,944 sampled responses** including worst-case anchor selection, reference/noise tests, training, audits and all evaluation milestones. The embedded snapshots already contain 64 Gemma and 128 Phi responses; these are reused once. Response lengths and training/backward cost determine runtime, so counts do not establish a GPU-hour estimate. 500 CU or one Colab session is not guaranteed sufficient. Measure actual progress and CU usage; do not select favourable seeds or cut evaluation after seeing results to fit a budget.

## Repository layout

- `notebooks/`: two standalone Colab notebooks.
- `src/`: experiment code, frozen configurations and data partitions.
- `snapshots/`: original partial preparation archives used to build the notebooks.
- `tests/`: CPU checks; no GPU training.
- `docs/`: execution and protocol notes.

## Rebuild and verification

`src/build_colab.py` reads the three original ZIPs under `snapshots/`, verifies duplicated Phi progress and reconstructs source bodies that match the saved fingerprints. It builds only these two notebooks. Do not remove or alter those saved ZIPs. `config.json` and the legacy five-seed/three-arm protocol are separate and unchanged.

CPU checks cover actual archive restoration, completed validation counts, exact embedded source fingerprints, full three-seed schedules, corruption rejection, preservation of newer local progress and complete checkpoint export. Full A100 runs and browser downloads remain untested.

From the repository root, rebuild with `python src/build_colab.py` and run CPU checks with `python -m pytest tests -q`. Root requirements support local tooling; each notebook installs its recorded Colab dependencies. The optional upstream AST check requires a sibling checkout of the pinned upstream repository; the bundled core checksum is always checked.
