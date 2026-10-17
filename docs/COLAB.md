# Colab notebooks

- [Gemma / Olympic mathematics](../notebooks/01_Gemma_Olympiad_3Seeds.ipynb): rebuilt on 17 October 2026 from Ngoc's main-branch PI protocol, commit `786291e`. See [matched configuration and migration](GEMMA_NGOC_MATCH.md).
- [Phi / Olympiad](../notebooks/02_Phi_Olympiad_3Seeds_Resume.ipynb): unchanged legacy two-arm experiment. It still embeds its original 2,048-candidate preparation and uses the earlier Phi configuration. It is not advertised as matching the updated Ngoc experiment.

## Gemma

Select A100, accept the Gemma license, fill `HF_TOKEN` in the first cell, then Run all. No Secrets or Drive mount is required. Download all output using the final button. The old 2,048-token output is kept separately; it cannot resume the new 1,536-token experiment.

The new notebook uses five arms and seeds 1, 2, 3. It runs the original TRL + vLLM scripts, BF16, 300 steps, PI budget 358, a single 1,536-token cap and the previously frozen 2,000 NuminaMath Olympic training / 2,000 Omni-MATH numeric test questions. Only the configuration follows Ngoc; the data remains Olympic. Qwen scores, noise estimates, anchors and adapters are not reused for Gemma.

## Phi

Select A100-SXM4-40GB and Run all. No model token is required. Phi still uses seeds 42, 43, 44, two arms, the old saved preparation and its recorded PyTorch/software identity. Its old ZIP is embedded; no upload is required to continue that embedded initial snapshot. The final button downloads a full output ZIP. A newer output ZIP is not automatically embedded back into the notebook.

## Output and interruptions

Results remain local until downloaded. Stop the running command, wait for it to finish, run the final cell and download partial output before ending the runtime. A destroyed runtime loses undownloaded progress. The ZIP excludes model weights, cache and credentials, but includes available training checkpoints and results. Clear the token cell before sharing a saved notebook.

Gemma's bucket and base-evaluation scripts reuse completed stage files; an interrupted stage may restart. Anchor selection now caches every 128-candidate batch and can extend the initial pool of 800 to the already frozen maximum of 2,048 if insufficient. The 128-anchor target and >=6/8 threshold remain fixed. See the configuration document for recovery from the initial 66-anchor result. Full training checkpoints are saved every 25 steps; retained milestone adapters alone cannot resume the optimizer.

## Repository

- `notebooks/`: standalone notebook files.
- `src/gemma_ngoc/`: Ngoc scripts with an Olympic input adapter, fixed Olympic question lists, matched plan and orchestration.
- `src/`: legacy Phi sources and the two notebook builders.
- `snapshots/`: original partial output archives, including preserved historical Gemma progress.
- `tests/`: CPU verification.

Rebuild Gemma only: `python src/build_gemma_ngoc.py`.
Rebuild through the shared entry point: `python -c "import sys; sys.path.insert(0, 'src'); from build_colab import build; build('gemma')"`.
Run CPU checks: `python -m pytest tests -q`.

Root requirements are for local tooling; each notebook installs its own runtime. Full A100 execution, wall time and CU consumption have not been measured for the rebuilt Gemma notebook. More compute does not guarantee NAFC superiority.
