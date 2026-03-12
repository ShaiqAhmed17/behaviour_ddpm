# behaviour_ddpm

Master’s project codebase (forked from `puria-radmard/behaviour_ddpm`) exploring diffusion-style models for behaviour/decision-time modelling, with supporting modules for DDM-style baselines, RL components, and a “dynamic observer” set of experiments.

> Status: the repository currently has almost no top-level documentation (the root `README.md` is just a title). This README is a best-effort based on the code/files currently present.

---

## What’s in this repo (high level)

- **`ddpm/`** – Core diffusion-model code + configs + training entry points.
  - Includes a small README with the intended training command:
    - `ddpm.train.multiepoch ddpm/configs/...relevant_config...`
- **`ddm/`** – Drift Diffusion Model-related code and tests/figures.
  - `ddm/ddm.py`, `ddm/ddm_test.py`, `ddm/rt_ddpm.py`, `ddm/ddm_results.png`
- **`drl/`** – Reinforcement learning utilities/agents/envs and training support.
  - Contains its own `setup.py` (suggesting it may be installable as a package).
- **`dynamic_observer/`** – Experimental notebooks/tests for score matching / sampling.
  - Includes scripts like `ct_exact_scorematching_test.py`, `palimpsest_test.py`, plus `test.ipynb`.
- **`hazard_rate_test.py`** – Standalone experiment/training script (PyTorch) for learning hazard-rate functions that match a target response-time distribution.
  - Produces `hazard_rate_test.png`.

There are also figure folders like `neurips_figures/` and `z_mate50_symposium_images/`.

---

## Quickstart

### 1) Clone
```bash
git clone https://github.com/ShaiqAhmed17/behaviour_ddpm.git
cd behaviour_ddpm
