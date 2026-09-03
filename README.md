# behaviour_ddpm

**Masters project**: diffusion-based generative modeling for behavioural and dynamical time-series, with an emphasis on methodological development, evaluation, and applications to reinforcement learning and dynamical system diagnostics.

## Project objective
This repository implements and evaluates diffusion-driven generative models (DDPM-style) and continuous-time score-matching methods for modelling time-series behaviour. The principal goals are:
- To develop and validate diffusion-based models capable of simulating realistic behavioural/dynamical trajectories.
- To investigate recovery and diagnostic methods for dynamical systems using generative and observer-based techniques.
- To integrate the generative models with reinforcement-learning experiments for downstream evaluation and analysis.

## Repository structure
```
ddpm/                    Core diffusion model code, training utilities, model definitions, configs
ddm/                     Dynamical DDM experiment scripts, simulation utilities and plotting (ddm.py, ddm_test.py)
drl/                     Reinforcement-learning experiments, agent/env scaffolds and config-driven runners (drl/setup.py)
dynamic_observer/       Continuous-time score matching and sampling experiments, tests, and a notebook
neurips_figures/         Figure assets for papers and presentations
z_mate50_symposium_images/  Presentation images
results_link_*           Symlinks to experiment output directories used during development
README.md                This file
hazard_rate_test.py      Top-level example / utility script
```

### How the pieces interact
- The DDPM modules define diffusion schedules, time embeddings and neural architectures; training and sampling are driven by YAML configurations under `ddpm/configs/`.
- The DDM components provide simulation and plotting utilities used to generate experimental figures and diagnostics.
- The DRL experiments evaluate trained or sampled behaviours within agent-environment setups; runners are driven by YAML configuration files.
- The dynamic observer folder contains continuous-time score-matching implementations, test suites and example notebooks used to validate methodological claims.

## Quickstart (minimal)
The repository uses configuration-driven runners. The examples below illustrate the common entry points used during development.

- Train or evaluate a DDPM model (see `ddpm/README.md` for example configs):
```bash
# example (replace <config> with an actual YAML config path)
ddpm.train.multiepoch ddpm/configs/<config>.yaml
```

- Run a DRL experiment (config path is required):
```bash
python drl/setup.py path/to/config.yaml
```

- Execute DDM examples / tests:
```bash
python ddm/ddm_test.py
python ddm/rt_ddpm.py
```

- Open and run the dynamic observer notebook or run its tests:
```bash
jupyter notebook dynamic_observer/test.ipynb
python dynamic_observer/ct_exact_scorematching_test.py
```

Notes:
- Most runners expect YAML configuration files and write model checkpoints and results to configured output directories. The repository contains development-time symlinks (`results_link_*`) that point to experiment outputs used in scripts and analyses.
- Adjust configuration paths (e.g., `save_base`, checkpoint paths) before running experiments.

## Dependencies
Typical dependencies used in the project include (representative):
- Python 3.8 or later
- PyTorch (CUDA optional)
- numpy, matplotlib, tqdm, pyyaml
- Additional utilities referenced by some scripts (for example: `purias_utils` used by certain DRL utilities)

Install basic dependencies (example):
```bash
pip install "torch" numpy matplotlib tqdm pyyaml
```

For reproducible environments, provide an environment specification (requirements.txt or environment.yml) appropriate for your platform and CUDA configuration.

## Development and conventions
- Configuration-driven: experiments and evaluations are run via YAML configuration files. See `ddpm/configs/` and `drl/configs/` for examples.
- Checkpoints and results: model state and results are written to configured output directories. Update configuration values to point to appropriate paths in your environment.
- Tests and examples: small test scripts and notebooks are provided under `ddm/` and `dynamic_observer/` for quick verification of functionality.

## Files of interest
- ddpm/: model definitions, training scripts, and configuration files
- ddm/ddm.py, ddm/ddm_test.py: simulation workflows and plotting utilities
- drl/setup.py: configuration-driven RL experiment runner
- dynamic_observer/: continuous-time score-matching and sampling scripts + notebook

## License and contact
- No license file is included in this repository. Add an explicit LICENSE file before distributing the code publicly.
- For questions or collaboration, contact the repository owner via the associated GitHub profile.
