# Energy-Guided Transport for Projection-Free Physics-Informed Flow Matching

This repository is a local adaptation built on top of the original PCFM work by Cai et al. It documents the training, dataset generation, and evaluation workflow used here.

## Quick Start

### 1) Virtual Environment Setup

The project has been tested with the `egt` conda environment.

```bash
conda create -n egt python=3.10 -y
conda activate egt
pip install -r requirements.txt
```

If the environment already exists, just activate it:

```bash
conda activate egt
```

### 2) Generate Synthetic Datasets

The repository includes scripts that generate the HDF5 datasets used by training and evaluation.
By default, they write to `datasets/data/`.

#### Burgers 1D

Run the Burgers generator to create the default train, test, and sampling datasets:

```bash
python datasets/generate_burgers1d_data.py
```

This creates files such as:

- `datasets/data/burgers_train_nIC80_nBC80.h5`
- `datasets/data/burgers_test_nIC30_nBC30.h5`
- `datasets/data/burgers_sampling_diffICs_nIC20_nBC512.h5`
- `datasets/data/burgers_sampling_diffBCs_nBC20_nIC512.h5`

#### Reaction-Diffusion 1D

Run the RD generator to create the default train, test, and sampling datasets:

```bash
python datasets/generate_RD1d_data.py
```

This creates files such as:

- `datasets/data/RD_neumann_train_nIC80_nBC80.h5`
- `datasets/data/RD_neumann_test_nIC30_nBC30.h5`
- `datasets/data/RD_sampling_diffICs_nIC20_nBC512.h5`

#### Navier-Stokes 2D

The Navier-Stokes generator exposes CLI arguments for the dataset size and output directory:

```bash
python datasets/generate_ns_2d.py --root datasets/data --nw 100 --nf 100 --s 64 --t 49 --steps 50 --mu 1e-3
```

This produces files named like:

- `datasets/data/ns_nw100_nf100_s64_t50_mu0.001.h5`

#### Heat

The heat dataset is generated procedurally by the dataset class, so there is no separate HDF5 generator.
Use the heat configs directly:

- `configs/heat_white.yml`
- `configs/heat_smooth.yml`

### 3) Train the Base Model

Base pretraining is handled by the training script under `scripts/training/`.
Choose the config that matches the dataset you want to train on:

- `configs/burgers1d_white.yml`
- `configs/burgers1d_smooth.yml`
- `configs/heat_white.yml`
- `configs/heat_smooth.yml`
- `configs/ns_lightning_white.yml`
- `configs/ns_lightning_smooth.yml`
- `configs/rd1d_white.yml`
- `configs/rd1d_smooth.yml`

Example: train the Burgers base model.

```bash
python scripts/training/main.py configs/burgers1d_white.yml --mode train --device cuda:0 --logdir logs --savename burgers1d_white
```

Useful flags:

- `--mode train`: training mode.
- `--resume <checkpoint>`: resume from a saved checkpoint.
- `--device cuda:0`: select the GPU device.
- `--logdir logs --savename <name>`: choose where checkpoints and TensorBoard logs are written.

The script saves checkpoints in the selected log directory, including `latest.pt` and periodic snapshots.

### 4) Inference with the Test Files

The benchmark/inference scripts live in `test/` and compare sampling methods on the test split.
They all share the same pattern:

```bash
python test/test_compare_<problem>_<mode>.py --config_path <config.yml> --ckpt_path <checkpoint> --n_samples 100 --models all --device cuda:0
```

Common examples:

```bash
# Burgers, full residual
python test/test_compare_burgers_full.py --config_path configs/burgers1d_white.yml --ckpt_path logs/burgers1d_white/latest.pt --n_samples 100 --models all --device cuda:0

# Burgers, partial residual
python test/test_compare_burgers_partial.py --config_path configs/burgers1d_white.yml --ckpt_path logs/burgers1d_white/latest.pt --n_samples 100 --models all --device cuda:0

# Heat
python test/test_compare_heat_full.py --config_path configs/heat_white.yml --ckpt_path logs/heat_white/latest.pt --n_samples 100 --models all --device cuda:0

# Navier-Stokes
python test/test_compare_ns_full.py --config_path configs/ns_lightning_white.yml --ckpt_path logs/ns_lightning_white/latest.pt --n_samples 100 --models all --device cuda:0

# Reaction-diffusion
python test/test_compare_rd_full.py --config_path configs/rd1d_white.yml --ckpt_path logs/rd1d_white/latest.pt --n_samples 100 --models all --device cuda:0
```

Useful flags for inference:

- `--models all`: run every supported sampler.
- `--models ours`: run only the PCFM-style method.
- `--gamma_list 0.1 1.0 2.0`: sweep guidance strengths for the custom method.
- `--n_steps 100`: set the number of sampling steps.
- `--run_name <name>`: label the WandB run.
- `--device cuda:0`: select the evaluation device.

The scripts log metrics to WandB and write summaries for physics error, distribution fit, and sample-wise MSE.

## Burgers Metrics Guide (Full vs Partial)

This repository uses two Burgers benchmark scripts with aligned logging schema:

- `test/test_compare_burgers_full.py`: IC + boundary residuals + nonlinear PDE residual (no mass term).
- `test/test_compare_burgers_partial.py`: IC + boundary residuals + mass residual (original PCFM residual for Burgers BC case).

Both scripts log the same metric keys to WandB so results are easy to compare side by side.

### Shared Metric Names

- `Physics_Error` (MAE): mean absolute value of the main residual used by that experiment.
- `Physics_Error_IC`: error on initial condition at `t=0`.
- `Physics_Error_BC_Left`: left boundary value error (Dirichlet side).
- `Physics_Error_BC_Right`: right boundary Neumann error computed from boundary gradient residual.
- `Physics_Error_BC`: average of left and right boundary errors.
- `Physics_Error_PDE_Scaled`: PDE residual with scaling (used for stable guidance when applicable).
- `Physics_Error_PDE_Raw`: PDE residual without scaling.
- `Physics_Error_Mass`: mass residual error.
- `Physics_Error_CL`: global mass-drift style diagnostic already present in the benchmark.
- `Distribution_MMSE`, `Distribution_SMSE`, `Distribution_SampleMSE`: distribution-level and sample-level data fit metrics.

### What Is Active In Each Experiment

- Full experiment (`test/test_compare_burgers_full.py`):
  - Active physics terms: IC, BC, PDE.
  - PDE scaling is applied consistently in both guidance and reported main residual.
  - `Physics_Error_Mass` is logged as NaN by design (not part of this residual).

- Partial experiment (`test/test_compare_burgers_partial.py`):
  - Active physics terms: IC, BC, Mass.
  - `Physics_Error_PDE_Scaled` and `Physics_Error_PDE_Raw` are logged as NaN by design.

### Practical Interpretation Rules

- Compare `Physics_Error` only across runs of the same experiment type (Full with Full, Partial with Partial).
- Use shared boundary metrics (`BC_Left`, `BC_Right`, `BC`) for cross-experiment boundary behavior checks.
- In Full runs, prioritize `Physics_Error_PDE_Scaled` for optimization-aligned comparisons and keep `Physics_Error_PDE_Raw` as physical interpretability support.
- In Partial runs, prioritize `Physics_Error_Mass` as the third physics component.
- Treat NaN component metrics as not applicable, not as failures.

### Why `BC_Right` Uses a Neumann Residual

For Burgers data generation, the right boundary is Neumann-like (zero gradient), so the right boundary error is measured via a gradient residual at the boundary, not by value matching at the boundary node.

## Project Structure

- `configs/`: experiment configurations.
- `datasets/`: dataset loaders and generation scripts.
- `models/`: flow models and constraint helpers.
- `pcfm/`: PCFM samplers and projection utilities.
- `scripts/training/`: base training and helper utilities.
- `test/`: inference and benchmark scripts.
- `results/`: generated outputs and comparisons.

## License

This repository contains code under two licenses because it combines original project files with components adapted from [amazon-science/ECI-sampling](https://github.com/amazon-science/ECI-sampling).

- [`LICENSE`](./LICENSE) applies to the code written for this repository and is the main license for the local project code.
- [`LICENSE-APACHE-2.0`](./LICENSE-APACHE-2.0) applies to the third-party portions derived from ECI-Sampling and keeps the original Apache 2.0 attribution requirements intact.

See [`NOTICE`](./NOTICE) for the attribution summary and the list of adapted files.

## Citation

If you use this repository, please cite:

```bibtex
@article{PCFM2025,
  title={Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints},
  author={Utkarsh, Utkarsh and Cai, Pengfei and Edelman, Alan and Gomez-Bombarelli, Rafael and Rackauckas, Christopher Vincent},
  journal={arXiv preprint arXiv:2506.04171},
  year={2025}
}
```
