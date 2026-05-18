# Energy-Guided Transport for Projection-Free Physics-Informed Flow Matching

This repository contains the official code for the paper **Energy-Guided Transport for Projection-Free Physics-Informed Flow Matching**.

This project is built on top of the original **PCFM (Physics-Constrained Flow Matching)** repository. We have extended their codebase to implement our novel zero-shot framework (EGT), which formulates non-linear physical constraints as a continuous energy landscape. This approach ensures stable and efficient generation without the structural disruptions caused by rigid algebraic projections.

---

## 1. Virtual Environment Setup

We recommend using `conda` to manage dependencies. You can create and activate the dedicated `egt` virtual environment with the following commands:

```bash
conda create -n egt python=3.10 -y
conda activate egt
pip install -r requirements.txt

```

If the environment already exists, simply activate it:

```bash
conda activate egt

```

## 2. Generating Synthetic Datasets

The repository includes scripts to generate the HDF5 datasets required for training and evaluation. By default, the files will be saved in the `datasets/data/` directory.

* **Burgers 1D:**
```bash
python datasets/generate_burgers1d_data.py

```


* **Reaction-Diffusion (RD) 1D:**
```bash
python datasets/generate_RD1d_data.py

```


* **Navier-Stokes 2D:**
This script exposes CLI arguments to customize the dataset size and resolution.
```bash
python datasets/generate_ns_2d.py --root datasets/data --nw 100 --nf 100 --s 64 --t 49 --steps 50 --mu 1e-3

```


* **Heat 1D:**
The Heat equation dataset is generated procedurally at runtime via the dataset class, so there is no separate HDF5 generator. Just use the provided configurations directly (`configs/heat_white.yml` or `configs/heat_smooth.yml`).

## 3. Training the Base Model

Pretraining for the base model (unguided Functional Flow Matching) is handled by the main script in `scripts/training/`. You will need to specify the configuration file corresponding to the physical problem and the type of prior (White Noise or Smooth Kernel) you want to use.

**Basic Command:**

```bash
python scripts/training/main.py configs/<config>.yml --mode train --device cuda:0 --logdir logs --savename <save_name>

```

**Useful Training Flags:**

* `--mode train`: Starts the process in training mode.
* `--device cuda:0`: Selects the GPU device.
* `--logdir` and `--savename`: Defines where TensorBoard logs and model checkpoints (e.g., `latest.pt`) are saved.
* `--resume <checkpoint>`: (Optional) Resumes training from a previously saved checkpoint.

## 4. Inference and Testing

The scripts for inference and benchmarking are located in the `test/` directory. These scripts allow you to apply zero-shot physical guidance (including our EGT framework and various baselines like PCFM, ECI, etc.) to the test set.

**General Command Structure:**

```bash
python test/test_compare_<problem>_<mode>.py --config_path <config.yml> --ckpt_path <checkpoint> [arguments...]

```

**Key Inference Flags:**

* `--models`: Chooses which samplers to run. Use `all` to benchmark all supported methods, or `ours` to run only the proposed EGT method.
* `--n_samples`: Number of samples to generate for metric calculation.
* `--n_steps`: Number of integration steps for the ODE solver.
* `--gamma_list`: Sweeps different guidance strengths (e.g., `--gamma_list 0.1 1.0 2.0`).
* `--device`: Specifies the GPU for inference.

### Example Usage

**Test on Burgers (Full PDE Residual) comparing all models:**

```bash
python test/test_compare_burgers_full.py \
    --config_path configs/burgers1d_white.yml \
    --ckpt_path logs/burgers1d_white/latest.pt \
    --n_samples 100 \
    --models all \
    --device cuda:0

```

**Test on Navier-Stokes 2D running ONLY the proposed method:**

```bash
python test/test_compare_ns_full.py \
    --config_path configs/ns_lightning_white.yml \
    --ckpt_path logs/ns_lightning_white/latest.pt \
    --n_samples 100 \
    --models ours \
    --device cuda:0

```

The scripts will automatically log the results (physical errors, distributional fidelity, etc.) to WandB and output a summary report.

---

## License and Acknowledgements

This repository combines original code with components adapted from the [amazon-science/ECI-sampling](https://github.com/amazon-science/ECI-sampling) project and is fundamentally built upon the **PCFM** architecture.

* [`LICENSE`](https://www.google.com/search?q=./LICENSE): Primary license for the code developed in this repository.
* [`LICENSE-APACHE-2.0`](https://www.google.com/search?q=./LICENSE-APACHE-2.0): Applies to third-party portions derived from ECI-Sampling.

## Citation

If you use this repository or find our work helpful, please cite the original PCFM paper upon which this framework is built, alongside our work:

```bibtex
@article{PCFM2025,
  title={Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints},
  author={Utkarsh, Utkarsh and Cai, Pengfei and Edelman, Alan and Gomez-Bombarelli, Rafael and Rackauckas, Christopher Vincent},
  journal={arXiv preprint arXiv:2506.04171},
  year={2025}
}

@article{EGT2026,
  title={Energy-Guided Transport for Projection-Free Physics-Informed Flow Matching},
  author={Anonymous},
  journal={Submitted to NeurIPS 2026},
  year={2026}
}

```

```

```