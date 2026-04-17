# PCFM: Physics-Constrained Flow Matching

[![arXiv](https://img.shields.io/badge/arXiv-2506.04171-b31b1b.svg)](https://arxiv.org/abs/2506.04171)
&nbsp;•&nbsp;
[Project Page](https://caipengfei.me/pcfm)
&nbsp;•&nbsp;
[Julia Version](https://github.com/utkarsh530/PCFM.jl)

  
[**Utkarsh\***](https://www.linkedin.com/in/utkarsh530/) •
[**Pengfei Cai\***](https://www.linkedin.com/in/pengfei-cai/) •
Alan Edelman •
Rafael Gómez-Bombarelli •
Christopher Rackauckas  

<em>*To appear at <a href="https://neurips.cc/virtual/2025/poster/117071">NeurIPS 2025</a>.*</em>
</div>

This repo implements *Physics-Constrained Flow Matching (PCFM)* -- a framework that enforces physical constraints during sampling of flow-based generative models. 

<img src="assets/pcfm_figure.png" width="67%" alt="PCFM summary figure">

## Burgers Metrics Guide (Full vs Partial)

This repository currently uses two Burgers benchmark scripts with aligned logging schema:

- [test_compare_burgers_full.py](test_compare_burgers_full.py): IC + boundary residuals + nonlinear PDE residual (no mass term).
- [test_compare_burgers_partial.py](test_compare_burgers_partial.py): IC + boundary residuals + mass residual (original PCFM residual for Burgers BC case).

Both scripts log the same metric keys to WandB so results are easy to compare side by side.

### Shared Metric Names

- Physics_Error (MAE): mean absolute value of the main residual used by that experiment.
- Physics_Error_IC: error on initial condition at t=0.
- Physics_Error_BC_Left: left boundary value error (Dirichlet side).
- Physics_Error_BC_Right: right boundary Neumann error computed from boundary gradient residual.
- Physics_Error_BC: average of left and right boundary errors.
- Physics_Error_PDE_Scaled: PDE residual with scaling (used for stable guidance when applicable).
- Physics_Error_PDE_Raw: PDE residual without scaling.
- Physics_Error_Mass: mass residual error.
- Physics_Error_CL: global mass-drift style diagnostic already present in the benchmark.
- Distribution_MMSE, Distribution_SMSE, Distribution_SampleMSE: distribution-level and sample-level data fit metrics.

### What Is Active In Each Experiment

- Full experiment ([test_compare_burgers_full.py](test_compare_burgers_full.py)):
  - Active physics terms: IC, BC, PDE.
  - PDE scaling is applied consistently in both guidance and reported main residual.
  - Physics_Error_Mass is logged as NaN by design (not part of this residual).

- Partial experiment ([test_compare_burgers_partial.py](test_compare_burgers_partial.py)):
  - Active physics terms: IC, BC, Mass.
  - Physics_Error_PDE_Scaled and Physics_Error_PDE_Raw are logged as NaN by design.

### Practical Interpretation Rules

- Compare Physics_Error (MAE) only across runs of the same experiment type (Full with Full, Partial with Partial).
- Use shared boundary metrics (BC_Left, BC_Right, BC) for cross-experiment boundary behavior checks.
- In Full runs, prioritize Physics_Error_PDE_Scaled for optimization-aligned comparisons and keep Physics_Error_PDE_Raw as physical interpretability support.
- In Partial runs, prioritize Physics_Error_Mass as the third physics component.
- Treat NaN component metrics as not-applicable, not as failures.

### Why BC_Right Uses Neumann Residual

For Burgers data generation, the right boundary is Neumann-like (zero gradient), so BC right error is measured via gradient residual at the boundary, not by value matching at the boundary node.


## License

PCFM is released under the **MIT License**.

This repository includes components derived from [amazon-science/ECI-sampling](https://github.com/amazon-science/ECI-sampling) licensed under the **Apache License 2.0**. See [`LICENSE`](./LICENSE), [`LICENSE-APACHE-2.0`](./LICENSE-APACHE-2.0), and [`NOTICE`](./NOTICE) for details.



## Citation

If you use this repository, please cite:

```bibtex
@article{PCFM2025,
  title={Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints},
  author={Utkarsh, Utkarsh and Cai, Pengfei and Edelman, Alan and Gomez-Bombarelli, Rafael and Rackauckas, Christopher Vincent},
  journal={arXiv preprint arXiv:2506.04171},
  year={2025}
}
