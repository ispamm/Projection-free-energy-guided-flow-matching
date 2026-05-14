import argparse
import csv
import math
import os
import random
import re
import sys
from collections import OrderedDict

import numpy as np
import torch

try:
    import h5py
except ImportError as exc:
    raise ImportError(
        "h5py is required to read Burgers/RD/NS HDF5 datasets. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc

# Ensure local repo modules are imported instead of similarly named site-packages.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets import get_dataset
from pcfm.constraints import Residuals, Residuals2D
from physics_engine import (
    BurgersEquationResidualsFullPDE,
    HeatEquationResidualsFull,
    HeatEquationResidualsFullPDE,
    NavierStokesResidualsFullPDE,
    ReactionDiffusionResidualsFull,
)
from scripts.training.utils import load_config


DEFAULT_CONFIGS = OrderedDict(
    [
        ("heat", "configs/heat_white.yml"),
        ("burgers", "configs/burgers1d_white.yml"),
        ("rd", "configs/rd1d_white.yml"),
        ("ns", "configs/ns_lightning_white.yml"),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ground-truth residual floors on datasets using the same "
            "partial/full residual definitions from compare scripts."
        )
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument(
        "--max_samples_per_dataset",
        type=int,
        default=0,
        help="0 means all samples in the selected split.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include",
        nargs="+",
        default=list(DEFAULT_CONFIGS.keys()),
        choices=list(DEFAULT_CONFIGS.keys()),
        help="Subset of datasets to evaluate.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help=(
            "Optional list like heat=configs/heat_white.yml burgers=configs/burgers1d_white.yml. "
            "Only keys in {heat,burgers,rd,ns} are allowed."
        ),
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
        help=(
            "Optional output CSV path. If empty, defaults to "
            "results/ground_truth_residuals_<split>.csv"
        ),
    )
    return parser.parse_args()


class MeanAccumulator:
    def __init__(self):
        self._sums = OrderedDict()
        self._count = 0

    def add(self, metrics):
        self._count += 1
        for k, v in metrics.items():
            if not np.isfinite(v):
                continue
            self._sums[k] = self._sums.get(k, 0.0) + float(v)

    def means(self):
        if self._count == 0:
            return OrderedDict()
        out = OrderedDict()
        for k, v in self._sums.items():
            out[k] = v / float(self._count)
        return out

    @property
    def count(self):
        return self._count


def mean_abs(x):
    return float(x.abs().mean().item())


def build_config_map(args):
    cfg_map = OrderedDict(DEFAULT_CONFIGS)
    if args.configs is None:
        return cfg_map

    for item in args.configs:
        if "=" not in item:
            raise ValueError(f"Invalid --configs item '{item}'. Expected key=path.")
        key, path = item.split("=", 1)
        key = key.strip()
        path = path.strip()
        if key not in cfg_map:
            raise ValueError(f"Unknown config key '{key}'. Allowed: {list(cfg_map.keys())}")
        cfg_map[key] = path
    return cfg_map


def infer_nu_from_filename(file_name, default=0.001):
    match = re.search(r"mu([0-9.]+)", file_name)
    if not match:
        return float(default)
    try:
        return float(match.group(1).rstrip("."))
    except ValueError:
        return float(default)


def load_split_dataset(cfg, split):
    train_set, test_set = get_dataset(cfg.datasets)
    return train_set if split == "train" else test_set


def evaluate_heat(cfg, dataset, n_samples, device):
    nx = int(cfg.sample_dims[0])
    nt = int(cfg.sample_dims[1])
    x_grid = torch.linspace(0.0, 2.0 * math.pi, nx, device=device)
    t_grid = torch.linspace(0.0, 1.0, nt, device=device)

    acc = MeanAccumulator()
    for i in range(n_samples):
        u_exact, nu_exact = dataset[i]
        u_exact = u_exact.to(device=device, dtype=torch.float32)
        nu_exact = float(nu_exact)
        u_flat = u_exact.flatten()

        partial_rules = Residuals(
            data=u_exact.unsqueeze(0),
            x=x_grid,
            t_grid=t_grid,
            nx=nx,
            nt=nt,
            nu=nu_exact,
        )
        full_rules = HeatEquationResidualsFullPDE(data=u_exact, nx=nx, nt=nt, nu=nu_exact)
        full_rules_raw = HeatEquationResidualsFull(data=u_exact, nx=nx, nt=nt, nu=nu_exact)

        metrics = OrderedDict()
        metrics["partial/physics_mae"] = mean_abs(partial_rules.full_residual_heat(u_flat))
        metrics["partial/ic_mae"] = mean_abs(partial_rules.ic_residual(u_flat))
        metrics["partial/mass_mae"] = mean_abs(partial_rules.mass_residual_heat(u_flat)[1:])

        metrics["full/physics_mae"] = mean_abs(full_rules(u_flat))
        metrics["full/ic_mae"] = mean_abs(full_rules.ic_residual(u_flat))
        metrics["full/pde_scaled_mae"] = mean_abs(full_rules.pde_residual_scaled(u_flat))
        metrics["full/pde_raw_mae"] = mean_abs(full_rules_raw.pde_residual(u_flat))
        metrics["full/mass_mae"] = mean_abs(full_rules.mass_residual(u_flat)[1:])

        acc.add(metrics)

    return acc.means(), acc.count


def evaluate_burgers(cfg, dataset, n_samples, device):
    nx = int(cfg.sample_dims[0])
    nt = int(cfg.sample_dims[1])
    x_grid = torch.linspace(0.0, 1.0, nx, device=device)
    t_grid = torch.linspace(0.0, 1.0, nt, device=device)
    nu_default = 0.01
    bc_values = None
    n_bc = None
    if hasattr(dataset, "file") and "bc" in dataset.file:
        bc_values = torch.from_numpy(dataset.file["bc"][:]).to(device=device, dtype=torch.float32)
        n_bc = int(bc_values.shape[0])

    acc = MeanAccumulator()
    for i in range(n_samples):
        if n_bc is not None:
            _, i_bc = divmod(i, n_bc)
            left_bc = bc_values[i_bc]
        else:
            left_bc = None

        u_exact = dataset[i].to(device=device, dtype=torch.float32)
        u_flat = u_exact.flatten()

        if left_bc is None:
            left_bc = u_exact[0, 0]

        partial_rules = Residuals(
            data=u_exact.unsqueeze(0),
            x=x_grid,
            t_grid=t_grid,
            nx=nx,
            nt=nt,
            nu=nu_default,
            left_bc=left_bc,
        )
        full_rules = BurgersEquationResidualsFullPDE(data=u_exact, nx=nx, nt=nt, nu=nu_default)
        mass_rules = Residuals(
            data=u_exact.unsqueeze(0),
            x=x_grid,
            t_grid=t_grid,
            nx=nx,
            nt=nt,
            nu=nu_default,
            left_bc=left_bc,
        )

        metrics = OrderedDict()
        metrics["partial/physics_mae"] = mean_abs(partial_rules.full_residual_burgers2(u_flat, start_step=1))
        metrics["partial/bc_mae"] = mean_abs(partial_rules.bc_residual_burgers(u_flat, start_step=1))
        metrics["partial/mass_mae"] = mean_abs(partial_rules.mass_residual_burgers(u_flat)[1:])

        metrics["full/physics_mae"] = mean_abs(full_rules(u_flat))
        metrics["full/ic_mae"] = mean_abs(full_rules.ic_residual(u_flat))
        metrics["full/bc_left_mae"] = mean_abs(full_rules.bc_left_residual(u_flat))
        metrics["full/bc_right_mae"] = mean_abs(full_rules.bc_right_residual(u_flat))
        metrics["full/bc_mae"] = mean_abs(full_rules.bc_residual(u_flat))
        metrics["full/pde_scaled_mae"] = mean_abs(full_rules.pde_residual_scaled(u_flat))
        metrics["full/pde_raw_mae"] = mean_abs(full_rules.pde_residual(u_flat))
        metrics["full/mass_mae"] = mean_abs(mass_rules.mass_residual_burgers(u_flat)[1:])

        acc.add(metrics)

    return acc.means(), acc.count


def evaluate_rd(cfg, dataset, n_samples, device):
    nx = int(cfg.sample_dims[0])
    nt = int(cfg.sample_dims[1])

    rho_default = 0.01
    nu_default = 0.005
    if hasattr(dataset, "file"):
        rho_default = float(dataset.file.attrs.get("rho", rho_default))
        nu_default = float(dataset.file.attrs.get("nu", nu_default))

    if hasattr(dataset, "file") and "x" in dataset.file and "t" in dataset.file:
        x_grid = torch.from_numpy(dataset.file["x"][:]).to(device=device, dtype=torch.float32)
        t_grid = torch.from_numpy(dataset.file["t"][:]).to(device=device, dtype=torch.float32)
    else:
        x_grid = torch.linspace(0.0, 1.0, nx, device=device)
        t_grid = torch.linspace(0.0, 0.99, nt, device=device)

    acc = MeanAccumulator()
    for i in range(n_samples):
        u_exact = dataset[i].to(device=device, dtype=torch.float32)
        u_flat = u_exact.flatten()

        partial_rules = Residuals(
            data=u_exact.unsqueeze(0),
            x=x_grid,
            t_grid=t_grid,
            nx=nx,
            nt=nt,
            rho=rho_default,
            nu=nu_default,
        )
        full_rules = ReactionDiffusionResidualsFull(
            data=u_exact,
            nx=nx,
            nt=nt,
            nu=nu_default,
            rho=rho_default,
            x_grid=x_grid,
            t_grid=t_grid,
        )

        metrics = OrderedDict()
        metrics["partial/physics_mae"] = mean_abs(partial_rules.full_residual_rd(u_flat))
        metrics["partial/ic_mae"] = mean_abs(partial_rules.ic_residual(u_flat))
        metrics["partial/mass_mae"] = mean_abs(partial_rules.mass_residual_rd(u_flat)[1:])

        metrics["full/physics_mae"] = mean_abs(full_rules.full_residual(u_flat))
        metrics["full/ic_mae"] = mean_abs(full_rules.ic_residual(u_flat))
        metrics["full/bc_left_mae"] = mean_abs(full_rules.bc_left_residual(u_flat))
        metrics["full/bc_right_mae"] = mean_abs(full_rules.bc_right_residual(u_flat))
        metrics["full/bc_mae"] = mean_abs(full_rules.bc_residual(u_flat))
        metrics["full/pde_scaled_mae"] = mean_abs(full_rules.pde_residual_scaled(u_flat))
        metrics["full/pde_raw_mae"] = mean_abs(full_rules.pde_residual(u_flat))
        metrics["full/mass_mae"] = mean_abs(full_rules.mass_residual(u_flat)[1:])

        acc.add(metrics)

    return acc.means(), acc.count


def evaluate_ns(cfg, dataset, n_samples, device):
    data_file = cfg.datasets.test.data_file if hasattr(cfg.datasets, "test") else cfg.datasets.train.data_file
    full_h5_path = os.path.join(cfg.datasets.root, data_file)
    with h5py.File(full_h5_path, "r") as h5_file:
        f_all = torch.from_numpy(h5_file["f"][:]).to(device=device, dtype=torch.float32)
        nf = int(f_all.shape[0])

    u0 = dataset[0]
    nx = int(u0.shape[0])
    ny = int(u0.shape[1])
    nt = int(u0.shape[2])

    nu_default = infer_nu_from_filename(data_file, default=0.001)
    x_grid = torch.linspace(0.0, 1.0, nx + 1, device=device)[:-1]
    y_grid = torch.linspace(0.0, 1.0, ny + 1, device=device)[:-1]
    t_grid = torch.linspace(0.0, 49.0, nt, device=device)

    acc = MeanAccumulator()
    for i in range(n_samples):
        u_exact = dataset[i].to(device=device, dtype=torch.float32)
        u_flat = u_exact.flatten()
        f_i = f_all[i % nf]

        partial_rules = Residuals2D(
            data=u_exact.unsqueeze(0),
            x=x_grid,
            y=y_grid,
            t_grid=t_grid,
            nx=nx,
            ny=ny,
            nt=nt,
            nu=nu_default,
            rho=1.0,
        )
        full_rules = NavierStokesResidualsFullPDE(
            data=u_exact,
            forcing=f_i,
            nx=nx,
            ny=ny,
            nt=nt,
            nu=nu_default,
            x_grid=x_grid,
            y_grid=y_grid,
            t_grid=t_grid,
        )

        metrics = OrderedDict()
        metrics["partial/physics_mae"] = mean_abs(partial_rules.full_residual_ns(u_flat))
        metrics["partial/ic_mae"] = mean_abs(partial_rules.ic_residual_ns(u_flat))
        metrics["partial/mass_mae"] = mean_abs(partial_rules.mass_residual_ns(u_flat)[1:])

        metrics["full/physics_mae"] = mean_abs(full_rules.full_residual(u_flat))
        metrics["full/ic_mae"] = mean_abs(full_rules.ic_residual(u_flat))
        metrics["full/bc_left_mae"] = mean_abs(full_rules.bc_left_residual(u_flat))
        metrics["full/bc_right_mae"] = mean_abs(full_rules.bc_right_residual(u_flat))
        metrics["full/bc_mae"] = mean_abs(full_rules.bc_residual(u_flat))
        metrics["full/pde_scaled_mae"] = mean_abs(full_rules.pde_residual_scaled(u_flat))
        metrics["full/pde_raw_mae"] = mean_abs(full_rules.pde_residual(u_flat))
        metrics["full/mass_mae"] = mean_abs(full_rules.mass_residual(u_flat)[1:])

        acc.add(metrics)

    return acc.means(), acc.count


def format_table(results):
    headers = ["dataset", "split", "n_samples", "metric", "value"]
    rows = []

    for dataset_name, payload in results.items():
        split = payload["split"]
        n_samples = payload["n_samples"]
        metrics = payload["metrics"]
        for k, v in metrics.items():
            rows.append([dataset_name, split, str(n_samples), k, f"{v:.8e}"])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def fmt_row(vals):
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals))

    out = []
    out.append(fmt_row(headers))
    out.append("-+-".join("-" * w for w in widths))
    for row in rows:
        out.append(fmt_row(row))
    return "\n".join(out)


def flatten_results_rows(results):
    rows = []
    for dataset_name, payload in results.items():
        split = payload["split"]
        n_samples = payload["n_samples"]
        metrics = payload["metrics"]
        for metric_name, value in metrics.items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "split": split,
                    "n_samples": int(n_samples),
                    "metric": metric_name,
                    "value": float(value),
                }
            )
    return rows


def write_csv(rows, csv_path):
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = ["dataset", "split", "n_samples", "metric", "value"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg_map = build_config_map(args)

    evaluators = {
        "heat": evaluate_heat,
        "burgers": evaluate_burgers,
        "rd": evaluate_rd,
        "ns": evaluate_ns,
    }

    results = OrderedDict()

    for key in args.include:
        cfg_path = cfg_map[key]
        cfg = load_config(cfg_path)
        dataset = load_split_dataset(cfg, args.split)

        total = len(dataset)
        n_samples = total if args.max_samples_per_dataset <= 0 else min(total, args.max_samples_per_dataset)
        print(f"[{key}] split={args.split} samples={n_samples}/{total} cfg={cfg_path}")

        metrics, used = evaluators[key](cfg, dataset, n_samples, device)
        results[key] = {
            "split": args.split,
            "n_samples": used,
            "metrics": metrics,
        }

    print("\nGround-truth residual floor summary (mean absolute values):")
    print(format_table(results))

    rows = flatten_results_rows(results)
    csv_path = args.csv_path.strip()
    if not csv_path:
        csv_path = os.path.join("results", f"ground_truth_residuals_{args.split}.csv")
    write_csv(rows, csv_path)
    print(f"\nSaved CSV summary to: {csv_path}")


if __name__ == "__main__":
    main()
