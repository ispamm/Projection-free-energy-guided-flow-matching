import os
import re
import argparse
import time
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from scripts.training.utils import load_config
from models import get_flow_model
from models.constraints import DirichletCondition
from pcfm.ffm_sampler import FFM_NS_sampler
from pcfm.pcfm_sampling import make_grid, fast_project_batched
from pcfm.constraints import Residuals2D
from metrics import (
    compute_physical_residual,
    compute_ns_physical_residual,
    compute_distribution_metrics,
    compute_samplewise_mse,
    MetricsTracker,
)
from physics_engine import NavierStokesResidualsFullPDE


try:
    from datasets import get_dataset
    from torch.utils.data import DataLoader
except ImportError:
    print("Warning: Check the import for 'get_dataset' in your repository.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generative PDE Benchmark & Ablation")

    # System & Reproducibility
    parser.add_argument("--config_path", type=str, default="configs/ns_lightning_white.yml", help="config.yml path for the experiment")
    parser.add_argument("--ckpt_path", type=str, default="logs/ns_lightning_white/latest.pt", help="Checkpoint (.pt) path for the pretrained model")
    parser.add_argument("--device", type=str, default="cuda:1", help="GPU (es. cuda:0)")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed for reproducibility")

    # Experiment Settings
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--models", nargs="+", default=["all"], help="Baseline to test: vanilla, pcfm, proflow, ours, all...")
    parser.add_argument("--n_steps", type=int, default=100, help="Number of sampling steps for each method")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to generate and compare")
    parser.add_argument("--log_every", type=int, default=10, help="Log intermediate metrics to WandB every N processed samples")

    # Ablation Parameters for "Ours"
    parser.add_argument("--gamma_list", nargs="+", type=float, default=[1.0], help="Ablation for gamma values. Es: --gamma_list 0.1 1.0 2.0")

    return parser.parse_args()


def _load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                clean_state_dict[key[6:]] = value
            elif key.startswith("net."):
                clean_state_dict[key[4:]] = value
            else:
                clean_state_dict[key] = value
        model.load_state_dict(clean_state_dict, strict=False)
        print("-> Successfully loaded PyTorch Lightning checkpoint.")
        return

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print("-> Successfully loaded standard PyTorch checkpoint.")
        return

    model.load_state_dict(ckpt)
    print("-> Successfully loaded raw state_dict.")


def _infer_nu_from_filename(file_name, default=0.001):
    match = re.search(r"mu([0-9.]+)", file_name)
    if not match:
        return float(default)
    try:
        return float(match.group(1).rstrip("."))
    except ValueError:
        return float(default)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    run_name = args.run_name if args.run_name else f"eval_partial_{args.n_samples}s_{args.n_steps}step"

    wandb.init(project="pcfm-physics-comparison", name=run_name, config=vars(args))

    print(f"1. Loading configuration from {args.config_path}...")
    config = load_config(args.config_path)
    cond_type = getattr(config, "cond_type", "ic")
    if cond_type != "ic":
        raise NotImplementedError(f"Unsupported cond_type for ns partial: {cond_type}")

    print(f"Loading model and weights from {args.ckpt_path} to {device}...")
    model = get_flow_model(config.model, config.encoder).to(device)
    _load_checkpoint(model, args.ckpt_path, device)
    model.eval()

    dims = config.sample_dims
    sampler = FFM_NS_sampler(model)

    print("2. Ground Truth Loading...")
    _, test_set = get_dataset(config.datasets)
    test_loader = DataLoader(test_set, batch_size=args.n_samples, shuffle=False)

    batch = next(iter(test_loader))
    u_exact = batch.to(device)

    if u_exact.dim() == 5 and u_exact.shape[1] == 1:
        u_exact = u_exact.squeeze(1)

    actual_samples = u_exact.shape[0]
    if actual_samples < args.n_samples:
        print(f"\n[WARNING] Asked for {args.n_samples} samples, but the test dataset has only {actual_samples}.")
        print(f"Cap down n_samples to {actual_samples} to avoid crashes.\n")
        args.n_samples = actual_samples

    u_exact = u_exact[:args.n_samples]
    u_exact_all = u_exact.cpu()

    actual_nx = u_exact.shape[1]
    actual_ny = u_exact.shape[2]
    actual_nt = u_exact.shape[3]

    full_h5_path = os.path.join(config.datasets.root, config.datasets.test.data_file)
    print(f"Opening H5 file for forcing: {full_h5_path}")
    with h5py.File(full_h5_path, "r") as h5_file:
        f_all = torch.from_numpy(h5_file["f"][:]).float()

    nu_default = _infer_nu_from_filename(config.datasets.test.data_file, default=0.001)

    x_grid = torch.linspace(0.0, 1.0, actual_nx + 1, device=device)[:-1]
    y_grid = torch.linspace(0.0, 1.0, actual_ny + 1, device=device)[:-1]
    # NS data is generated with T=49 and 50 recorded snapshots.
    t_grid = torch.linspace(0.0, 49.0, actual_nt, device=device)

    actual_dims = [actual_nx, actual_ny, actual_nt]

    print("3. Generation of initial noise (u0)...")
    grid = make_grid(actual_dims, device)
    u0 = model.gp.sample(grid, actual_dims, n_samples=args.n_samples).to(device)

    print("4. Setup Trackers...")
    trackers = {}
    method_active = {}
    method_success_counts = {}
    method_failure_sample = {}
    method_component_sums = {}

    def mark_success(name, sample, residual, start_t, end_t):
        trackers[name].record_step(sample, residual, start_t, end_t)
        method_success_counts[name] += 1

    def deactivate_if_nan(name, residual, sample_idx):
        if not np.isfinite(residual):
            method_active[name] = False
            method_failure_sample[name] = sample_idx + 1
            print(
                f"[DISABLE] {name} produced Physics_Error={residual} "
                f"at sample {sample_idx + 1}/{args.n_samples}. Skipping it for remaining samples."
            )
            return True
        return False

    def log_periodic_metrics(processed_samples: int):
        if processed_samples <= 0:
            return

        periodic_log = {
            "Progress/processed_samples": processed_samples,
            "Progress/processed_ratio": processed_samples / max(1, args.n_samples),
        }

        for name, tracker in trackers.items():
            if len(tracker.times) == 0:
                continue
            periodic_log[f"Periodic/Speed (sec/sample)/{name}"] = tracker.get_average_speed()
            periodic_log[f"Periodic/Physics_Error (MAE)/{name}"] = tracker.get_average_residual()

        periodic_table = wandb.Table(
            columns=[
                "Method",
                "Success Rate (%)",
                "Generated Samples",
                "Speed (sec/sample)",
                "Physical Residual (MAE)",
                "IC Error",
                "BC Left Error",
                "BC Right Error",
                "BC Error",
                "PDE Error (Scaled)",
                "PDE Error (Raw)",
                "Mass Error",
                "CL Error",
                "MMSE",
                "SMSE",
                "SampleMSE",
            ]
        )

        for name, tracker in trackers.items():
            success_count = method_success_counts[name]
            success_rate = 100.0 * success_count / max(1, args.n_samples)
            failed_with_nan = method_failure_sample[name] is not None

            if (not failed_with_nan) and success_count == args.n_samples:
                u_pred_all = tracker.get_all_samples_tensor()
                u_exact_ref = u_exact_all

                component_means = get_component_means(name, success_count)
                ce_ic = component_means["ic"]
                ce_bc_left = component_means["bc_left"]
                ce_bc_right = component_means["bc_right"]
                ce_bc = component_means["bc"]
                ce_pde_scaled = component_means["pde_scaled"]
                ce_pde_raw = component_means["pde_raw"]
                ce_mass = component_means["mass"]

                mass_t0 = u_pred_all[:, :, :, 0].mean(dim=(1, 2), keepdim=True)
                mass_all_t = u_pred_all.mean(dim=(1, 2))
                ce_cl = (mass_all_t - mass_t0).abs().mean().item()

                mmse, smse = compute_distribution_metrics(u_pred_all, u_exact_ref)
                sample_mse = compute_samplewise_mse(u_pred_all, u_exact_ref)
                speed = tracker.get_average_speed()
                residual = tracker.get_average_residual()
            else:
                speed = np.nan
                residual = np.nan
                ce_ic = np.nan
                ce_bc_left = np.nan
                ce_bc_right = np.nan
                ce_bc = np.nan
                ce_pde_scaled = np.nan
                ce_pde_raw = np.nan
                ce_mass = np.nan
                ce_cl = np.nan
                mmse = np.nan
                smse = np.nan
                sample_mse = np.nan

            periodic_log[f"Periodic/IC_Error/{name}"] = ce_ic
            periodic_log[f"Periodic/BC_Left_Error/{name}"] = ce_bc_left
            periodic_log[f"Periodic/BC_Right_Error/{name}"] = ce_bc_right
            periodic_log[f"Periodic/BC_Error/{name}"] = ce_bc
            periodic_log[f"Periodic/PDE_Error_Scaled/{name}"] = ce_pde_scaled
            periodic_log[f"Periodic/PDE_Error_Raw/{name}"] = ce_pde_raw
            periodic_log[f"Periodic/Mass_Error/{name}"] = ce_mass

            periodic_table.add_data(
                name,
                success_rate,
                success_count,
                speed,
                residual,
                ce_ic,
                ce_bc_left,
                ce_bc_right,
                ce_bc,
                ce_pde_scaled,
                ce_pde_raw,
                ce_mass,
                ce_cl,
                mmse,
                smse,
                sample_mse,
            )

        if len(periodic_log) > 2:
            periodic_log["Periodic/Results_Table"] = periodic_table
            wandb.log(periodic_log)

    if "all" in args.models or "vanilla" in args.models:
        trackers["Vanilla"] = MetricsTracker("Vanilla")

    if "all" in args.models or "proflow" in args.models:
        trackers["PROFlow"] = MetricsTracker("PROFlow")

    if "all" in args.models or "pcfm" in args.models:
        trackers["PCFM"] = MetricsTracker("PCFM")

    if "all" in args.models or "ours" in args.models:
        for g in args.gamma_list:
            trackers[f"Ours_g{g}"] = MetricsTracker(f"Ours (gamma={g})")

    if "all" in args.models or "eci" in args.models:
        trackers["ECI"] = MetricsTracker("ECI")

    if "all" in args.models or "diffusionpde" in args.models:
        trackers["DiffusionPDE"] = MetricsTracker("DiffusionPDE")

    if "all" in args.models or "dflow" in args.models:
        trackers["DFlow"] = MetricsTracker("DFlow")

    for method_name in trackers:
        method_active[method_name] = True
        method_success_counts[method_name] = 0
        method_failure_sample[method_name] = None
        method_component_sums[method_name] = {
            "ic": 0.0,
            "bc_left": 0.0,
            "bc_right": 0.0,
            "bc": 0.0,
            "pde_scaled": 0.0,
            "pde_raw": 0.0,
            "mass": 0.0,
        }

    def compute_component_errors(u_pred, u_true, physics_rules_eval):
        u_pred_flat = u_pred.flatten()
        u_true_flat = u_true.flatten()

        ic_error = torch.nn.functional.mse_loss(u_pred[:, :, :, 0], u_true[:, :, :, 0]).item()

        bc_left_error = physics_rules_eval.bc_left_residual(u_pred_flat).abs().mean().item()
        bc_right_error = physics_rules_eval.bc_right_residual(u_pred_flat).abs().mean().item()
        bc_error = 0.5 * (bc_left_error + bc_right_error)

        pde_scaled_error = physics_rules_eval.pde_residual_scaled(u_pred_flat).abs().mean().item()
        pde_raw_error = physics_rules_eval.pde_residual(u_pred_flat).abs().mean().item()
        mass_error = physics_rules_eval.mass_residual(u_pred_flat)[1:].abs().mean().item()

        return {
            "ic": ic_error,
            "bc_left": bc_left_error,
            "bc_right": bc_right_error,
            "bc": bc_error,
            "pde_scaled": pde_scaled_error,
            "pde_raw": pde_raw_error,
            "mass": mass_error,
        }

    def record_component_errors(name, component_errors):
        for key, value in component_errors.items():
            method_component_sums[name][key] += float(value)

    def get_component_means(name, denom):
        return {key: value / max(1, denom) for key, value in method_component_sums[name].items()}

    dx_val = (x_grid[1] - x_grid[0]).item()
    dy_val = (y_grid[1] - y_grid[0]).item()
    dt_val = (t_grid[1] - t_grid[0]).item()

    print("5. Start Inference Loop...")
    for i in range(args.n_samples):
        if not any(method_active.values()):
            print("\n[EARLY STOP] All methods disabled due to NaN/Inf Physics_Error.")
            break

        if i % 10 == 0:
            print(f"\n--- Processing Sample {i + 1}/{args.n_samples} ---")

        u0_i = u0[i : i + 1]
        u_exact_i = u_exact[i : i + 1].to(device)

        nf_unique = f_all.shape[0]
        f_i = f_all[i % nf_unique].to(device)

        # Partial guidance: original constraints residual (IC + mass) from constraints.py
        physics_rules_guidance = Residuals2D(
            data=u_exact_i,
            x=x_grid,
            y=y_grid,
            t_grid=t_grid,
            nx=actual_nx,
            ny=actual_ny,
            nt=actual_nt,
            nu=nu_default,
            rho=1.0,
        )

        # Eval residual: full NS residual for richer diagnostics/metrics.
        physics_rules_eval = NavierStokesResidualsFullPDE(
            data=u_exact_i,
            forcing=f_i,
            nx=actual_nx,
            ny=actual_ny,
            nt=actual_nt,
            nu=nu_default,
            x_grid=x_grid,
            y_grid=y_grid,
            t_grid=t_grid,
        )

        def hfunc(u_flat):
            return physics_rules_guidance.full_residual_ns(u_flat)

        eval_hfunc = hfunc

        ic_err = physics_rules_guidance.ic_residual_ns(u_exact_i.flatten()).abs().mean().item()
        mass_err = physics_rules_guidance.mass_residual_ns(u_exact_i.flatten())[1:].abs().mean().item()
        pde_err = physics_rules_eval.pde_residual(u_exact_i.flatten()).abs().mean().item()
        print(f"Sanity Check -> IC: {ic_err:.5f} | Mass: {mass_err:.5f} | PDE: {pde_err:.5f}")

        mask_bool = torch.zeros_like(u_exact_i, dtype=torch.bool)
        mask_bool[:, :, :, 0] = True
        mask_float = mask_bool.float()

        def composite_loss_fn(u_pred, u_true, mask_tensor):
            data_loss = ((u_pred - u_true) * mask_tensor).square().sum()
            pinn_residual = hfunc(u_pred)
            pinn_loss = (pinn_residual ** 2).sum() * 0.01
            return data_loss + pinn_loss

        constraint = DirichletCondition(value=u_exact_i, mask=mask_bool)

        if method_active.get("ECI", False):
            start_t = time.time()
            u_eci = sampler.eci_sample(u0_i, args.n_steps, n_mix=5, resample_step=5, constraint=constraint)
            res_eci = compute_physical_residual(u_eci, eval_hfunc)
            end_t = time.time()
            if not deactivate_if_nan("ECI", res_eci, i):
                record_component_errors("ECI", compute_component_errors(u_eci, u_exact_i, physics_rules_eval))
                mark_success("ECI", u_eci, res_eci, start_t, end_t)

        if method_active.get("DiffusionPDE", False):
            start_t = time.time()
            u_diffpde = sampler.guided_sample(
                u0_i,
                u_exact_i,
                mask_float,
                args.n_steps,
                loss_fn=composite_loss_fn,
                eta=0.01,
            )
            res_diffpde = compute_physical_residual(u_diffpde, eval_hfunc)
            end_t = time.time()
            if not deactivate_if_nan("DiffusionPDE", res_diffpde, i):
                record_component_errors("DiffusionPDE", compute_component_errors(u_diffpde, u_exact_i, physics_rules_eval))
                mark_success("DiffusionPDE", u_diffpde, res_diffpde, start_t, end_t)

        if method_active.get("DFlow", False):
            start_t = time.time()
            u_dflow = sampler.dflow_ns_sample(
                u_exact_i,
                mask_float,
                n_sample=1,
                n_step=args.n_steps,
                n_iter=10,
                lr=0.01,
                loss_fn=composite_loss_fn,
            )
            res_dflow = compute_physical_residual(u_dflow, eval_hfunc)
            end_t = time.time()
            if not deactivate_if_nan("DFlow", res_dflow, i):
                record_component_errors("DFlow", compute_component_errors(u_dflow, u_exact_i, physics_rules_eval))
                mark_success("DFlow", u_dflow, res_dflow, start_t, end_t)

        if method_active.get("Vanilla", False):
            start_t = time.time()
            u_vanilla = sampler.vanilla_sample(u0_i, args.n_steps)
            res_vanilla = compute_physical_residual(u_vanilla, eval_hfunc)
            end_t = time.time()
            if not deactivate_if_nan("Vanilla", res_vanilla, i):
                record_component_errors("Vanilla", compute_component_errors(u_vanilla, u_exact_i, physics_rules_eval))
                mark_success("Vanilla", u_vanilla, res_vanilla, start_t, end_t)

        if method_active.get("PROFlow", False):
            start_t = time.time()
            u_proflow = sampler.proflow_sample(u0_i, args.n_steps, hfunc, K=3, lr_base=0.1)
            res_proflow = compute_physical_residual(u_proflow, eval_hfunc)
            end_t = time.time()
            if not deactivate_if_nan("PROFlow", res_proflow, i):
                record_component_errors("PROFlow", compute_component_errors(u_proflow, u_exact_i, physics_rules_eval))
                mark_success("PROFlow", u_proflow, res_proflow, start_t, end_t)

        if method_active.get("PCFM", False):
            start_t = time.time()
            u_pcfm_i = sampler.pcfm_sample(
                u0_i,
                args.n_steps,
                hfunc=hfunc,
                newtonsteps=1,
                guided_interpolation=False,
                interpolation_params={"custom_lam": 1.0, "step_size": 0.01, "num_steps": 20},
            )
            u_pcfm_final_32 = u_pcfm_i.detach()
            u_flat_64 = u_pcfm_final_32.flatten().unsqueeze(0).to(torch.float64)
            u_pcfm_final_proj = fast_project_batched(u_flat_64, hfunc, max_iter=2)
            u_pcfm = u_pcfm_final_proj.view(u_pcfm_final_32.shape).to(torch.float32).detach()

            res_pcfm = compute_physical_residual(u_pcfm, hfunc)
            end_t = time.time()
            if not deactivate_if_nan("PCFM", res_pcfm, i):
                record_component_errors("PCFM", compute_component_errors(u_pcfm, u_exact_i, physics_rules_eval))
                mark_success("PCFM", u_pcfm, res_pcfm, start_t, end_t)

        for g in args.gamma_list:
            tracker_key = f"Ours_g{g}"
            if method_active.get(tracker_key, False):
                start_t = time.time()
                u_ours, _ = sampler.continuous_guided_sample(
                    u0_i,
                    args.n_steps,
                    hfunc,
                    gamma_max=g,
                    final_refinement=False,
                    refinement_steps=1,
                    refinement_lr=0.2,
                )
                res_ours = compute_physical_residual(u_ours, hfunc)
                end_t = time.time()
                if not deactivate_if_nan(tracker_key, res_ours, i):
                    record_component_errors(tracker_key, compute_component_errors(u_ours, u_exact_i, physics_rules_eval))
                    mark_success(tracker_key, u_ours, res_ours, start_t, end_t)

        if args.log_every > 0 and ((i + 1) % args.log_every == 0):
            print(f"[WandB] Periodic log at sample {i + 1}/{args.n_samples}")
            log_periodic_metrics(i + 1)

    log_periodic_metrics(args.n_samples)

    print("\n6. Computing Distribution Metrics and Logging to Wandb...")
    wandb_log_dict = {}
    columns = [
        "Method",
        "Success Rate (%)",
        "Generated Samples",
        "Speed (sec/sample)",
        "Physical Residual (MAE)",
        "IC Error",
        "BC Left Error",
        "BC Right Error",
        "BC Error",
        "PDE Error (Scaled)",
        "PDE Error (Raw)",
        "Mass Error",
        "CL Error",
        "MMSE",
        "SMSE",
        "SampleMSE",
    ]
    results_table = wandb.Table(columns=columns)

    for name, tracker in trackers.items():
        success_count = method_success_counts[name]
        success_rate = 100.0 * success_count / max(1, args.n_samples)
        failed_with_nan = method_failure_sample[name] is not None

        if (not failed_with_nan) and success_count == args.n_samples:
            tracker.print_summary()
            u_pred_all = tracker.get_all_samples_tensor()
            u_exact_ref = u_exact_all

            component_means = get_component_means(name, success_count)
            ce_ic = component_means["ic"]
            ce_bc_left = component_means["bc_left"]
            ce_bc_right = component_means["bc_right"]
            ce_bc = component_means["bc"]
            ce_pde_scaled = component_means["pde_scaled"]
            ce_pde_raw = component_means["pde_raw"]
            ce_mass = component_means["mass"]

            mass_t0 = u_pred_all[:, :, :, 0].mean(dim=(1, 2), keepdim=True)
            mass_all_t = u_pred_all.mean(dim=(1, 2))
            ce_cl = (mass_all_t - mass_t0).abs().mean().item()

            mmse, smse = compute_distribution_metrics(u_pred_all, u_exact_ref)
            sample_mse = compute_samplewise_mse(u_pred_all, u_exact_ref)
            speed = tracker.get_average_speed()
            residual = tracker.get_average_residual()

            save_dir = f"results/{run_name}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(u_pred_all.cpu(), os.path.join(save_dir, f"{name}_tensors.pt"))
        else:
            speed = np.nan
            residual = np.nan
            ce_ic = np.nan
            ce_bc_left = np.nan
            ce_bc_right = np.nan
            ce_bc = np.nan
            ce_pde_scaled = np.nan
            ce_pde_raw = np.nan
            ce_mass = np.nan
            ce_cl = np.nan
            mmse = np.nan
            smse = np.nan
            sample_mse = np.nan

            if failed_with_nan:
                print(f"\n--- {name} Summary ---")
                print(
                    f"Disabled at sample {method_failure_sample[name]}/{args.n_samples} "
                    f"due to NaN/Inf Physics_Error. Final quality metrics are set to NaN."
                )
            else:
                print(f"\n--- {name} Summary ---")
                print("Evaluation did not complete on the full sample set. Final quality metrics are set to NaN.")

        wandb_log_dict[f"Success_Rate (%)/{name}"] = success_rate
        wandb_log_dict[f"Success_Samples/{name}"] = success_count
        wandb_log_dict[f"Speed (sec/sample)/{name}"] = speed
        wandb_log_dict[f"Physics_Error (MAE)/{name}"] = residual
        wandb_log_dict[f"Physics_Error_IC/{name}"] = ce_ic
        wandb_log_dict[f"Physics_Error_BC_Left/{name}"] = ce_bc_left
        wandb_log_dict[f"Physics_Error_BC_Right/{name}"] = ce_bc_right
        wandb_log_dict[f"Physics_Error_BC/{name}"] = ce_bc
        wandb_log_dict[f"Physics_Error_PDE_Scaled/{name}"] = ce_pde_scaled
        wandb_log_dict[f"Physics_Error_PDE_Raw/{name}"] = ce_pde_raw
        wandb_log_dict[f"Physics_Error_Mass/{name}"] = ce_mass
        wandb_log_dict[f"Physics_Error_CL/{name}"] = ce_cl
        wandb_log_dict[f"Distribution_MMSE/{name}"] = mmse
        wandb_log_dict[f"Distribution_SMSE/{name}"] = smse
        wandb_log_dict[f"Distribution_SampleMSE/{name}"] = sample_mse
        results_table.add_data(
            name,
            success_rate,
            success_count,
            speed,
            residual,
            ce_ic,
            ce_bc_left,
            ce_bc_right,
            ce_bc,
            ce_pde_scaled,
            ce_pde_raw,
            ce_mass,
            ce_cl,
            mmse,
            smse,
            sample_mse,
        )

    wandb_log_dict["Final_Results_Table"] = results_table
    wandb.log(wandb_log_dict)

    print("\n7. Drawing random results for comparison...")
    plottable_methods = [(name, tracker) for name, tracker in trackers.items() if method_success_counts[name] > 0]

    if len(plottable_methods) == 0:
        print("No plottable methods: all methods were disabled or produced zero valid samples.")
    else:
        max_common_samples = min(method_success_counts[name] for name, _ in plottable_methods)
        plot_count = min(6, max_common_samples)
        plot_indices = random.sample(range(max_common_samples), plot_count)
        num_rows = 1 + len(plottable_methods)
        fig, axes = plt.subplots(num_rows, len(plot_indices), figsize=(18, 3 * num_rows))

        if num_rows == 1:
            axes = [axes]
        if len(plot_indices) == 1:
            axes = [[ax] for ax in axes]

        for col, idx in enumerate(plot_indices):
            axes[0][col].imshow(u_exact_all[idx, :, :, -1].numpy(), cmap="magma", origin="lower")
            axes[0][col].set_title(f"Exact {idx} (t=End)")
            axes[0][col].axis("off")

            for row, (name, tracker) in enumerate(plottable_methods, start=1):
                u_plot = tracker.get_all_samples_tensor()[idx, :, :, -1].numpy()
                axes[row][col].imshow(u_plot, cmap="magma", origin="lower")
                axes[row][col].set_title(f"{name} {idx}")
                axes[row][col].axis("off")

        plt.tight_layout()
        wandb.log({"Comparison_Plot": wandb.Image(fig)})
        plt.savefig("comparison_plot.png")
        print("Plot saved as comparison_plot.png")

    wandb.finish()


if __name__ == "__main__":
    main()
