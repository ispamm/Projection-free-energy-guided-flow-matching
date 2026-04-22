import os
import argparse
import torch
import time
import random
import wandb
import numpy as np
import matplotlib.pyplot as plt
from scripts.training.utils import load_config
from models import get_flow_model
from models.constraints import DirichletCondition
from pcfm.ffm_sampler import FFM_sampler
from pcfm.pcfm_sampling import make_grid, fast_project_batched
from pcfm.constraints import Residuals
from metrics import compute_physical_residual, compute_speed, compute_distribution_metrics, compute_samplewise_mse, MetricsTracker
from physics_engine import HeatEquationResidualsFull


try:
    from datasets import get_dataset
    from torch.utils.data import DataLoader
except ImportError:
    print("Warning: Check the import for 'get_dataset' in your repository.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generative PDE Benchmark & Ablation")
    
    # System & Reproducibility
    parser.add_argument("--config_path", type=str, default="configs/heat_white.yml", help="config.yml path for the experiment")
    parser.add_argument("--ckpt_path", type=str, default="logs/heat_white_test/latest.pt", help="Checkpoint (.pt) path for the pretrained model")
    parser.add_argument("--device", type=str, default="cuda:1", help="GPU (es. cuda:0)")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed for reproducibility")
    
    # Experiment Settings
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--models", nargs="+", default=["all"], 
                        help="Baseline to test: vanilla, pcfm, proflow, ours, all...")
    parser.add_argument("--n_steps", type=int, default=100, help="Number of sampling steps for each method")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to generate and compare")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log intermediate metrics to WandB every N processed samples")
    
    # Ablation Parameters for "Ours"
    parser.add_argument("--gamma_list", nargs="+", type=float, default=[1.0], 
                        help="Ablation for gamma values. Es: --gamma_list 0.1 1.0 2.0")
    
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    run_name = args.run_name if args.run_name else f"eval_partial_{args.n_samples}s_{args.n_steps}step"
    
    wandb.init(
        project="pcfm-physics-comparison", 
        name=run_name,
        config=vars(args)
    )

    print(f"1. Loading configuration from {args.config_path}...")
    config = load_config(args.config_path)
    cond_type = getattr(config, "cond_type", "ic")
    if cond_type != "ic":
        raise NotImplementedError(f"Unsupported cond_type for heat partial: {cond_type}")
    
    print(f"Loading model and weights from {args.ckpt_path} to {device}...")
    model = get_flow_model(config.model, config.encoder).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    dims = config.sample_dims
    sampler = FFM_sampler(model, model.gp)

    print("2. Ground Truth Loading...")
    train_set, test_set = get_dataset(config.datasets)
    test_loader = DataLoader(test_set, batch_size=args.n_samples, shuffle=False)
    
    # 1. The batch is the real data (u_exact) we want to compare against
    batch = next(iter(test_loader))
    u_exact = batch[0]
    v_exact = batch[1].to(device)
    # if the tensor has an unnecessary 'channel' dimension (e.g., [3, 1, 100, 100]), we remove it
    if u_exact.dim() == 4 and u_exact.shape[1] == 1:
        u_exact = u_exact.squeeze(1)
    actual_samples = u_exact.shape[0]
    if actual_samples < args.n_samples:
        print(f"\n[WARNING] Asked for {args.n_samples} samples, but the test dataset has only {actual_samples}.")
        print(f"Cap down n_samples to {actual_samples} to avoid crashes.\n")
        args.n_samples = actual_samples
        
    
    u_exact_all = u_exact.cpu()  # Keep a copy on CPU for later metric computations
        
    # 2. Build the spatial and temporal grids (x_grid, t_grid) based on the dimensions specified in the config
    x_grid = torch.linspace(0, 2 * np.pi, dims[0], device=device)
    t_grid = torch.linspace(0, 1, dims[1], device=device)

    print("3. Generation of initial noise (u0)...")
    grid = make_grid(dims, device, start=(0.0, 0.0), end=(2 * np.pi, 1.0))
    u0 = model.gp.sample(grid, dims, n_samples=args.n_samples).to(device)

    print("4. Setup Trackers...")
    trackers = {}
    method_active = {}
    method_success_counts = {}
    method_failure_sample = {}
    method_component_sums = {}

    def mark_success(name, sample, residual, start_t, end_t):
        """Record valid sample stats and keep the method active."""
        trackers[name].record_step(sample, residual, start_t, end_t)
        method_success_counts[name] += 1

    def deactivate_if_nan(name, residual, sample_idx):
        """Disable a method permanently when Physics_Error becomes NaN/Inf."""
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
        """Log running averages to WandB during inference."""
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

        periodic_table = wandb.Table(columns=["Method", "Success Rate (%)", "Generated Samples", "Speed (sec/sample)", "Physical Residual (MAE)", "IC Error", "BC Left Error", "BC Right Error", "BC Error", "PDE Error (Scaled)", "PDE Error (Raw)", "Mass Error", "CL Error", "MMSE", "SMSE", "SampleMSE"])
        for name, tracker in trackers.items():
            success_count = method_success_counts[name]
            success_rate = 100.0 * success_count / max(1, args.n_samples)
            failed_with_nan = method_failure_sample[name] is not None

            if (not failed_with_nan) and success_count == args.n_samples:
                u_pred_all = tracker.get_all_samples_tensor()
                u_exact_ref = u_exact_all

                component_means = get_component_means(name, success_count)
                ce_ic = component_means["ic"]
                ce_bc_left = np.nan
                ce_bc_right = np.nan
                ce_bc = np.nan
                ce_pde_scaled = component_means["pde_scaled"]
                ce_pde_raw = component_means["pde_raw"]
                ce_mass = component_means["mass"]
                mass_t0 = u_pred_all[:, :, 0].mean(dim=1, keepdim=True)
                mass_all_t = u_pred_all.mean(dim=1)
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
            periodic_table.add_data(name, success_rate, success_count, speed, residual, ce_ic, ce_bc_left, ce_bc_right, ce_bc, ce_pde_scaled, ce_pde_raw, ce_mass, ce_cl, mmse, smse, sample_mse)

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
            "pde_scaled": 0.0,
            "pde_raw": 0.0,
            "mass": 0.0,
        }

    def compute_component_errors(u_pred, u_true, physics_rules, physics_rules_pde):
        u_pred_flat = u_pred.flatten()
        ic_error = torch.nn.functional.mse_loss(u_pred[:, :, 0], u_true[:, :, 0]).item()
        pde_raw_error = physics_rules_pde.pde_residual(u_pred_flat).abs().mean().item()
        pde_scaled_error = pde_raw_error * (physics_rules_pde.dx ** 2)
        mass_error = physics_rules.mass_residual_heat(u_pred_flat)[1:].abs().mean().item()
        return {
            "ic": ic_error,
            "pde_scaled": pde_scaled_error,
            "pde_raw": pde_raw_error,
            "mass": mass_error,
        }

    def record_component_errors(name, component_errors):
        for key, value in component_errors.items():
            method_component_sums[name][key] += float(value)

    def get_component_means(name, denom):
        return {key: value / max(1, denom) for key, value in method_component_sums[name].items()}

    print("4. Iterative Generation (to respect the original Residuals class)...")
    sampler = FFM_sampler(model, model.gp)

    
    print("\n5. Start Inference Loop...")
    for i in range(args.n_samples):

        if not any(method_active.values()):
            print("\n[EARLY STOP] All methods disabled due to NaN/Inf Physics_Error.")
            break

        if i % 10 == 0:
             print(f"\n--- Processing Sample {i+1}/{args.n_samples} ---")
                
        # Extract the individual sample while maintaining the fake batch dimension [1, nx, nt]
        u0_i = u0[i:i+1]
        u_exact_i = u_exact[i:i+1].to(device)
        nu_exact = float(v_exact[i])
        
        # Initialize the physics rules ONLY for this specific sample.
        # Heat configs in this repository are IC-only, so the data mask follows cond_type=ic.
        physics_rules = Residuals(
            data=u_exact_i,
            x=x_grid,
            t_grid=t_grid,
            nx=dims[0],
            nt=dims[1],
            nu=nu_exact,
        )
        physics_rules_pde = HeatEquationResidualsFull(data=u_exact_i, nx=dims[0], nt=dims[1], nu=nu_exact)

        def hfunc(u_flat):
            return physics_rules.full_residual_heat(u_flat)

        eval_hfunc = hfunc

        ic_err = physics_rules.ic_residual(u_exact_i.flatten()).abs().mean().item()
        mass_err = physics_rules.mass_residual_heat(u_exact_i.flatten())[1:].abs().mean().item()

        print(f"Sanity Check -> IC: {ic_err:.5f} | Mass: {mass_err:.5f}")

        # 1. Boolean mask creation for the constrained points (IC at t=0)
        mask_bool = torch.zeros_like(u_exact_i, dtype=torch.bool)
        mask_bool[:, :, 0] = True  # Condizione Iniziale (t=0)

        # Float version of the mask for loss computations in guided methods
        mask_float = mask_bool.float()

        # 2. Loss Function for guided methods (DiffusionPDE, DFlow, PROFlow)
        def composite_loss_fn(u_pred, u_true, mask_tensor):
            data_loss = ((u_pred - u_true) * mask_tensor).square().sum()
            pinn_residual = hfunc(u_pred)
            pinn_loss = (pinn_residual ** 2).sum() * 0.01
            return data_loss + pinn_loss

        constraint = DirichletCondition(value=u_exact_i, mask=mask_bool)
        
        # 5. ECI (Exact Constraint Injection)
        if method_active.get("ECI", False):
            start_t = time.time()
            u_eci = sampler.eci_sample(u0_i, args.n_steps, n_mix=5, resample_step=5, constraint=constraint)
            res_eci = compute_physical_residual(u_eci, eval_hfunc)
            end_t = time.time()
            if not deactivate_if_nan("ECI", res_eci, i):
                record_component_errors("ECI", compute_component_errors(u_eci, u_exact_i, physics_rules, physics_rules_pde))
                mark_success("ECI", u_eci, res_eci, start_t, end_t)

        # 6. DiffusionPDE
        if method_active.get("DiffusionPDE", False):
            start_t = time.time()
            u_diffpde = sampler.guided_sample(
                u0_i, u_exact_i, mask_float, args.n_steps, 
                loss_fn=composite_loss_fn, eta=0.01 
            )
            res_diffpde = compute_physical_residual(u_diffpde, eval_hfunc)
            end_t = time.time()
            if not deactivate_if_nan("DiffusionPDE", res_diffpde, i):
                record_component_errors("DiffusionPDE", compute_component_errors(u_diffpde, u_exact_i, physics_rules, physics_rules_pde))
                mark_success("DiffusionPDE", u_diffpde, res_diffpde, start_t, end_t)

        # 7. D-Flow
        if method_active.get("DFlow", False):
            start_t = time.time()
            u_dflow = sampler.dflow_sample(
                u_exact_i, mask_float, n_sample=1, n_step=args.n_steps, 
                n_iter=20, lr=1, loss_fn=composite_loss_fn
            )
            res_dflow = compute_physical_residual(u_dflow, eval_hfunc)
            end_t = time.time()
            if not deactivate_if_nan("DFlow", res_dflow, i):
                record_component_errors("DFlow", compute_component_errors(u_dflow, u_exact_i, physics_rules, physics_rules_pde))
                mark_success("DFlow", u_dflow, res_dflow, start_t, end_t)

        # 1. Vanilla FM
        if method_active.get("Vanilla", False):
            start_t = time.time()
            u_vanilla = sampler.vanilla_sample(u0_i, args.n_steps)
            res_vanilla = compute_physical_residual(u_vanilla, eval_hfunc)
            end_t = time.time()
            if not deactivate_if_nan("Vanilla", res_vanilla, i):
                record_component_errors("Vanilla", compute_component_errors(u_vanilla, u_exact_i, physics_rules, physics_rules_pde))
                mark_success("Vanilla", u_vanilla, res_vanilla, start_t, end_t)


        # 3. PROFlow
        if method_active.get("PROFlow", False): 
            start_t = time.time()
            u_proflow = sampler.proflow_sample(u0_i, args.n_steps, hfunc, K=3, lr_base=0.01)
            res_proflow = compute_physical_residual(u_proflow, eval_hfunc)
            end_t = time.time()
            if not deactivate_if_nan("PROFlow", res_proflow, i):
                record_component_errors("PROFlow", compute_component_errors(u_proflow, u_exact_i, physics_rules, physics_rules_pde))
                mark_success("PROFlow", u_proflow, res_proflow, start_t, end_t)

        # 4. Original PCFM with the Float64 final projection fix
        if method_active.get("PCFM", False):
            start_t = time.time()
            # Flow Matching con parametri Appendice H
            u_pcfm_i = sampler.pcfm_sample(
                u0_i, args.n_steps, hfunc=hfunc, newtonsteps=1,
                guided_interpolation=True,
                interpolation_params={'custom_lam': 1.0, 'step_size': 0.01, 'num_steps': 20}
            )
            # Final projection Float64
            u_pcfm_final_32 = u_pcfm_i.detach()
            u_flat_64 = u_pcfm_final_32.flatten().unsqueeze(0).to(torch.float64)
            u_pcfm_final_proj = fast_project_batched(u_flat_64, hfunc, max_iter=2)
            u_pcfm = u_pcfm_final_proj.view(u_pcfm_final_32.shape).to(torch.float32).detach()
            
            res_pcfm = compute_physical_residual(u_pcfm, eval_hfunc)
            end_t = time.time()
            if not deactivate_if_nan("PCFM", res_pcfm, i):
                record_component_errors("PCFM", compute_component_errors(u_pcfm, u_exact_i, physics_rules, physics_rules_pde))
                mark_success("PCFM", u_pcfm, res_pcfm, start_t, end_t)


        # 2. Ours (Continuous Guided)
        for g in args.gamma_list:
            tracker_key = f"Ours_g{g}"
            if method_active.get(tracker_key, False):
                start_t = time.time()
                u_ours, _ = sampler.continuous_guided_sample(u0_i, args.n_steps, hfunc, gamma_max=g, final_refinement=True, refinement_steps=200, refinement_lr=0.1, gamma_schedule="sine")
                res_ours = compute_physical_residual(u_ours, eval_hfunc)
                end_t = time.time()
                if not deactivate_if_nan(tracker_key, res_ours, i):
                    record_component_errors(tracker_key, compute_component_errors(u_ours, u_exact_i, physics_rules, physics_rules_pde))
                    mark_success(tracker_key, u_ours, res_ours, start_t, end_t)

        if args.log_every > 0 and ((i + 1) % args.log_every == 0):
            print(f"[WandB] Periodic log at sample {i+1}/{args.n_samples}")
            log_periodic_metrics(i + 1)

    # Ensure at least one periodic log is sent at the end even if n_samples < log_every
    log_periodic_metrics(args.n_samples)


    # 5. Global metrics and Wnadb logging 
    print("\n6. Computing Distribution Metrics and Logging to Wandb...")
    wandb_log_dict = {}
    columns = ["Method", "Success Rate (%)", "Generated Samples", "Speed (sec/sample)", "Physical Residual (MAE)", "IC Error", "BC Left Error", "BC Right Error", "BC Error", "PDE Error (Scaled)", "PDE Error (Raw)", "Mass Error", "CL Error", "MMSE", "SMSE", "SampleMSE"]
    results_table = wandb.Table(columns=columns)

    for name, tracker in trackers.items():
        success_count = method_success_counts[name]
        success_rate = 100.0 * success_count / max(1, args.n_samples)
        failed_with_nan = method_failure_sample[name] is not None

        if (not failed_with_nan) and success_count == args.n_samples:
            tracker.print_summary()
            # Retrieve all generated samples for this method (shape: [n_sample, nx, nt])
            u_pred_all = tracker.get_all_samples_tensor()
            u_exact_ref = u_exact_all

            component_means = get_component_means(name, success_count)
            ce_ic = component_means["ic"]
            ce_bc_left = np.nan
            ce_bc_right = np.nan
            ce_bc = np.nan
            ce_pde_scaled = component_means["pde_scaled"]
            ce_pde_raw = component_means["pde_raw"]
            ce_mass = component_means["mass"]

            mass_t0 = u_pred_all[:, :, 0].mean(dim=1, keepdim=True) # [N, 1]
            mass_all_t = u_pred_all.mean(dim=1)                     # [N, nt]
            ce_cl = (mass_all_t - mass_t0).abs().mean().item()

            # Compute distribution metrics (MMSE, SMSE) against the ground truth
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

        # Populate the wandb log dict and results table
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
        results_table.add_data(name, success_rate, success_count, speed, residual, ce_ic, ce_bc_left, ce_bc_right, ce_bc, ce_pde_scaled, ce_pde_raw, ce_mass, ce_cl, mmse, smse, sample_mse)

    wandb_log_dict["Final_Results_Table"] = results_table
    wandb.log(wandb_log_dict)


    # =========================================================
    # 6. PLOTTING
    # =========================================================
    print("\n6. Drawing random results for comparison...")
    plottable_methods = [
        (name, tracker)
        for name, tracker in trackers.items()
        if method_success_counts[name] > 0
    ]

    if len(plottable_methods) == 0:
        print("No plottable methods: all methods were disabled or produced zero valid samples.")
    else:
        max_common_samples = min(method_success_counts[name] for name, _ in plottable_methods)
        plot_count = min(6, max_common_samples)
        plot_indices = random.sample(range(max_common_samples), plot_count)
        num_rows = 1 + len(plottable_methods)
        fig, axes = plt.subplots(num_rows, len(plot_indices), figsize=(18, 3 * num_rows))

        if num_rows == 1: axes = [axes]
        if len(plot_indices) == 1: axes = [[ax] for ax in axes]

        for col, idx in enumerate(plot_indices):
            # Riga 0: Ground Truth
            axes[0][col].imshow(u_exact_all[idx].numpy(), cmap='bwr', aspect='auto')
            axes[0][col].set_title(f'Exact {idx}')
            axes[0][col].axis('off')

            # Righe successive: Metodi testati
            for row, (name, tracker) in enumerate(plottable_methods, start=1):
                u_plot = tracker.get_all_samples_tensor()[idx].numpy()
                axes[row][col].imshow(u_plot, cmap='bwr', aspect='auto')
                axes[row][col].set_title(f'{name} {idx}')
                axes[row][col].axis('off')

        plt.tight_layout()
        wandb.log({"Comparison_Plot": wandb.Image(fig)})
        plt.savefig("comparison_plot.png")
        print("Plot saved as comparison_plot.png")
    
    wandb.finish()

if __name__ == "__main__":
    main()