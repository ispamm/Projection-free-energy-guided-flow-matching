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
from pcfm.ffm_sampler import FFM_NS_sampler, FFM_sampler
from pcfm.pcfm_sampling import make_grid, fast_project_batched
from pcfm.constraints import Residuals, Residuals2D
from metrics import compute_physical_residual, compute_ns_physical_residual, compute_speed, compute_distribution_metrics, compute_samplewise_mse, MetricsTracker


class NavierStokesResiduals:
    def __init__(self, data, f_field, x, y, t_grid, nx, ny, nt, nu):
        device = data.device
        self.data = data.to(device=device, dtype=torch.float32)
        self.f = f_field.to(device=device, dtype=torch.float32)
        self.x, self.y, self.t_grid = (
            x.to(device=device, dtype=torch.float32),
            y.to(device=device, dtype=torch.float32),
            t_grid.to(device=device, dtype=torch.float32),
        )
        self.nx, self.ny, self.nt = nx, ny, nt
        self.nu = float(nu)
        
        # Verifica che i passi non siano zero o troppo piccoli
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dt = self.t_grid[1] - self.t_grid[0]

        if self.dx == 0 or self.dy == 0 or self.dt == 0:
            print(f"!!! CRITICAL ERROR: Grid spacing is zero! dx:{self.dx}, dy:{self.dy}, dt:{self.dt}")

        X, Y = torch.meshgrid(self.x, self.y, indexing='ij')
        self.f_const = (
            0.1
            * torch.sqrt(torch.tensor(2.0, device=device, dtype=torch.float32))
            * (
                torch.sin(2 * torch.pi * (X + Y))
                + torch.cos(2 * torch.pi * (X + Y))
            )
        ).to(device=device, dtype=torch.float32)

    def check_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"--- [!] NaN detected in: {name} ---")
            print(f"    Min: {tensor.min().item():.4e}, Max: {tensor.max().item():.4e}")
            return True
        return False

    def ic_residual(self, u_flat):
        u = u_flat.to(dtype=torch.float32).view(self.nx, self.ny, self.nt)
        target = self.data[0][:, :, 0].to(u.device, dtype=torch.float32)
        res = (u[:, :, 0] - target).flatten()
        self.check_nan(res, "IC Residual")
        return res

    def mass_residual(self, u_flat):
        u = u_flat.to(dtype=torch.float32).view(self.nx, self.ny, self.nt)
        mass_t = u.sum(dim=(0, 1)) * self.dx * self.dy
        res = (mass_t[1:] - mass_t[0]).flatten()
        self.check_nan(res, "Mass Residual")
        return res

    def pde_residual(self, u_flat):
        w = u_flat.to(dtype=torch.float32).view(self.nx, self.ny, self.nt)
        
        # Controllo input
        if self.check_nan(w, "Input u_flat (w)"): return torch.zeros_like(w[1:-1, 1:-1, :]).flatten()

        # 1. Termine Temporale
        w_t = (w[:, :, 1:] - w[:, :, :-1]) / self.dt
        self.check_nan(w_t, "w_t (Temporal)")

        # 2. Termini Spaziali (Laplaciano)
        # Dividere per dx^2 (es. 0.015^2 = 0.000225) amplifica enormemente i valori!
        w_xx = (w[2:, 1:-1, :-1] - 2*w[1:-1, 1:-1, :-1] + w[:-2, 1:-1, :-1]) / (self.dx**2)
        w_yy = (w[1:-1, 2:, :-1] - 2*w[1:-1, 1:-1, :-1] + w[1:-1, :-2, :-1]) / (self.dy**2)
        
        diffusion = self.nu * (w_xx + w_yy)
        self.check_nan(diffusion, "Diffusion term")

        # 3. Forcing
        f_term = self.f[1:-1, 1:-1, None]
        
        res = w_t[1:-1, 1:-1, :] - diffusion - f_term
        self.check_nan(res, "Full PDE Residual")
        
        return res.flatten()

    def full_residual(self, u_flat):
        ic = self.ic_residual(u_flat)
        mass = self.mass_residual(u_flat)
        pde = self.pde_residual(u_flat)
        
        return torch.cat([
            ic, 
            mass, 
            pde
        ], dim=0)


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
    
    # Ablation Parameters for "Ours"
    parser.add_argument("--gamma_list", nargs="+", type=float, default=[1.0], 
                        help="Ablation for gamma values. Es: --gamma_list 0.1 1.0 2.0")
    
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    run_name = args.run_name if args.run_name else f"eval_{args.n_samples}s_{args.n_steps}step"
    
    wandb.init(
        project="pcfm-physics-comparison", 
        name=run_name,
        config=vars(args)
    )

    print(f"1. Loading configuration from {args.config_path}...")
    config = load_config(args.config_path)
    
    print(f"Loading model and weights from {args.ckpt_path} to {device}...")
    model = get_flow_model(config.model, config.encoder).to(device)
    
    # weights_only=False sopprime il FutureWarning di sicurezza (poiché ti fidi dei tuoi ckpt)
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    
    # Caricamento robusto: gestisce sia PyTorch standard che PyTorch Lightning
    if 'state_dict' in ckpt:
        # È un checkpoint di PyTorch Lightning
        state_dict = ckpt['state_dict']
        clean_state_dict = {}
        # Rimuoviamo l'eventuale prefisso "model." o "net." inserito da Lightning
        for k, v in state_dict.items():
            if k.startswith('model.'):
                clean_state_dict[k[6:]] = v
            elif k.startswith('net.'):
                clean_state_dict[k[4:]] = v
            else:
                clean_state_dict[k] = v
        # Strict=False aiuta se ci sono parametri extra salvati da Lightning (es. loss)
        model.load_state_dict(clean_state_dict, strict=False)
        print("-> Successfully loaded PyTorch Lightning checkpoint.")
    elif 'model' in ckpt:
        # È un checkpoint salvato manualmente (PyTorch standard)
        model.load_state_dict(ckpt['model'])
        print("-> Successfully loaded standard PyTorch checkpoint.")
    else:
        # Fallback: il file È il state_dict stesso
        model.load_state_dict(ckpt)
        print("-> Successfully loaded raw state_dict.")
        
    model.eval()

    dims = config.sample_dims
    sampler = FFM_sampler(model, model.gp)

    print("2. Ground Truth Loading...")
    train_set, test_set = get_dataset(config.datasets)
    test_loader = DataLoader(test_set, batch_size=args.n_samples, shuffle=False)
    batch = next(iter(test_loader))
    
    # 1. The batch is the real data (u_exact) we want to compare against
    u_exact = batch.to(device)
    # if the tensor has an unnecessary 'channel' dimension (e.g., [3, 1, 100, 100]), we remove it
    actual_samples = u_exact.shape[0]
    if actual_samples < args.n_samples:
        print(f"\n[WARNING] Asked for {args.n_samples} samples, but the test dataset has only {actual_samples}.")
        print(f"Cap down n_samples to {actual_samples} to avoid crashes.\n")
        args.n_samples = actual_samples
        
    
    if u_exact.dim() == 5:
        u_exact = u_exact.squeeze(1)
    u_exact_all = u_exact.cpu()  # Keep a copy on CPU for later metric computations
    actual_nx = u_exact.shape[1]
    actual_ny = u_exact.shape[2]
    actual_nt = u_exact.shape[3]

    import h5py
    full_h5_path = os.path.join(config.datasets.root, config.datasets.test.data_file)
    
    print(f"Opening H5 file for forcing: {full_h5_path}")
    with h5py.File(full_h5_path, 'r') as h5_file:
        f_all = torch.from_numpy(h5_file['f'][:]).float() 
        f_all = f_all.reshape(-1, actual_nx, actual_ny)
        
    # 2. Build the spatial and temporal grids (x_grid, t_grid) based on the dimensions specified in the config
    x_grid = torch.linspace(0, 1, actual_nx, device=device)
    y_grid = torch.linspace(0, 1, actual_ny, device=device)
    t_grid = torch.linspace(0, 1, actual_nt, device=device)

    actual_dims = [actual_nx, actual_ny, actual_nt]

    print("3. Generation of initial noise (u0)...")
    grid = make_grid(actual_dims, device)
    u0 = model.gp.sample(grid, actual_dims, n_samples=args.n_samples).to(device)

    print("4. Setup Trackers...")
    trackers = {}
    
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

    dx_val = (x_grid[1] - x_grid[0]).item()
    dy_val = (y_grid[1] - y_grid[0]).item()
    dt_val = (t_grid[1] - t_grid[0]).item()

    print("4. Iterative Generation (to respect the original Residuals class)...")
    sampler = FFM_NS_sampler(model)
    #sampler = FFM_sampler(model, model.gp)
    
    print("\n5. Start Inference Loop...")
    for i in range(args.n_samples):

        if i % 10 == 0:
             print(f"\n--- Processing Sample {i+1}/{args.n_samples} ---")
                
        # Extract the individual sample while maintaining the fake batch dimension [1, nx, nt]
        u0_i = u0[i:i+1]
        u_exact_i = u_exact[i:i+1]
        nf_unique = f_all.shape[0] 
        f_i = f_all[i % nf_unique]
        
        # Initialize the physics rules ONLY for this specific sample
        physics_rules = NavierStokesResiduals(
            data=u_exact_i, 
            f_field=f_i,
            x=x_grid, 
            y=y_grid, 
            t_grid=t_grid, 
            nx=actual_nx, ny=actual_ny, nt=actual_nt,
            nu=0.001 
        )

        # physics_rules = Residuals2D(
        #     data=u_exact_i, 
        #     x=x_grid, 
        #     y=y_grid, 
        #     t_grid=t_grid, 
        #     nx=actual_nx, 
        #     ny=actual_ny, 
        #     nt=actual_nt,
        #     nu=0.001, # Viscosità
        #     rho=1.0   # Densità (default nel paper)
        # )

        hfunc = physics_rules.full_residual
        
        # 1. Boolean mask creation for the constrained points (t=0, x=0, x=L)
        mask_bool = torch.zeros_like(u_exact_i, dtype=torch.bool)
        mask_bool[:, :, :, 0] = True  # Condizione Iniziale (t=0)
        #mask_bool[:, 0, :] = True  # Bordo sinistro (x=0)
        #mask_bool[:, -1, :] = True # Bordo destro (x=L)
        
        # Float version of the mask for loss computations in guided methods
        mask_float = mask_bool.float()
        
        # 2. Loss Function for guided methods (DiffusionPDE, DFlow, PROFlow)
        def composite_loss_fn(u_pred, u_true, mask_tensor):
            # Data term (IC/BC)
            data_loss = ((u_pred - u_true) * mask_tensor).square().sum()
            
            # PINN term (Physics) weighted at 1e-2 as per the paper
            # Calculate the residual and take its loss (MSE)
            pinn_residual = hfunc(u_pred)
            pinn_loss = (pinn_residual ** 2).sum() * 0.01
            
            return data_loss + pinn_loss

        constraint = DirichletCondition(value=u_exact_i, mask=mask_bool)
        
        # 5. ECI (Exact Constraint Injection)
        if "ECI" in trackers:
            start_t = time.time()
            u_eci = sampler.eci_sample(u0_i, args.n_steps, n_mix=5, resample_step=5, constraint=constraint)
            res_eci = compute_ns_physical_residual(u_eci, f_i, dx=dx_val, dy=dy_val, dt=dt_val)
            trackers["ECI"].record_step(u_eci, res_eci, start_t, time.time())

        # 6. DiffusionPDE
        if "DiffusionPDE" in trackers:
            start_t = time.time()
            u_diffpde = sampler.guided_sample(
                u0_i, u_exact_i, mask_float, args.n_steps, 
                loss_fn=composite_loss_fn, eta=0.01 
            )
            res_diffpde = compute_ns_physical_residual(u_diffpde, f_i, dx=dx_val, dy=dy_val, dt=dt_val)
            trackers["DiffusionPDE"].record_step(u_diffpde, res_diffpde, start_t, time.time())

        # 7. D-Flow
        if "DFlow" in trackers:
            start_t = time.time()
            u_dflow = sampler.dflow_ns_sample(
                u_exact_i, mask_float, n_sample=1, n_step=args.n_steps, 
                n_iter=10, lr=0.01, loss_fn=composite_loss_fn
            )
            res_dflow = compute_ns_physical_residual(u_dflow, f_i, dx=dx_val, dy=dy_val, dt=dt_val)
            trackers["DFlow"].record_step(u_dflow, res_dflow, start_t, time.time())

        # 1. Vanilla FM
        if "Vanilla" in trackers:
            start_t = time.time()
            u_vanilla = sampler.vanilla_sample(u0_i, args.n_steps)
            res_vanilla = compute_ns_physical_residual(u_vanilla, f_i, dx=dx_val, dy=dy_val, dt=dt_val)
            trackers["Vanilla"].record_step(u_vanilla, res_vanilla, start_t, time.time())


        # 3. PROFlow
        if "PROFlow" in trackers: 
            start_t = time.time()
            u_proflow = sampler.proflow_sample(u0_i, args.n_steps, hfunc, K=3, lr_base=0.1)
            res_proflow = compute_ns_physical_residual(u_proflow, f_i, dx=dx_val, dy=dy_val, dt=dt_val)
            trackers["PROFlow"].record_step(u_proflow, res_proflow, start_t, time.time())

        # 4. Original PCFM with the Float64 final projection fix
        if "PCFM" in trackers:
            start_t = time.time()
            # Flow Matching con parametri Appendice H
            u_pcfm_i = sampler.pcfm_sample(
                u0_i, args.n_steps, hfunc=hfunc, newtonsteps=1,
                guided_interpolation=False,
                interpolation_params={'custom_lam': 1.0, 'step_size': 0.01, 'num_steps': 20}
            )
            # Final projection Float64
            u_pcfm_final_32 = u_pcfm_i.detach()
            u_flat_64 = u_pcfm_final_32.flatten().unsqueeze(0).to(torch.float64)
            u_pcfm_final_proj = fast_project_batched(u_flat_64, hfunc, max_iter=2)
            u_pcfm = u_pcfm_final_proj.view(u_pcfm_final_32.shape).to(torch.float32).detach()
            
            res_pcfm = compute_ns_physical_residual(u_pcfm, f_i, dx=dx_val, dy=dy_val, dt=dt_val)
            trackers["PCFM"].record_step(u_pcfm, res_pcfm, start_t, time.time())


        # 2. Ours (Continuous Guided)
        for g in args.gamma_list:
            tracker_key = f"Ours_g{g}"
            if tracker_key in trackers:    
                start_t = time.time()
                u_ours, _ = sampler.continuous_guided_sample(u0_i, args.n_steps, hfunc, gamma_max=g, final_refinement=False, refinement_steps=1, refinement_lr=0.2)
                res_ours = compute_ns_physical_residual(u_ours, f_i, dx=dx_val, dy=dy_val, dt=dt_val)
                trackers[tracker_key].record_step(u_ours, res_ours, start_t, time.time())


    # 5. Global metrics and Wnadb logging 
    print("\n6. Computing Distribution Metrics and Logging to Wandb...")
    wandb_log_dict = {}
    columns = ["Method", "Speed (sec/sample)", "Physical Residual (MAE)", "IC Error", "BC Error", "CL Error", "MMSE", "SMSE", "SampleMSE"]
    results_table = wandb.Table(columns=columns)
    
    mask_bc = torch.zeros_like(u_exact_all, dtype=torch.bool)
    mask_bc[:, 0, :, :] = True
    mask_bc[:, -1, :, :] = True
    mask_bc[:, :, 0, :] = True
    mask_bc[:, :, -1, :] = True

    for name, tracker in trackers.items():
        tracker.print_summary()         
        # Retrieve all generated samples for this method (shape: [n_sample, nx, nt])
        u_pred_all = tracker.get_all_samples_tensor()
        
        ce_ic = torch.nn.functional.mse_loss(
            u_pred_all[:, :, 0], 
            u_exact_all[:, :, 0]
        ).item()
        
        ce_bc = torch.nn.functional.mse_loss(
            u_pred_all[mask_bc], 
            u_exact_all[mask_bc]
        ).item()

        mass_t0 = u_pred_all[:, :, :, 0].mean(dim=(1, 2), keepdim=True) # [N, 1]
        mass_all_t = u_pred_all.mean(dim=(1, 2))                     # [N, nt]
        ce_cl = (mass_all_t - mass_t0).abs().mean().item()

        # Compute distribution metrics (MMSE, SMSE) against the ground truth
        mmse, smse = compute_distribution_metrics(u_pred_all, u_exact_all)
        sample_mse = compute_samplewise_mse(u_pred_all, u_exact_all)
        speed = tracker.get_average_speed()
        residual = tracker.get_average_residual()

        save_dir = f"results/{run_name}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(u_pred_all.cpu(), os.path.join(save_dir, f"{name}_tensors.pt"))
        
        # Populate the wandb log dict and results table
        wandb_log_dict[f"Speed (sec/sample)/{name}"] = speed
        wandb_log_dict[f"Physics_Error (MAE)/{name}"] = residual
        wandb_log_dict[f"Physics_Error_IC/{name}"] = ce_ic
        wandb_log_dict[f"Physics_Error_BC/{name}"] = ce_bc
        wandb_log_dict[f"Physics_Error_CL/{name}"] = ce_cl
        wandb_log_dict[f"Distribution_MMSE/{name}"] = mmse
        wandb_log_dict[f"Distribution_SMSE/{name}"] = smse
        wandb_log_dict[f"Distribution_SampleMSE/{name}"] = sample_mse
        results_table.add_data(name, speed, residual, ce_ic, ce_bc, ce_cl, mmse, smse, sample_mse)

    wandb_log_dict["Final_Results_Table"] = results_table
    wandb.log(wandb_log_dict)


    # =========================================================
    # 6. PLOTTING
    # =========================================================
    print("\n6. Drawing random results for comparison...")
    plot_indices = random.sample(range(args.n_samples), min(6, args.n_samples))
    num_rows = 1 + len(trackers)
    fig, axes = plt.subplots(num_rows, len(plot_indices), figsize=(18, 3 * num_rows))

    if num_rows == 1: axes = [axes]
    if len(plot_indices) == 1: axes = [[ax] for ax in axes]
    
    for col, idx in enumerate(plot_indices):
        # Riga 0: Ground Truth
        axes[0][col].imshow(u_exact_all[idx, :, :, -1].numpy(), cmap='magma', origin='lower')
        axes[0][col].set_title(f'Exact {idx} (t=End)')
        axes[0][col].axis('off')
        
        # Righe successive: Metodi testati
        for row, (name, tracker) in enumerate(trackers.items(), start=1):
            u_plot = tracker.get_all_samples_tensor()[idx, :, :, -1].numpy()
            axes[row][col].imshow(u_plot, cmap='magma', origin='lower')
            axes[row][col].set_title(f'{name} {idx}')
            axes[row][col].axis('off')

    plt.tight_layout()
    wandb.log({"Comparison_Plot": wandb.Image(fig)})
    plt.savefig("comparison_plot.png")
    print("Plot saved as comparison_plot.png")
    
    wandb.finish()

if __name__ == "__main__":
    main()