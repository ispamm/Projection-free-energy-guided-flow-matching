import argparse
import torch
import time
import random
import wandb
import numpy as np
import matplotlib.pyplot as plt
from scripts.training.utils import load_config
from models import get_flow_model
from pcfm.ffm_sampler import FFM_sampler
from pcfm.pcfm_sampling import make_grid, fast_project_batched
from pcfm.constraints import Residuals
from metrics import compute_physical_residual, compute_speed, compute_distribution_metrics, compute_samplewise_mse, MetricsTracker


try:
    from datasets import get_dataset
    from torch.utils.data import DataLoader
except ImportError:
    print("Warning: Check the import for 'get_dataset' in your repository.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generative PDE Benchmark & Ablation")
    
    # Parametri di Sistema e Modello
    parser.add_argument("--config_path", type=str, default="configs/heat_white.yml", help="Percorso del file config.yml")
    parser.add_argument("--ckpt_path", type=str, default="logs/heat_white_test/latest.pt", help="Percorso del checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda:1", help="GPU da usare (es. cuda:0)")
    parser.add_argument("--seed", type=int, default=42, help="Seed per la generazione del rumore di test")
    
    # Parametri di Esperimento
    parser.add_argument("--models", nargs="+", default=["all"], 
                        help="Baseline da testare: vanilla, pcfm, proflow, ours, all")
    parser.add_argument("--n_steps", type=int, default=100, help="Step di discretizzazione del solutore ODE")
    parser.add_argument("--n_samples", type=int, default=100, help="Dimensione del batch di test")
    
    # Parametri Ablation per "Ours"
    parser.add_argument("--gamma_list", nargs="+", type=float, default=[1.0], 
                        help="Ablation: lista di gamma_max. Es: --gamma_list 0.1 1.0 2.0")
    
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="pcfm-physics-comparison", 
        name=f"eval_{args.n_samples}s_{args.n_steps}step",
        config=vars(args)
    )

    print(f"1. Loading configuration from {args.config_path}...")
    config = load_config(args.config_path)
    
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
    batch = next(iter(test_loader))
    
    # 1. The batch is the real data (u_exact) we want to compare against
    u_exact = batch.to(device)
    # if the tensor has an unnecessary 'channel' dimension (e.g., [3, 1, 100, 100]), we remove it
    actual_samples = u_exact.shape[0]
    if actual_samples < args.n_samples:
        print(f"\n[WARNING] Asked for {args.n_samples} samples, but the test dataset has only {actual_samples}.")
        print(f"Cap down n_samples to {actual_samples} to avoid crashes.\n")
        args.n_samples = actual_samples
        
        # Dobbiamo anche tagliare u0 per assicurarci che abbiano la stessa lunghezza!
        u0 = u0[:actual_samples]
    
    if u_exact.dim() == 4:
        u_exact = u_exact.squeeze(1)
    u_exact_all = u_exact.cpu()  # Keep a copy on CPU for later metric computations
        
    # 2. Build the spatial and temporal grids (x_grid, t_grid) based on the dimensions specified in the config
    x_grid = torch.linspace(0, 1, dims[0], device=device)
    t_grid = torch.linspace(0, 1, dims[1], device=device)

    print("3. Generation of initial noise (u0)...")
    grid = make_grid(dims, device)
    u0 = model.gp.sample(grid, dims, n_samples=args.n_samples).to(device)

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

    print("4. Iterative Generation (to respect the original Residuals class)...")
    sampler = FFM_sampler(model, model.gp)

    
    print("\n5. Start Inference Loop...")
    for i in range(args.n_samples):

        if i % 10 == 0:
             print(f"\n--- Processing Sample {i+1}/{args.n_samples} ---")
                
        # Extract the individual sample while maintaining the fake batch dimension [1, nx, nt]
        u0_i = u0[i:i+1]
        u_exact_i = u_exact[i:i+1]
        
        # Initialize the physics rules ONLY for this specific sample
        physics_rules = Residuals(
            data=u_exact_i, 
            x=x_grid, 
            t_grid=t_grid, 
            nx=dims[0], 
            nt=dims[1]
        )
        hfunc = physics_rules.full_residual_heat
        
        # 1. Vanilla FM
        if "Vanilla" in trackers:
            start_t = time.time()
            u_vanilla = sampler.vanilla_sample(u0_i, args.n_steps)
            res_vanilla = compute_physical_residual(u_vanilla, hfunc)
            trackers["Vanilla"].record_step(u_vanilla, res_vanilla, start_t, time.time())


        # 3. PROFlow
        if "PROFlow" in trackers: 
            start_t = time.time()
            u_proflow = sampler.proflow_sample(u0_i, args.n_steps, hfunc, K=3, lr_base=0.1)
            res_proflow = compute_physical_residual(u_proflow, hfunc)
            trackers["PROFlow"].record_step(u_proflow, res_proflow, start_t, time.time())

        # 4. Original PCFM with the Float64 final projection fix
        if "PCFM" in trackers:
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
            
            res_pcfm = compute_physical_residual(u_pcfm, hfunc)
            trackers["PCFM"].record_step(u_pcfm, res_pcfm, start_t, time.time())


        # 2. Ours (Continuous Guided)
        for g in args.gamma_list:
            tracker_key = f"Ours_g{g}"
            if tracker_key in trackers:    
                start_t = time.time()
                u_ours, _ = sampler.continuous_guided_sample(u0_i, args.n_steps, hfunc, gamma_max=g)
                res_ours = compute_physical_residual(u_ours, hfunc)
                trackers[tracker_key].record_step(u_ours, res_ours, start_t, time.time())


    # 5. Global metrics and Wnadb logging 
    print("\n6. Computing Distribution Metrics and Logging to Wandb...")
    wandb_log_dict = {}
    columns = ["Method", "Speed (sec/sample)", "Physical Residual (MAE)", "MMSE", "SMSE", "SampleMSE"]
    results_table = wandb.Table(columns=columns)
    
    for name, tracker in trackers.items():
        tracker.print_summary()         
        # Retrieve all generated samples for this method (shape: [n_sample, nx, nt])
        u_pred_all = tracker.get_all_samples_tensor()
        
        # COmpute distribution metrics (MMSE, SMSE) against the ground truth
        mmse, smse = compute_distribution_metrics(u_pred_all, u_exact_all)
        sample_mse = compute_samplewise_mse(u_pred_all, u_exact_all)
        speed = tracker.get_average_speed()
        residual = tracker.get_average_residual()
        
        # Populate the wandb log dict and results table
        wandb_log_dict[f"Speed (sec/sample)/{name}"] = speed
        wandb_log_dict[f"Physics_Error (MAE)/{name}"] = residual
        wandb_log_dict[f"Distribution_MMSE/{name}"] = mmse
        wandb_log_dict[f"Distribution_SMSE/{name}"] = smse
        wandb_log_dict[f"Distribution_SampleMSE/{name}"] = sample_mse
        results_table.add_data(name, speed, residual, mmse, smse, sample_mse)

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
        axes[0][col].imshow(u_exact_all[idx].numpy(), cmap='bwr', aspect='auto')
        axes[0][col].set_title(f'Exact {idx}')
        axes[0][col].axis('off')
        
        # Righe successive: Metodi testati
        for row, (name, tracker) in enumerate(trackers.items(), start=1):
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