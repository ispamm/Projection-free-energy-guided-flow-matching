import torch
import time
import numpy as np

# ==========================================
# 1. PHYSICAL CONSISTENCY AND TIME METRICS
# ==========================================

def compute_physical_residual(u_pred, hfunc):
    """
    Compute the physical residual (Constraint Error) for a sample,
    exactly as implemented in your original script.
    """
    with torch.no_grad():
        # Flatten tensor for `hfunc`; return mean absolute residual (MAE)
        residual = hfunc(u_pred.flatten())
        error = residual.abs().mean().item()
    return error

def compute_ns_physical_residual(u_pred, f_i, nu=0.001, dx=1/64, dy=1/64, dt=1.0):
    """
    Compute the PDE residual, ensuring inputs and tensors are on the same device.
    Returns the mean absolute residual over the interior grid points.
    """
    with torch.no_grad():
        
        u = u_pred.squeeze()
        if u.dim() == 4:
            u = u[0]
            
        device = u.device 
        u = u.to(torch.float64)
        f = f_i.squeeze().to(device=device, dtype=torch.float64)
        
        if not (u.shape[0] == u.shape[1]):
            u = u.permute(1, 2, 0)

        u_t = (u[:, :, 1:] - u[:, :, :-1]) / dt
        
        u_xx = (u[2:, 1:-1, :-1] - 2*u[1:-1, 1:-1, :-1] + u[:-2, 1:-1, :-1]) / (dx**2)
        u_yy = (u[1:-1, 2:, :-1] - 2*u[1:-1, 1:-1, :-1] + u[1:-1, :-2, :-1]) / (dy**2)
        diffusion = nu * (u_xx + u_yy)
        forcing = f[1:-1, 1:-1].unsqueeze(-1)
        pde_res = u_t[1:-1, 1:-1, :] - diffusion - forcing
        
        return pde_res.abs().mean().item()

def compute_speed(total_time_seconds, num_samples):
    """
    Compute the inference speed in seconds per sample (sec/sample).
    If you pass the time of an entire batch, divide by the batch size.
    """
    return total_time_seconds / num_samples


# ==========================================
# 2. DISTRIBUTION METRICS
# ==========================================

def compute_distribution_metrics(u_pred_batch, u_true_batch):
    """
    Compute MMSE and SMSE between the generated and true distributions.
    Both tensors should have shape [Batch, N_x, N_t]
    """
    with torch.no_grad():
        # Average along the batch dimension
        mu_pred = u_pred_batch.mean(dim=0)
        mu_true = u_true_batch.mean(dim=0)
        
        # Standard deviation along the batch dimension
        std_pred = u_pred_batch.std(dim=0)
        std_true = u_true_batch.std(dim=0)
        
        # MMSE: Mean of Mean Squared Error
        mmse = torch.nn.functional.mse_loss(mu_pred, mu_true).item()
        
        # SMSE: Standard deviation Mean Squared Error
        smse = torch.nn.functional.mse_loss(std_pred, std_true).item()
        
    return mmse, smse

def compute_samplewise_mse(u_pred_batch, u_true_batch):
    """
    Compute the classic MSE point-wise between the entire predicted batch and the ground truth.
    """
    with torch.no_grad():
        mse = torch.nn.functional.mse_loss(u_pred_batch, u_true_batch).item()
    return mse


# ==========================================
# 3. UTILITY CLASS FOR TRACKING
# ==========================================

class MetricsTracker:
    """
    Support class for accumulating the results of each individual method
    during the for loop, without having to create dozens of separate lists.
    """
    def __init__(self, name="Method"):
        self.name = name
        self.times = []
        self.residuals = []
        self.samples = []
        
    def record_step(self, sample, residual_val, start_time, end_time, batch_size=1):
        """Store the sample and residual; record time per sample."""
        # Record elapsed time per sample for this step
        self.times.append((end_time - start_time) / batch_size)
        self.residuals.append(residual_val)
        self.samples.append(sample.cpu())  # Move to CPU for later analysis if needed
        
    def get_average_speed(self):
        """Returns the average seconds per sample."""
        return np.mean(self.times)
        
    def get_average_residual(self):
        """Returns the average physical residual."""
        return np.mean(self.residuals)
    
        
    def get_all_samples_tensor(self):
        """Concatenate all samples into a single large tensor [N_tot, N_x, N_t]."""
        return torch.cat(self.samples, dim=0)
    
    def print_summary(self):
        print(f"\n--- {self.name} Summary ---")
        print(f"Speed:             {self.get_average_speed():.4f} sec/sample")
        print(f"Physical Residual: {self.get_average_residual():.5f}")