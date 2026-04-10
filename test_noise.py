import torch
import matplotlib.pyplot as plt

# Parametri griglia e kernel
nx = 100
x = torch.linspace(0, 1, nx)
length_scale = 0.001

# Costruiamo la matrice di covarianza (Matern nu=0.5 è l'Esponenziale)
dist_matrix = torch.abs(x.unsqueeze(1) - x.unsqueeze(0))
cov_matrix = torch.exp(-dist_matrix / length_scale)

# Generiamo un campione usando la decomposizione di Cholesky
L = torch.linalg.cholesky(cov_matrix + 1e-6 * torch.eye(nx)) # 1e-6 per stabilità
z = torch.randn(nx)
matern_sample = L @ z

# Generiamo un campione di puro rumore bianco
white_noise = torch.randn(nx)

# --- PLOT ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Matrice di Covarianza
im = axes[0].imshow(cov_matrix.numpy(), cmap='hot')
axes[0].set_title("Covariance Matrix (Length=0.001)")
plt.colorbar(im, ax=axes[0])

# 2. Il loro "Matern"
axes[1].plot(x.numpy(), matern_sample.numpy(), color='blue')
axes[1].set_title("Matern Sample (nu=0.5, l=0.001)")

# 3. Puro White Noise
axes[2].plot(x.numpy(), white_noise.numpy(), color='red')
axes[2].set_title("Pure torch.randn()")

plt.tight_layout()
plt.savefig("noise_comparison.png")
print("Grafico salvato in noise_comparison.png!")