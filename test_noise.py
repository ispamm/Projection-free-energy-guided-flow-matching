import numpy as np
import matplotlib.pyplot as plt

# Impostazioni estetiche per paper
plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

def compute_2d_points(N=50):
    """Genera griglia di punti 2D standardizzata [0,1]x[0,1]"""
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    # Appiattisce i punti per il calcolo della matrice di covarianza (N^2, 2)
    return np.vstack([X.ravel(), Y.ravel()]).T, X, Y, N

def squared_exponential_kernel_2d(pts, length_scale):
    """Calcola matrice di covarianza SE per punti 2D"""
    # Calcolo robusto delle distanze euclidee quadrate
    sqdist = np.sum(pts**2, 1).reshape(-1, 1) + np.sum(pts**2, 1) - 2 * np.dot(pts, pts.T)
    # Assicura che non ci siano valori negativi infinitesimali per precisione numerica
    sqdist = np.maximum(sqdist, 0)
    return np.exp(-.5 * (1/length_scale**2) * sqdist)

def matern_kernel_2d(pts, length_scale, nu=1.5):
    """Calcola matrice di covarianza Matérn (nu=3/2) per punti 2D"""
    sqdist = np.sum(pts**2, 1).reshape(-1, 1) + np.sum(pts**2, 1) - 2 * np.dot(pts, pts.T)
    dist = np.sqrt(np.maximum(sqdist, 1e-12))
    factor = np.sqrt(3) * dist / length_scale
    return (1. + factor) * np.exp(-factor)

def sample_grf(K):
    """Campiona un campo gaussiano data la matrice K"""
    N_total = K.shape[0]
    jitter = 1e-5 * np.eye(N_total) # Stabilizzazione numerica per Cholesky
    try:
        L = np.linalg.cholesky(K + jitter)
    except np.linalg.LinAlgError:
        # Se fallisce, tenta un jitter leggermente più aggressivo
        L = np.linalg.cholesky(K + 1e-4 * np.eye(N_total))
    
    z = np.random.normal(size=(N_total, 1))
    return np.dot(L, z)

# --- SETUP ---
np.random.seed(123) # Seed per riproducibilità
N_grid = 50 # Risoluzione della griglia visiva (50x50 = 2500 punti)
pts, X_grid, Y_grid, _ = compute_2d_points(N=N_grid)

print("Generazione campioni dei Prior 2D...")

# 1. White Noise Prior (Uncorrelated Gaussian)
# Campioniamo direttamente senza covarianza per vero White Noise
sample_white = np.random.normal(size=(N_grid, N_grid))

# 2. Matérn Prior (Heat Equation, l = 0.2)
K_matern = matern_kernel_2d(pts, length_scale=0.2)
sample_matern_flat = sample_grf(K_matern)
sample_matern = sample_matern_flat.reshape(N_grid, N_grid)

# 3. Squared Exponential Prior (RD/Burgers, l = 0.05)
K_se = squared_exponential_kernel_2d(pts, length_scale=0.05)
sample_se_flat = sample_grf(K_se)
sample_se = sample_se_flat.reshape(N_grid, N_grid)

# --- PLOTTING ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

# Colormap per divergenza (RdBu_r è ottima per la fisica)
cmap_style = 'RdBu_r'

# Impostiamo vmin/vmax unificati per confronto onesto
vmin_val = -3.0
vmax_val = 3.0

# a) White Noise
im0 = axes[0].imshow(sample_white, cmap=cmap_style, vmin=vmin_val, vmax=vmax_val,
                     origin='lower', extent=[0, 1, 0, 1], interpolation='nearest')
axes[0].set_title("a) White Noise Prior\n(Uncorrelated Pixels)", fontweight='bold')
axes[0].set_xlabel("Spatial $x$")
axes[0].set_ylabel("Spatial $y$")

# b) Matérn
im1 = axes[1].imshow(sample_matern, cmap=cmap_style, vmin=vmin_val, vmax=vmax_val,
                     origin='lower', extent=[0, 1, 0, 1], interpolation='bilinear')
axes[1].set_title("b) Matérn Kernel Prior\n(Heat Eq, $\ell = 0.2$)", fontweight='bold')
axes[1].set_xlabel("Spatial $x$")
# axes[1].set_ylabel("Spatial $y$") # Nascondiamo per pulizia

# c) Squared Exponential
im2 = axes[2].imshow(sample_se, cmap=cmap_style, vmin=vmin_val, vmax=vmax_val,
                     origin='lower', extent=[0, 1, 0, 1], interpolation='bilinear')
axes[2].set_title("c) Squared Exponential Kernel\n(RD/Burgers Eq, $\ell = 0.05$)", fontweight='bold')
axes[2].set_xlabel("Spatial $x$")
# axes[2].set_ylabel("Spatial $y$") # Nascondiamo per pulizia

# Rimuoviamo i tick Y interni per pulizia visiva
axes[1].set_yticks([])
axes[2].set_yticks([])

# Aggiungiamo colorbar unificata
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70]) # [left, bottom, width, height]
cbar = fig.colorbar(im0, cax=cbar_ax)
cbar.set_label("Amplitude $u_0(\\mathbf{x})$")

plt.savefig("generative_priors_comparison_2d.pdf", format='pdf', bbox_inches='tight')
print("Immagine salvata come generative_priors_comparison_2d.pdf")