import torch
import math

def _reshape_to_2d(data, nx, nt):
    """Robustly reshape data to [nx, nt] handling various input shapes."""
    data = data.to(dtype=torch.float32)
    original_shape = tuple(data.shape)
    
    # Remove batch dimension if present (leading dimension of 1)
    if data.dim() >= 3 and data.shape[0] == 1:
        data = data.squeeze(0)
    
    # Remove singleton dimensions from the front
    while data.dim() > 1 and data.shape[0] == 1:
        data = data.squeeze(0)
    
    # Now handle 1D and 2D cases
    if data.dim() == 1:
        # 1D tensor - try to reshape to [nx, nt]
        if data.numel() != nx * nt:
            raise ValueError(
                f"Input shape {original_shape} -> 1D tensor with {data.numel()} elements, "
                f"but nx*nt={nx}*{nt}={nx*nt}. Cannot reshape to [{nx}, {nt}]."
            )
        return data.view(nx, nt)
    
    elif data.dim() == 2:
        # 2D tensor - check if it matches [nx, nt] or [nt, nx]
        if data.shape == (nx, nt):
            return data
        elif data.shape == (nt, nx):
            return data.T
        elif data.numel() == nx * nt:
            # Different shape but correct total elements - try reshape
            return data.reshape(nx, nt)
        else:
            raise ValueError(
                f"Input shape {original_shape} -> 2D tensor has shape {tuple(data.shape)} with {data.numel()} elements, "
                f"expected [{nx}, {nt}] (total {nx*nt}) or [{nt}, {nx}]."
            )
    
    else:
        raise ValueError(
            f"Input shape {original_shape}: Expected 1D, 2D, or 3D tensor. Got {data.dim()}D."
        )


class HeatEquationResidualsFull:
    def __init__(self, data, nx=100, nt=100, nu=0.01, spatial_domain=(0.0, 2.0 * math.pi), time_domain=(0.0, 1.0)):
        self.device = data.device
        self.nx = nx
        self.nt = nt
        self.nu = float(nu)
        self.spatial_domain = tuple(spatial_domain)
        self.time_domain = tuple(time_domain)
        
        # Robustly reshape input to [nx, nt]
        data_reshaped = _reshape_to_2d(data, nx, nt)
        self.data = data_reshaped.to(device=self.device, dtype=torch.float32)
        
        # Diffusion dataset uses periodic x-grid with endpoint excluded and full [t0, t1] time range.
        self.dx = (self.spatial_domain[1] - self.spatial_domain[0]) / self.nx
        self.dt = (self.time_domain[1] - self.time_domain[0]) / (self.nt - 1)

    def ic_residual(self, u_flat):
        u = u_flat.to(dtype=torch.float32).view(self.nx, self.nt)
        target = self.data[:, 0].to(u.device) 
        return (u[:, 0] - target).flatten()

    def mass_residual(self, u_flat):
        u = u_flat.to(dtype=torch.float32).view(self.nx, self.nt)
        mass_t = u.sum(dim=0) * self.dx 
        return (mass_t[1:] - mass_t[0]).flatten()

    def pde_residual(self, u_flat):
        u = u_flat.to(dtype=torch.float32).view(self.nx, self.nt)
        
        u_t = (u[:, 1:] - u[:, :-1]) / self.dt
        u_xx = torch.zeros_like(u[:, :-1])
        u_xx[1:-1, :] = (u[2:, :-1] - 2*u[1:-1, :-1] + u[:-2, :-1]) / (self.dx**2)
        u_xx[0, :] = (u[1, :-1] - 2*u[0, :-1] + u[-1, :-1]) / (self.dx**2)
        u_xx[-1, :] = (u[0, :-1] - 2*u[-1, :-1] + u[-2, :-1]) / (self.dx**2)
        
        res = u_t - self.nu * u_xx
        return res.flatten()

    def full_residual(self, u_flat):
        return torch.cat([
            self.ic_residual(u_flat), 
            self.mass_residual(u_flat), 
            self.pde_residual(u_flat)
        ], dim=0)


class HeatEquationResidualsFullPDE(HeatEquationResidualsFull):
    """Heat residual with IC, mass conservation, and PDE term for full guidance."""

    def __init__(self, data, nx=100, nt=100, nu=0.01, spatial_domain=(0.0, 2.0 * math.pi), time_domain=(0.0, 1.0), pde_scale=None):
        super().__init__(
            data=data,
            nx=nx,
            nt=nt,
            nu=nu,
            spatial_domain=spatial_domain,
            time_domain=time_domain,
        )
        self.pde_scale = float(self.dx ** 2 if pde_scale is None else pde_scale)

    def pde_residual_scaled(self, u_flat):
        return self.pde_residual(u_flat) * self.pde_scale

    def full_residual(self, u_flat):
        return torch.cat([
            self.ic_residual(u_flat),
            self.mass_residual(u_flat),
            self.pde_residual_scaled(u_flat),
        ], dim=0)

    def full_residual_unscaled(self, u_flat):
        return torch.cat([
            self.ic_residual(u_flat),
            self.mass_residual(u_flat),
            self.pde_residual(u_flat),
        ], dim=0)

    def __call__(self, u_flat):
        return self.full_residual(u_flat)


class ReactionDiffusionResidualsFull:
    """Reaction-Diffusion residuals aligned with the generated RD dataset."""

    def __init__(
        self,
        data,
        nx=128,
        nt=100,
        nu=0.005,
        rho=0.01,
        x_grid=None,
        t_grid=None,
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 0.99),
        pde_scale=None,
    ):
        self.device = data.device
        self.nx = nx
        self.nt = nt
        self.nu = float(nu)
        self.rho = float(rho)

        data_reshaped = _reshape_to_2d(data, nx, nt)
        self.data = data_reshaped.to(device=self.device, dtype=torch.float32)

        if x_grid is None:
            self.x_grid = torch.linspace(spatial_domain[0], spatial_domain[1], nx, device=self.device)
        else:
            self.x_grid = x_grid.to(device=self.device, dtype=torch.float32)

        if t_grid is None:
            self.t_grid = torch.linspace(time_domain[0], time_domain[1], nt, device=self.device)
        else:
            self.t_grid = t_grid.to(device=self.device, dtype=torch.float32)

        self.dx = (self.x_grid[1] - self.x_grid[0]).item()
        self.dt = (self.t_grid[1] - self.t_grid[0]).item()
        self.pde_scale = float(self.dx ** 2 if pde_scale is None else pde_scale)

    def ic_residual(self, u_flat):
        u = u_flat.to(dtype=torch.float32).view(self.nx, self.nt)
        return (u[:, 0] - self.data[:, 0]).flatten()

    def _boundary_fluxes(self, u):
        # Match the same 4th-order one-sided flux approximation used in pcfm.constraints.Residuals.mass_residual_rd
        gL_t = -self.nu * (-25 * u[0, :] + 48 * u[1, :] - 36 * u[2, :] + 16 * u[3, :] - 3 * u[4, :]) / (12 * self.dx)
        gR_t = -self.nu * (25 * u[-1, :] - 48 * u[-2, :] + 36 * u[-3, :] - 16 * u[-4, :] + 3 * u[-5, :]) / (12 * self.dx)
        return gL_t, gR_t

    def bc_left_flux(self, u_flat):
        u = u_flat.to(dtype=torch.float32).view(self.nx, self.nt)
        gL_t, _ = self._boundary_fluxes(u)
        return gL_t

    def bc_right_flux(self, u_flat):
        u = u_flat.to(dtype=torch.float32).view(self.nx, self.nt)
        _, gR_t = self._boundary_fluxes(u)
        return gR_t

    def bc_left_residual(self, u_flat):
        u = u_flat.to(dtype=torch.float32).view(self.nx, self.nt)
        gL_t, _ = self._boundary_fluxes(u)
        gL_ref, _ = self._boundary_fluxes(self.data)
        return gL_t - gL_ref

    def bc_right_residual(self, u_flat):
        u = u_flat.to(dtype=torch.float32).view(self.nx, self.nt)
        _, gR_t = self._boundary_fluxes(u)
        _, gR_ref = self._boundary_fluxes(self.data)
        return gR_t - gR_ref

    def bc_residual(self, u_flat):
        return torch.cat([self.bc_left_residual(u_flat), self.bc_right_residual(u_flat)], dim=0)

    def mass_residual(self, u_flat):
        # Integral balance: mass(t) = mass(0) + integrated reaction + integrated boundary flux
        u = u_flat.to(dtype=torch.float32).view(self.nx, self.nt)
        mass = u.sum(dim=0) * self.dx

        source = self.rho * (u * (1.0 - u)).sum(dim=0) * self.dx
        source_mid = 0.5 * (source[:-1] + source[1:])
        dt_vec = (self.t_grid[1:] - self.t_grid[:-1]).to(u.device)
        source_cum = torch.cat([torch.zeros(1, device=u.device), torch.cumsum(source_mid * dt_vec, dim=0)], dim=0)

        gL_t, gR_t = self._boundary_fluxes(u)
        flux = gL_t - gR_t
        flux_mid = 0.5 * (flux[:-1] + flux[1:])
        flux_cum = torch.cat([torch.zeros(1, device=u.device), torch.cumsum(flux_mid * dt_vec, dim=0)], dim=0)

        return mass - (mass[0] + source_cum + flux_cum)

    def pde_residual(self, u_flat):
        u = u_flat.to(dtype=torch.float32).view(self.nx, self.nt)

        u_t = (u[:, 1:] - u[:, :-1]) / self.dt
        u_xx = (u[2:, :-1] - 2.0 * u[1:-1, :-1] + u[:-2, :-1]) / (self.dx ** 2)
        reaction = self.rho * u[1:-1, :-1] * (1.0 - u[1:-1, :-1])
        res = u_t[1:-1, :] - self.nu * u_xx - reaction
        return res.flatten()

    def pde_residual_scaled(self, u_flat):
        return self.pde_residual(u_flat) * self.pde_scale

    def full_residual(self, u_flat):
        return torch.cat([
            self.ic_residual(u_flat),
            self.bc_residual(u_flat),
            self.pde_residual_scaled(u_flat),
            self.mass_residual(u_flat)[1:],
        ], dim=0)

    def full_residual_unscaled(self, u_flat):
        return torch.cat([
            self.ic_residual(u_flat),
            self.bc_residual(u_flat),
            self.pde_residual(u_flat),
            self.mass_residual(u_flat)[1:],
        ], dim=0)


class BurgersEquationResidualsFull:
    def __init__(self, data, nx=100, nt=100, nu=0.01, spatial_domain=[0, 2*math.pi], time_domain=[0, 1]):
        self.device = data.device
        self.nx = nx
        self.nt = nt
        self.nu = float(nu)
        
        # Robustly reshape input to [nx, nt]
        data_reshaped = _reshape_to_2d(data, nx, nt)
        self.data = data_reshaped.to(device=self.device, dtype=torch.float32)
        
        # Calcolo dinamico dei passi spaziotemporali (come in constraints_pcfm.py)
        self.dx = (spatial_domain[1] - spatial_domain[0]) / (self.nx - 1)
        self.dt = (time_domain[1] - time_domain[0]) / (self.nt - 1)

    def ic_residual(self, u):
        """Residuo Condizione Iniziale (t=0)"""
        # Accetta input 1D/2D/3D e normalizza a [nx, nt]
        u = _reshape_to_2d(u, self.nx, self.nt).to(device=self.device)
        target = self.data[:, 0]
        return (u[:, 0] - target).flatten()

    def bc_residual(self, u):
        """Residuo Condizioni al Contorno (Dirichlet a sx, Neumann a dx)"""
        # Accetta input 1D/2D/3D e normalizza a [nx, nt]
        u = _reshape_to_2d(u, self.nx, self.nt).to(device=self.device)
        # Dirichlet a sinistra: u(0, t) = target_left
        left_target = self.data[0, :]
        res_left = u[0, :] - left_target
        res_right = u[-1, :] - u[-2, :]
        
        return torch.cat([res_left, res_right]).flatten()

    def bc_left_residual(self, u):
        """Residuo della sola condizione al contorno sinistra (Dirichlet)."""
        u = _reshape_to_2d(u, self.nx, self.nt).to(device=self.device)
        left_target = self.data[0, :]
        return (u[0, :] - left_target).flatten()

    def bc_right_residual(self, u):
        """Residuo della sola condizione al contorno destra (Neumann zero-gradient)."""
        u = _reshape_to_2d(u, self.nx, self.nt).to(device=self.device)
        return (u[-1, :] - u[-2, :]).flatten()

    def pde_residual(self, u):
        """Residuo Equazione di Burgers: u_t + u*u_x = nu*u_xx"""
        # Accetta input 1D/2D/3D e normalizza a [nx, nt]
        u = _reshape_to_2d(u, self.nx, self.nt).to(device=self.device)
        
        # Derivata temporale (Eulero avanti)
        u_t = (u[:, 1:] - u[:, :-1]) / self.dt
        
        # Derivate spaziali (Differenze centrali sul punto t, o t+1)
        # Usiamo lo stato a t per calcolare le derivate spaziali
        u_mid = u[1:-1, :-1]
        u_x = (u[2:, :-1] - u[:-2, :-1]) / (2 * self.dx)
        u_xx = (u[2:, :-1] - 2 * u[1:-1, :-1] + u[:-2, :-1]) / (self.dx**2)
        
        # Equazione: u_t + u*u_x - nu*u_xx = 0
        pde_res = u_t[1:-1, :] + u_mid * u_x - self.nu * u_xx
        return pde_res.flatten()

    def __call__(self, u_flat):
        """Interfaccia hfunc per il campionatore"""
        u = _reshape_to_2d(u_flat, self.nx, self.nt).to(device=self.device)
        
        ic = self.ic_residual(u)
        bc = self.bc_residual(u)
        pde = self.pde_residual(u)
        
        # Concateniamo tutto: questo è il vettore h(u)
        return torch.cat([ic, bc, pde])


class BurgersEquationResidualsFullPDE(BurgersEquationResidualsFull):
    """Burgers residual with IC, boundary errors, and nonlinear PDE only.

    This variant keeps the same boundary convention as the generated dataset
    and scales the PDE block for stable guided sampling.
    """

    def __init__(self, data, nx=100, nt=100, nu=0.01, spatial_domain=[0, 1], time_domain=[0, 1], pde_scale=None):
        super().__init__(
            data=data,
            nx=nx,
            nt=nt,
            nu=nu,
            spatial_domain=spatial_domain,
            time_domain=time_domain,
        )
        self.pde_scale = float(self.dx ** 2 if pde_scale is None else pde_scale)

    def full_residual(self, u_flat):
        ic = self.ic_residual(u_flat)
        bc = self.bc_residual(u_flat)
        pde = self.pde_residual(u_flat) * self.pde_scale
        return torch.cat([ic, bc, pde], dim=0)

    def pde_residual_scaled(self, u_flat):
        return self.pde_residual(u_flat) * self.pde_scale

    def full_residual_unscaled(self, u_flat):
        ic = self.ic_residual(u_flat)
        bc = self.bc_residual(u_flat)
        pde = self.pde_residual(u_flat)
        return torch.cat([ic, bc, pde], dim=0)

    def __call__(self, u_flat):
        return self.full_residual(u_flat)