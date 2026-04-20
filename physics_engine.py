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

    def __init__(
        self,
        data,
        nx=100,
        nt=100,
        nu=0.01,
        spatial_domain=(0.0, 2.0 * math.pi),
        time_domain=(0.0, 1.0),
        pde_scale=None,
        include_mass=False,
    ):
        super().__init__(
            data=data,
            nx=nx,
            nt=nt,
            nu=nu,
            spatial_domain=spatial_domain,
            time_domain=time_domain,
        )
        self.pde_scale = float(self.dx ** 2 if pde_scale is None else pde_scale)
        self.include_mass = bool(include_mass)

    def pde_residual_scaled(self, u_flat):
        return self.pde_residual(u_flat) * self.pde_scale

    def full_residual(self, u_flat):
        blocks = [
            self.ic_residual(u_flat),
            self.pde_residual_scaled(u_flat),
        ]
        if self.include_mass:
            blocks.append(self.mass_residual(u_flat))
        return torch.cat(blocks, dim=0)

    def full_residual_unscaled(self, u_flat):
        blocks = [
            self.ic_residual(u_flat),
            self.pde_residual(u_flat),
        ]
        if self.include_mass:
            blocks.append(self.mass_residual(u_flat))
        return torch.cat(blocks, dim=0)

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
        include_mass=False,
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
        self.include_mass = bool(include_mass)

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
        blocks = [
            self.ic_residual(u_flat),
            self.pde_residual_scaled(u_flat),
        ]
        if self.include_mass:
            blocks.append(self.mass_residual(u_flat)[1:])
        return torch.cat(blocks, dim=0)

    def full_residual_unscaled(self, u_flat):
        blocks = [
            self.ic_residual(u_flat),
            self.pde_residual(u_flat),
        ]
        if self.include_mass:
            blocks.append(self.mass_residual(u_flat)[1:])
        return torch.cat(blocks, dim=0)


class NavierStokesResidualsFullPDE:
    """2D vorticity-form Navier-Stokes residuals for periodic datasets.

    The formulation matches the generated dataset conventions:
    - periodic domain on x and y in [0, 1)
    - forcing field f(x, y) constant in time
    - viscosity nu (mu in dataset filename)
    """

    def __init__(
        self,
        data,
        forcing,
        nx=64,
        ny=64,
        nt=50,
        nu=0.001,
        x_grid=None,
        y_grid=None,
        t_grid=None,
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 49.0),
        pde_scale=None,
        include_mass=False,
    ):
        self.device = data.device
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.nu = float(nu)

        data_tensor = data.to(device=self.device, dtype=torch.float32)
        if data_tensor.dim() == 4 and data_tensor.shape[0] == 1:
            data_tensor = data_tensor.squeeze(0)
        if data_tensor.dim() != 3:
            raise ValueError(f"Expected NS data with 3 dims [nx, ny, nt], got {tuple(data.shape)}")
        if data_tensor.shape != (nx, ny, nt):
            raise ValueError(
                f"NS data shape mismatch: expected {(nx, ny, nt)}, got {tuple(data_tensor.shape)}"
            )
        self.data = data_tensor

        forcing_tensor = forcing.to(device=self.device, dtype=torch.float32)
        if forcing_tensor.dim() == 3 and forcing_tensor.shape[0] == 1:
            forcing_tensor = forcing_tensor.squeeze(0)
        if forcing_tensor.dim() != 2:
            raise ValueError(f"Expected forcing with 2 dims [nx, ny], got {tuple(forcing.shape)}")
        if forcing_tensor.shape != (nx, ny):
            raise ValueError(
                f"Forcing shape mismatch: expected {(nx, ny)}, got {tuple(forcing_tensor.shape)}"
            )
        self.forcing = forcing_tensor

        if x_grid is None:
            self.x_grid = torch.linspace(spatial_domain[0], spatial_domain[1], nx + 1, device=self.device)[:-1]
        else:
            self.x_grid = x_grid.to(device=self.device, dtype=torch.float32)

        if y_grid is None:
            self.y_grid = torch.linspace(spatial_domain[0], spatial_domain[1], ny + 1, device=self.device)[:-1]
        else:
            self.y_grid = y_grid.to(device=self.device, dtype=torch.float32)

        if t_grid is None:
            self.t_grid = torch.linspace(time_domain[0], time_domain[1], nt, device=self.device)
        else:
            self.t_grid = t_grid.to(device=self.device, dtype=torch.float32)

        self.dx = (self.x_grid[1] - self.x_grid[0]).item()
        self.dy = (self.y_grid[1] - self.y_grid[0]).item()
        self.dt = (self.t_grid[1] - self.t_grid[0]).item()
        self.pde_scale = float((min(self.dx, self.dy) ** 2) if pde_scale is None else pde_scale)
        self.include_mass = bool(include_mass)

        kx = torch.fft.fftfreq(self.nx, d=self.dx, device=self.device).view(self.nx, 1)
        ky = torch.fft.fftfreq(self.ny, d=self.dy, device=self.device).view(1, self.ny)
        self.kx = kx
        self.ky = ky
        self.lap = (2.0 * math.pi) ** 2 * (kx ** 2 + ky ** 2)
        self.lap[0, 0] = 1.0

    def _reshape(self, u_flat):
        u = u_flat.to(dtype=torch.float32)
        if u.dim() == 1:
            u = u.view(self.nx, self.ny, self.nt)
        elif u.dim() == 4 and u.shape[0] == 1:
            u = u.squeeze(0)
        elif u.dim() == 3:
            pass
        else:
            raise ValueError(f"Unsupported NS tensor shape: {tuple(u.shape)}")

        if u.shape != (self.nx, self.ny, self.nt):
            raise ValueError(
                f"NS tensor shape mismatch: expected {(self.nx, self.ny, self.nt)}, got {tuple(u.shape)}"
            )
        return u

    def ic_residual(self, u_flat):
        u = self._reshape(u_flat)
        return (u[:, :, 0] - self.data[:, :, 0]).flatten()

    def mass_residual(self, u_flat):
        u = self._reshape(u_flat)
        mass_t = u.sum(dim=(0, 1)) * self.dx * self.dy
        return (mass_t[1:] - mass_t[0]).flatten()

    def bc_left_residual(self, u_flat):
        u = self._reshape(u_flat)
        # On endpoint-excluded periodic grids, u[0] and u[-1] are not the same point.
        # We compare the seam jump against the reference data seam jump.
        pred_jump = u[0, :, :] - u[-1, :, :]
        ref_jump = self.data[0, :, :] - self.data[-1, :, :]
        return (pred_jump - ref_jump).flatten()

    def bc_right_residual(self, u_flat):
        u = self._reshape(u_flat)
        pred_jump = u[:, 0, :] - u[:, -1, :]
        ref_jump = self.data[:, 0, :] - self.data[:, -1, :]
        return (pred_jump - ref_jump).flatten()

    def bc_residual(self, u_flat):
        return torch.cat([self.bc_left_residual(u_flat), self.bc_right_residual(u_flat)], dim=0)

    def pde_residual(self, u_flat):
        w = self._reshape(u_flat)

        w_t = (w[:, :, 1:] - w[:, :, :-1]) / self.dt
        w_old = w[:, :, :-1]

        w_h = torch.fft.fftn(w_old, dim=(0, 1), norm='backward')
        lap = self.lap.unsqueeze(-1)
        kx = self.kx.unsqueeze(-1)
        ky = self.ky.unsqueeze(-1)

        psi_h = w_h / lap

        q_h = 1j * 2.0 * math.pi * ky * psi_h
        v_h = -1j * 2.0 * math.pi * kx * psi_h
        q = torch.fft.ifftn(q_h, dim=(0, 1), norm='backward').real
        v = torch.fft.ifftn(v_h, dim=(0, 1), norm='backward').real

        w_x_h = 1j * 2.0 * math.pi * kx * w_h
        w_y_h = 1j * 2.0 * math.pi * ky * w_h
        w_x = torch.fft.ifftn(w_x_h, dim=(0, 1), norm='backward').real
        w_y = torch.fft.ifftn(w_y_h, dim=(0, 1), norm='backward').real

        advection = q * w_x + v * w_y

        lap_w_h = -lap * w_h
        lap_w = torch.fft.ifftn(lap_w_h, dim=(0, 1), norm='backward').real
        diffusion = self.nu * lap_w

        forcing = self.forcing.unsqueeze(-1)
        res = w_t + advection - diffusion - forcing
        return res.flatten()

    def pde_residual_scaled(self, u_flat):
        return self.pde_residual(u_flat) * self.pde_scale

    def full_residual(self, u_flat):
        blocks = [
            self.ic_residual(u_flat),
            self.bc_residual(u_flat),
            self.pde_residual_scaled(u_flat),
        ]
        if self.include_mass:
            blocks.append(self.mass_residual(u_flat)[1:])
        return torch.cat(blocks, dim=0)

    def full_residual_unscaled(self, u_flat):
        blocks = [
            self.ic_residual(u_flat),
            self.bc_residual(u_flat),
            self.pde_residual(u_flat),
        ]
        if self.include_mass:
            blocks.append(self.mass_residual(u_flat)[1:])
        return torch.cat(blocks, dim=0)

    def __call__(self, u_flat):
        return self.full_residual(u_flat)


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