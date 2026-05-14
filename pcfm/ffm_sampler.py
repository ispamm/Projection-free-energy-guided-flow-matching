# Core sampler module to implement PCFM, vanilla flow matching, ECI, DiffusionPDE's guided sample, and D-Flow sampling

import gc
import math
import torch
from tqdm import tqdm
from torchdiffeq import odeint
try:
    from torchdiffeq import odeint_adjoint
except ImportError:
    odeint_adjoint = None
from .pcfm_sampling import make_grid, pcfm_batched, pcfm_2d_batched


def _compute_gamma_t(gamma_max, t_scalar, gamma_schedule):
    # Normalize schedule name to be case-insensitive
    sched = str(gamma_schedule).lower()
    if sched == 'parabolic':
        return gamma_max * (t_scalar ** 2)
    if sched == 'sine':
        return gamma_max * math.sin(math.pi * t_scalar)
    if sched == 'constant':
        return gamma_max
    if sched == 'linear':
        return gamma_max * t_scalar
    if sched == 'cosine':
        return gamma_max * (1 - math.cos(math.pi * t_scalar)) / 2
    raise ValueError(f"Unsupported gamma_schedule '{gamma_schedule}'. Supported: parabolic, sine, constant, linear, cosine.")


class FFM_sampler:
    """
    Collection of samplers using a pretrained functional flow matching model 
    """
    def __init__(self, model, gp):
        self.model = model
        self.gp = gp

    def pcfm_sample(self, u0, n_step, hfunc, mode='root', newtonsteps=1, eps=1e-6,
                    guided_interpolation=True, interpolation_params={}, use_vmap=False):
        """
        PCFM sampler
        """
        dt = 1.0 / n_step
        u = u0.clone()
        for t in tqdm(torch.linspace(0, 1, n_step + 1, device=u0.device)[:-1], desc="PCFM sampling"):
            vf = self.model(t, u)
            v_proj = pcfm_batched(
                ut=u, vf=vf, t=t, u0=u0, dt=dt,
                hfunc=hfunc, mode=mode, newtonsteps=newtonsteps,
                guided_interpolation=guided_interpolation,
                interpolation_params=interpolation_params,
                eps=eps,
                use_vmap=use_vmap
            )
            u = u + dt * v_proj
        return u.detach()
    
    def proflow_sample(self, u0, n_step, hfunc, K=3, lr_base=0.1):
        """
        PROFlow: Zero-Shot Physics-Consistent Sampling (Yu et al., 2026)
        Implementation following Appendix C.
        """
        dt = 1.0 / n_step
        u = u0.clone()
        ts = torch.linspace(0, 1, n_step + 1, device=u0.device)
        
        
        grid = make_grid(u.shape[-2:], u.device)

        for t in tqdm(ts[:-1], desc="PROFlow sampling"):
            # 1. Velocity Field evaluation
            vf = self.model(t, u)
            
            u_1_hat = u + (1 - t) * vf
            
            lr_t = lr_base * (1.0 - t.item())
            
            u_1_refined = u_1_hat.detach().requires_grad_(True)
            for _ in range(K):  
                residual = hfunc(u_1_refined)
                loss = (residual ** 2).sum()
                
                grad = torch.autograd.grad(loss, u_1_refined)[0]
                
                # Proximal update
                u_1_refined = u_1_refined.detach() - lr_t * grad
                u_1_refined = u_1_refined.requires_grad_(True)
            
            u_1_refined = u_1_refined.detach()

            t_next = t + dt
            if t_next < 1.0:
                epsilon = self.gp.sample(grid, u.shape[-2:], n_samples=u.shape[0])
                u = (1 - t_next) * epsilon + t_next * u_1_refined
            else:
                u = u_1_refined

        return u.detach()

    @torch.no_grad()
    def vanilla_sample(self, u0, n_step):
        """
        Vanilla FFM
        """
        dt = 1.0 / n_step
        u = u0.clone()
        for t in tqdm(torch.linspace(0, 1, n_step + 1, device=u0.device)[:-1], desc="Vanilla"):
            vf = self.model(t, u)
            u = u + dt * vf
        return u.detach()
    
    def continuous_guided_sample(self, u0, n_step, hfunc, gamma_max=10.0, gamma_schedule='parabolic', track_energy=False, final_refinement=False, refinement_steps=5, refinement_lr=1e-2, grad_clip=1.0):
        """
        Current state continuous Gradient Flow with configurable Scheduling,
        Gradient Clipping, and optional final physics refinement.
        """
        dt = 1.0 / n_step
        u = u0.clone()
        ts = torch.linspace(0, 1, n_step + 1, device=u0.device)
        
        energy_history = [] if track_energy else None

        for t in tqdm(ts[:-1], desc="Continuous Guided Flow", leave=False):
            gamma_t = _compute_gamma_t(gamma_max, t.item(), gamma_schedule)
            u = u.detach().requires_grad_(True)
            vf = self.model(t, u)
            residual = hfunc(u)
            energy = (residual ** 2).sum()
            
            if track_energy:
                energy_history.append(energy.item())
            
            grad_E = torch.autograd.grad(energy, u)[0]
            
            # --- GRADIENT CLIPPING ---
            grad_norm = torch.norm(grad_E)
            if grad_norm > grad_clip:
                clipped_grad_E = grad_E * (grad_clip / grad_norm)
            else:
                clipped_grad_E = grad_E
        
            u = u.detach() + dt * (vf.detach() - gamma_t * clipped_grad_E)

        if final_refinement:
            u = u.detach().requires_grad_(True)
            for _ in range(refinement_steps):
                residual = hfunc(u)
                energy = (residual ** 2).sum()
                
                grad_E = torch.autograd.grad(energy, u)[0]
                
                grad_norm = torch.norm(grad_E)
                if grad_norm > grad_clip:
                    clipped_grad_E = grad_E * (grad_clip / grad_norm)
                else:
                    clipped_grad_E = grad_E
                
                # Simple gradient descent without vector field
                u = u.detach() - refinement_lr * clipped_grad_E
                u = u.requires_grad_(True)    

        return u.detach(), energy_history

    @torch.no_grad()
    def eci_sample(self, u0, n_step, n_mix, resample_step, constraint):
        """
        ECI sampling
        """
        u = u0.clone()
        ts = torch.linspace(0, 1, n_step + 1, device=u0.device)
        cnt = 0
        dt = 1 / n_step
        grid = make_grid(u.shape[-2:], u.device)

        if resample_step == 0 or resample_step is None:
            resample_step = n_step * n_mix + 1

        for t in tqdm(ts[:-1], desc='ECI sampling'):
            for mix in range(n_mix):
                cnt += 1
                if cnt % resample_step == 0:
                    u0 = self.gp.sample(grid, u.shape[-2:], n_samples=u.shape[0])
                vf = self.model(t, u)
                u1 = u + vf * (1 - t)
                u1 = constraint.adjust(u1)
                if mix < n_mix - 1:
                    u = u1 * t + u0 * (1 - t)
                else:
                    u = u1 * (t + dt) + u0 * (1 - t - dt)
        return u.detach()


    def guided_sample(self, u0, u1_true, mask, n_step, loss_fn, eta=2e2):
        """
        DiffusionPDE: takes an IC and PINN loss (if known) on the extrapolated sample and updates the vector field 
        """
        device = u0.device
        u = u0.clone().to(device)
        u1_true = u1_true.to(device)
        mask = mask.to(device)
        ts = torch.linspace(0, 1, n_step + 1, device=device)

        for t in tqdm(ts[:-1], desc='DiffusionPDE sampling'):
            vf = self.model(t, u).detach()
            if t < ts[-2]:
                vf2 = self.model(t + 1 / n_step, u).detach()
                vf = (vf + vf2) / 2

            u.requires_grad_(True)
            u1_pred = u + vf * (1 - t)
            loss = loss_fn(u1_pred, u1_true, mask)
            loss.backward()
            grad = u.grad
            u = u.detach() + vf / n_step - eta * grad
            del loss, u1_pred, grad
            torch.cuda.empty_cache()
        return u.detach()

    
    def dflow_sample(self, u1_true, mask, n_sample, n_step, n_iter=20, lr=1e-1, loss_fn=None):
        """
        D-Flow: optimizes the noise by differentiating through the flow matching ODE steps 
        """
        device = u1_true.device
        mask = mask.to(device)
        grid = make_grid(u1_true.size()[1:], device)

        noise = self.gp.sample(grid, u1_true.size()[1:], n_samples=n_sample).to(device)
        noise.requires_grad_(True)

        ts = torch.linspace(0, 1, n_step + 1, device=device)

        def default_loss_fn(u_pred, u_true, mask):
            return ((u_pred - u_true) * mask).square().sum()
        loss_fn = loss_fn or default_loss_fn

        def euler_ffm(u):
            #print("DFlow sampling...")
            tspan = torch.tensor([0, 1.], device=device)
            u = odeint(self.model, u, tspan, method="euler", options = {"step_size":ts[1]-ts[0]})[-1]
            return u 

        def closure():
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            u_pred = euler_ffm(noise)
            loss = loss_fn(u_pred, u1_true, mask)
            loss.backward()
            return loss

        optimizer = torch.optim.LBFGS([noise], max_iter=n_iter, lr=lr)
        optimizer.step(closure)

        with torch.no_grad():
            u_final = euler_ffm(noise)
        return u_final.detach()

    def dflow_sampling(self, *args, **kwargs):
        return self.dflow_sample(*args, **kwargs)

    def dflow_ns_sampling(
        self,
        u1_true,
        mask,
        n_sample,
        n_step,
        n_iter=10,
        lr=1e-2,
        loss_fn=None,
        freeze_model=True,
        use_adjoint=True,
    ):
        """
        Memory-optimized D-Flow variant for Navier-Stokes.
        Keeps the original dflow_sample unchanged.
        """
        device = u1_true.device
        u1_true = u1_true.to(device)
        mask = mask.to(device)

        grid = make_grid(u1_true.size()[1:], device)
        noise = self.gp.sample(grid, u1_true.size()[1:], n_samples=n_sample).to(device)
        noise.requires_grad_(True)

        ts = torch.linspace(0, 1, n_step + 1, device=device)
        step_size = ts[1] - ts[0]

        def default_loss_fn(u_pred, u_true, m):
            return ((u_pred - u_true) * m).square().sum()

        loss_fn = loss_fn or default_loss_fn
        ode_solver = odeint_adjoint if (use_adjoint and odeint_adjoint is not None) else odeint

        def euler_ffm(u):
            tspan = torch.tensor([0.0, 1.0], device=device)
            return ode_solver(
                self.model,
                u,
                tspan,
                method="euler",
                options={"step_size": step_size},
            )[-1]

        prev_requires_grad = None
        if freeze_model:
            prev_requires_grad = [p.requires_grad for p in self.model.parameters()]
            for p in self.model.parameters():
                p.requires_grad_(False)

        optimizer = torch.optim.LBFGS([noise], max_iter=n_iter, lr=lr)

        try:
            def closure():
                optimizer.zero_grad(set_to_none=True)
                u_pred = euler_ffm(noise)
                loss = loss_fn(u_pred, u1_true, mask)
                loss.backward()
                return loss

            optimizer.step(closure)
        finally:
            if prev_requires_grad is not None:
                for p, req in zip(self.model.parameters(), prev_requires_grad):
                    p.requires_grad_(req)

        with torch.no_grad():
            u_final = euler_ffm(noise)

        return u_final.detach()

    def dflow_ns_sample(self, *args, **kwargs):
        return self.dflow_ns_sampling(*args, **kwargs)


class FFM_NS_sampler:
    """
    Collection of samplers using a pretrained functional flow matching model for Navier Stokes (2d)
    """
    def __init__(self, model):
        self.model = model

    # PCFM
    def pcfm_sample(self, u0, n_step, hfunc, mode='root', newtonsteps=1, eps=1e-6,
                    guided_interpolation=True, interpolation_params={}):
        dt = 1.0 / n_step
        u = u0.clone()
        for t in tqdm(torch.linspace(0, 1, n_step + 1, device=u0.device)[:-1], desc="PCFM sampling"):
            vf = self.model(t, u)
            v_proj = pcfm_2d_batched(
                ut=u, vf=vf, t=t, u0=u0, dt=dt,
                hfunc=hfunc, mode=mode, newtonsteps=newtonsteps,
                guided_interpolation=guided_interpolation,
                interpolation_params=interpolation_params,
                eps=eps
            )
            u = u + dt * v_proj
        return u.detach()

    def proflow_sample(self, u0, n_step, hfunc, K=3, lr_base=0.1):
        """
        PROFlow variant for Navier-Stokes sampler.
        Uses fresh Gaussian noise for re-interpolation in 2D settings.
        """
        dt = 1.0 / n_step
        u = u0.clone()
        ts = torch.linspace(0, 1, n_step + 1, device=u0.device)

        for t in tqdm(ts[:-1], desc="PROFlow sampling"):
            vf = self.model(t, u)
            u_1_hat = u + (1 - t) * vf

            lr_t = lr_base * (1.0 - t.item())

            u_1_refined = u_1_hat.detach().requires_grad_(True)
            for _ in range(K):
                residual = hfunc(u_1_refined)
                loss = (residual ** 2).sum()
                grad = torch.autograd.grad(loss, u_1_refined)[0]
                u_1_refined = u_1_refined.detach() - lr_t * grad
                u_1_refined = u_1_refined.requires_grad_(True)

            u_1_refined = u_1_refined.detach()

            t_next = t + dt
            if t_next < 1.0:
                epsilon = torch.randn_like(u)
                u = (1 - t_next) * epsilon + t_next * u_1_refined
            else:
                u = u_1_refined

        return u.detach()

    @torch.no_grad()
    def vanilla_sample(self, u0, n_step):
        dt = 1.0 / n_step
        u = u0.clone()
        for t in tqdm(torch.linspace(0, 1, n_step + 1, device=u0.device)[:-1], desc="Vanilla"):
            vf = self.model(t, u)
            u = u + dt * vf
        return u.detach()

    def continuous_guided_sample(self, u0, n_step, hfunc, gamma_max=10.0, gamma_schedule='parabolic', track_energy=False, final_refinement=False, refinement_steps=5, refinement_lr=1e-2):
        """
        Current state continuous Gradient Flow with configurable Scheduling and optional final physics refinement.
        """
        dt = 1.0 / n_step
        u = u0.clone()
        ts = torch.linspace(0, 1, n_step + 1, device=u0.device)

        energy_history = [] if track_energy else None

        for t in tqdm(ts[:-1], desc="Continuous Guided Flow", leave=False):
            gamma_t = _compute_gamma_t(gamma_max, t.item(), gamma_schedule)
            u = u.detach().requires_grad_(True)
            vf = self.model(t, u)
            residual = hfunc(u)
            energy = (residual ** 2).sum()

            if track_energy:
                energy_history.append(energy.item())

            grad_E = torch.autograd.grad(energy, u)[0]
            u = u.detach() + dt * (vf.detach() - gamma_t * grad_E)

        if final_refinement:
            u = u.detach().requires_grad_(True)
            for _ in range(refinement_steps):
                residual = hfunc(u)
                energy = (residual ** 2).sum()
                grad_E = torch.autograd.grad(energy, u)[0]
                u = u.detach() - refinement_lr * grad_E
                u = u.requires_grad_(True)

        return u.detach(), energy_history

    @torch.no_grad()
    def eci_sample(self, u0, n_step, n_mix, resample_step, constraint):
        u = u0.clone()
        ts = torch.linspace(0, 1, n_step + 1, device=u0.device)
        cnt = 0
        dt = 1 / n_step
        grid = make_grid(u.shape[-2:], u.device)

        if resample_step == 0 or resample_step is None:
            resample_step = n_step * n_mix + 1

        for t in tqdm(ts[:-1], desc='ECI sampling'):
            for mix in range(n_mix):
                cnt += 1
                if cnt % resample_step == 0:
                    u0 = torch.randn_like(u)
                vf = self.model(t, u)
                u1 = u + vf * (1 - t)
                u1 = constraint.adjust(u1)
                if mix < n_mix - 1:
                    u = u1 * t + u0 * (1 - t)
                else:
                    u = u1 * (t + dt) + u0 * (1 - t - dt)
        return u.detach()

    def guided_sample(self, u0, u1_true, mask, n_step, loss_fn, eta=2e2):
        device = u0.device
        u = u0.clone().to(device)
        u1_true = u1_true.to(device)
        mask = mask.to(device)
        ts = torch.linspace(0, 1, n_step + 1, device=device)

        for t in tqdm(ts[:-1], desc='DiffusionPDE sampling'):
            vf = self.model(t, u).detach()
            if t < ts[-2]:
                vf2 = self.model(t + 1 / n_step, u).detach()
                vf = (vf + vf2) / 2

            u.requires_grad_(True)
            u1_pred = u + vf * (1 - t)
            loss = loss_fn(u1_pred, u1_true, mask)
            loss.backward()
            grad = u.grad
            u = u.detach() + vf / n_step - eta * grad
        return u.detach()

    def dflow_sample(self, u1_true, mask, n_sample, n_step, n_iter=20, lr=1e-1, loss_fn=None):
        device = u1_true.device
        mask = mask.to(device)
        grid = make_grid(u1_true.size()[1:], device)

        noise = torch.randn_like(u1_true) 
        noise.requires_grad_(True)
        ts = torch.linspace(0, 1, n_step + 1, device=device)
        
        def default_loss_fn(u_pred, u_true, mask):
            return ((u_pred - u_true) * mask).square().sum()
        loss_fn = loss_fn or default_loss_fn

        def euler_ffm(u):
            print("DFlow sampling...")
            tspan = torch.tensor([0, 1.], device=device)
            u = odeint(self.model, u, tspan, method="euler", options = {"step_size":ts[1]-ts[0]})[-1]
            return u 

        def closure():
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            u_pred = euler_ffm(noise)
            loss = loss_fn(u_pred, u1_true, mask)
            loss.backward()
            return loss

        optimizer = torch.optim.LBFGS([noise], max_iter=n_iter, lr=lr)
        optimizer.step(closure)

        with torch.no_grad():
            u_final = euler_ffm(noise)
        return u_final.detach()

    def dflow_sampling(self, *args, **kwargs):
        return self.dflow_sample(*args, **kwargs)

    def dflow_ns_sample(
        self,
        u1_true,
        mask,
        n_sample,
        n_step,
        n_iter=10,
        lr=1e-2,
        loss_fn=None,
        freeze_model=True,
        use_adjoint=True,
    ):
        """
        Memory-optimized D-Flow variant for Navier-Stokes.
        Keeps the original dflow_sample unchanged.
        """
        device = u1_true.device
        u1_true = u1_true.to(device)
        mask = mask.to(device)

        noise = torch.randn_like(u1_true)
        noise.requires_grad_(True)

        ts = torch.linspace(0, 1, n_step + 1, device=device)
        step_size = ts[1] - ts[0]

        def default_loss_fn(u_pred, u_true, m):
            return ((u_pred - u_true) * m).square().sum()

        loss_fn = loss_fn or default_loss_fn
        ode_solver = odeint_adjoint if (use_adjoint and odeint_adjoint is not None) else odeint

        def euler_ffm(u):
            tspan = torch.tensor([0.0, 1.0], device=device)
            return ode_solver(
                self.model,
                u,
                tspan,
                method="euler",
                options={"step_size": step_size},
            )[-1]

        prev_requires_grad = None
        if freeze_model:
            prev_requires_grad = [p.requires_grad for p in self.model.parameters()]
            for p in self.model.parameters():
                p.requires_grad_(False)

        optimizer = torch.optim.LBFGS([noise], max_iter=n_iter, lr=lr)

        try:
            def closure():
                optimizer.zero_grad(set_to_none=True)
                u_pred = euler_ffm(noise)
                loss = loss_fn(u_pred, u1_true, mask)
                loss.backward()
                return loss

            optimizer.step(closure)
        finally:
            if prev_requires_grad is not None:
                for p, req in zip(self.model.parameters(), prev_requires_grad):
                    p.requires_grad_(req)

        with torch.no_grad():
            u_final = euler_ffm(noise)

        return u_final.detach()

    def dflow_ns_sampling(self, *args, **kwargs):
        return self.dflow_ns_sample(*args, **kwargs)

