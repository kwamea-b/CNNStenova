import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import numpy as np


class CNNKernelLearner(nn.Module):
    """
    General CNN‐based kernel learner for finite‐difference/Adams–Bashforth update.

    Args:
      bBC, tBC        Dirichlet boundary values at the left (x=0) and right (x=L).
      kernel_size     Number of stencil points N (must be odd).

    """

    def __init__(self,
                 tBC=None, 
                 bBC=None, 
                 kernel_size=3,
                 in_channels: int = 1,
                 pad_mode: str = 'mirror',
                 initial_weights = "random",
                 device=None):
        super(CNNKernelLearner, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        # Define a single 1D convolutional layer with 1 input and 1 output channel
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=kernel_size,
                              padding=0, bias=False)
        if tBC != None:
            self.tBC = tBC
        else:
            self.tBC = None
        if bBC != None:
            self.bBC = bBC
        else:
            self.bBC = None

        self.N = kernel_size
        self.pad = kernel_size // 2
        self.in_ch = in_channels
        self.pad_mode = pad_mode
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # Learnable parameters (kernel weights)
        self.raw_weight = nn.Parameter(torch.randn(kernel_size))

        #Set initial weights for network
        if initial_weights == "random":
            with torch.no_grad():
                nn.init.xavier_uniform_(self.conv.weight, gain=1)  # Xavier initialization
        elif type(initial_weights) is list and len(initial_weights) == kernel_size:
            with torch.no_grad():
                self.conv.weight[:] = torch.tensor(initial_weights, dtype=torch.float32).view(1, 1, -1)
            print(initial_weights, self.conv.weight[:])

        else:
            pass

    def mirror_pad(self, u: torch.Tensor, tBC: float, bBC: float) -> torch.Tensor:
        """
        Manual mirror padding for Dirichlet BC:
        u: (batch, 1, L) → returns (batch, 1, L + 2*pad)
        """
        N, C, L = u.shape
        p = self.pad

        # allocate extended tensor
        u_ext = torch.empty((N, 1, L + 2*p), device=self.device, dtype=u.dtype)
        # center
        u_ext[:, :, p:p+L] = u

        # left ghosts
        # mirror about the boundary value bBC:
        # u_ext[..., p- i] = 2*bBC - u_ext[..., p + (i-1)]
        for i in range(1, p+1):
            u_ext[:, :, p - i] = 2*bBC - u_ext[:, :, p + (i-1)]
        # right ghosts
        for i in range(1, p+1):
            u_ext[:, :, p + L -1 + i] = 2*tBC - u_ext[:, :, p + L - i]

        return u_ext

    def forward(self, x):

        # Set boundary conditions
        tBC = self.tBC if self.tBC is not None else x[:, :, -1]
        bBC = self.bBC if self.bBC is not None else x[:, :, 0]

        if self.pad_mode == 'mirror' and self.pad > 0:
            x_pad = self.mirror_pad(x[:, :, :], tBC, bBC)
        elif self.pad_mode == 'zero' and self.pad > 0:
            x_pad = F.pad(x, (self.pad, self.pad), mode='constant', value=0)
        else:
            x_pad = x

        x = self.conv(x_pad)
        x[:, :, 0] = bBC
        x[:, :, -1] = tBC
        return x

    def constraint_loss(self, C):
        """
            Computes soft constraint penalties on stencil weights.
            Inputs: C - second moment constraint aims to match 2C 
            
        """
        # conv.weight shape: (1,1,N) → flatten to (N,)
        w = self.conv.weight.view(-1)

        j_values = torch.arange(-(self.N // 2),
                                (self.N // 2) + 1,
                                dtype=w.dtype,
                                device=w.device)

        # constraints
        sum_bj = w.sum()
        sum_bj_j = (w * j_values).sum()
        sum_bj_j2 = (w * j_values**2).sum()

        loss_1 = (sum_bj - 1) ** 2          # zeroth moment (sum of weights)
        loss_2 = sum_bj_j ** 2              # first moment = 0
        loss_3 = (sum_bj_j2 - 2*C)**2       # second moment

        return loss_1, loss_2, loss_3

    def call_model(self, u, time_steps, tBC=None, bBC=None, include_initial=True):

        # Move model and inputs to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      
        self = self.to(device)

        if type(u) == np.ndarray:
            u = torch.tensor(u, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        else:
            u = u.to(device)
        
        # Apply boundary conditions if provided
        if tBC is not None:
            self.tBC = tBC.to(device) if isinstance(tBC, torch.Tensor) else tBC
        if bBC is not None:
            self.bBC = bBC.to(device) if isinstance(bBC, torch.Tensor) else bBC

        if include_initial:
            uhist = [u[0, 0, :].detach().clone()]
        else:
            uhist = []
        for _ in range(time_steps):

            # Forward pass (no gradient if just inference)
            with torch.no_grad():
                u = self(u)

            # Save current state (still on GPU)
            uhist.append(u[0, 0, :].detach().clone())

        # Stack and move to CPU once at the end
        uhist = torch.stack(uhist).cpu().numpy()
        return uhist

    def set_CFL(self, dx, dt, mu):
        self.dt = dt
        self.dx = dx
        self.mu = mu
        self.CFL = mu*dt/dx**2

    def to(self, device):
        super().to(device)
        self.device = device
        return self

class BurgersCNNKernelLearner(CNNKernelLearner):
    """
    CNN kernel learner for the (optionally forced) Burgers equation.

    Inherits all infrastructure from CNNKernelLearner (padding, BC handling,
    call_model, set_CFL, etc.) and extends it with:

      * Multi-channel input  (u, u²/2, optional forcing f)
      * Residual connection  u_{n+1} = u_n + Conv([u, u²/2, f])
      * Periodic boundary condition support
      * Physics-informed weight initialisation (diffusion + convection stencils)
      * predict_forward() — time-stepping with optional forcing history

    Args
    ----
    bBC, tBC        : Dirichlet boundary values (ignored when periodic=True).
    kernel_size     : Stencil width (must be odd).
    in_channels     : 1 = u only, 2 = [u, u²/2], 3 = [u, u²/2, f].
    residual        : If True, output = u + Conv(input).
    pad_mode        : 'mirror' (Dirichlet) or 'circular' (periodic).
    periodic        : Shorthand to switch to circular padding / no BC pinning.
    physical_init   : Seed weights with diffusion+convection stencil values.
    C_diff_init     : μ dt/dx²  used for the diffusion stencil seed.
    C_conv_init     : dt/(2 dx) used for the convection stencil seed.
    noise_factor    : Noise amplitude added on top of physical init.
    bias            : Whether the Conv1d layer has a bias term.
    device          : Torch device; defaults to CUDA if available.
    """

    def __init__(
        self,
        bBC=None,
        tBC=None,
        kernel_size: int = 3,
        in_channels: int = 2,
        residual: bool = True,
        pad_mode: str = 'mirror',
        periodic: bool = False,
        physical_init: bool = True,
        C_diff_init=None,
        C_conv_init=None,
        noise_factor: float = 0.01,
        bias: bool = False,
        device=None,
    ):
        # ------------------------------------------------------------------ #
        # Delegate construction to the parent.  We pass initial_weights=None  #
        # so the parent does NOT re-initialise the conv weights after we set   #
        # them below in reset_weights().                                       #
        # ------------------------------------------------------------------ #
        super().__init__(
            bBC=bBC,
            tBC=tBC,
            kernel_size=kernel_size,
            in_channels=in_channels,
            pad_mode='mirror' if not periodic else 'zero',  # overridden below
            initial_weights=None,   # we handle init ourselves
            device=device,
        )

        # Burgers-specific attributes
        self.residual = residual
        self.periodic = periodic
        self.physical_init = physical_init
        self.C_diff_init = C_diff_init
        self.C_conv_init = C_conv_init
        self.noise_factor = noise_factor

        # Replace the conv layer: parent created one for in_channels already,
        # but we want explicit bias control.
        self.conv = nn.Conv1d(
            in_channels, 1, kernel_size, padding=0, bias=bias
        )

        # Apply physical or random initialisation
        self.reset_weights()

    # ---------------------------------------------------------------------- #
    # Weight initialisation                                                    #
    # ---------------------------------------------------------------------- #

    def reset_weights(
        self,
        physical_init=None,
        C_diff=None,
        C_conv=None,
        noise_factor=None,
    ):
        """
        (Re-)initialise conv weights.

        For a 3-point, 2-channel kernel with physical_init=True the stencils are:
          channel 0 (u)    → diffusion:  [C_diff,  -2*C_diff,  C_diff]
          channel 1 (u²/2) → convection: [C_conv,  -C_conv,    0     ]
          channel 2 (f)    → forcing:    [0,        1,          0     ]  (if present)
        """
        if physical_init is None:
            physical_init = self.physical_init
        C_diff = C_diff if C_diff is not None else self.C_diff_init
        C_conv = C_conv if C_conv is not None else self.C_conv_init
        noise_factor = noise_factor if noise_factor is not None else self.noise_factor

        with torch.no_grad():
            w = self.conv.weight.data
            w.zero_()

            if physical_init and self.in_ch >= 2 and self.N == 3:
                C_d = float(C_diff) if C_diff is not None else 0.0
                C_c = float(C_conv) if C_conv is not None else 0.0

                # Diffusion stencil on the u channel
                w[0, 0, :] = torch.tensor([C_d, -2.0 * C_d, C_d])
                # Upwind convection stencil on the u²/2 channel
                if self.in_ch >= 2:
                    w[0, 1, :] = torch.tensor([C_c, -C_c, 0.0])
                # Forcing channel: pass through the centre value × 1
                if self.in_ch >= 3:
                    w[0, 2, :] = torch.tensor([0.0, 1.0, 0.0])
            else:
                w.copy_(torch.randn_like(w) * noise_factor)

            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def constraint_loss(self, C):
        """
        Override parent to handle multi-channel weights.
        Constraints applied to the u-channel (channel 0) only.
        """
        # Only take the u-channel weights: shape (kernel_size,)
        w = self.conv.weight[0, 0, :]

        j_values = torch.arange(
            -(self.N // 2), (self.N // 2) + 1,
            dtype=w.dtype, device=w.device
        )

        sum_bj = w.sum()
        sum_bj_j = (w * j_values).sum()
        sum_bj_j2 = (w * j_values ** 2).sum()

        loss_1 = (sum_bj - 1) ** 2
        loss_2 = sum_bj_j ** 2
        loss_3 = (sum_bj_j2 - 2 * C) ** 2

        return loss_1, loss_2, loss_3

    # ---------------------------------------------------------------------- #
    # Padding                                                                  #
    # ---------------------------------------------------------------------- #

    def pad_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Route to circular or multi-channel mirror padding depending on
        self.periodic.  The parent's mirror_pad() handles only a single
        channel, so we override it here for the multi-channel case.
        """
        if self.pad == 0:
            return x

        if self.periodic:
            return F.pad(x, (self.pad, self.pad), mode='circular')

        # Dirichlet mirror-pad: channel 0 uses bBC/tBC; other channels
        # reflect their own boundary values.
        return self._mirror_pad_multi_channel(x)

    def _mirror_pad_multi_channel(self, x: torch.Tensor) -> torch.Tensor:
        """Mirror-pad each input channel with appropriate boundary values."""
        N, C, L = x.shape
        device, dtype = x.device, x.dtype
        padded_channels = []

        for ch in range(C):
            x_ch = x[:, ch:ch + 1, :]

            if ch == 0:
                # Use Dirichlet BCs for the solution channel
                bBC_val = (
                    torch.full((N, 1, 1), float(self.bBC), device=device, dtype=dtype)
                    if self.bBC is not None
                    else x_ch[:, :, 0:1].detach()
                )
                tBC_val = (
                    torch.full((N, 1, 1), float(self.tBC), device=device, dtype=dtype)
                    if self.tBC is not None
                    else x_ch[:, :, -1:].detach()
                )
            else:
                # Auxiliary channels reflect their own edge values
                bBC_val = x_ch[:, :, 0:1].detach()
                tBC_val = x_ch[:, :, -1:].detach()

            x_ext = torch.empty((N, 1, L + 2 * self.pad), device=device, dtype=dtype)
            x_ext[:, :, self.pad:self.pad + L] = x_ch

            for i in range(1, self.pad + 1):
                x_ext[:, :, self.pad - i] = (
                    2.0 * bBC_val.squeeze(-1) - x_ext[:, :, self.pad + (i - 1)]
                )
                x_ext[:, :, self.pad + L - 1 + i] = (
                    2.0 * tBC_val.squeeze(-1) - x_ext[:, :, self.pad + L - i]
                )

            padded_channels.append(x_ext)

        return torch.cat(padded_channels, dim=1)

    # ---------------------------------------------------------------------- #
    # Boundary condition enforcement                                           #
    # ---------------------------------------------------------------------- #

    def _apply_dirichlet(self, u: torch.Tensor) -> torch.Tensor:
        """Pin the endpoints of u to the Dirichlet values (no-op if periodic)."""
        if self.periodic:
            return u
        if self.bBC is not None:
            u[:, :, 0] = float(self.bBC)
        if self.tBC is not None:
            u[:, :, -1] = float(self.tBC)
        return u

    # ---------------------------------------------------------------------- #
    # Forward pass                                                             #
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Accepts:
          x shaped (N, 1, L)           — raw u field; u²/2 is computed internally
          x shaped (N, in_ch, L)       — pre-assembled multi-channel input

        Returns u_{n+1} shaped (N, 1, L).
        """
        N, C_in, L = x.shape

        # Auto-assemble multi-channel input when only u is supplied
        if C_in == 1 and self.in_ch >= 2:
            u = x
            u_sq = 0.5 * u * u
            if self.in_ch == 2:
                x = torch.cat([u, u_sq], dim=1)
            elif self.in_ch == 3:
                zeros = torch.zeros_like(u)
                x = torch.cat([u, u_sq, zeros], dim=1)
        elif C_in != self.in_ch:
            raise ValueError(
                f"Input has {C_in} channel(s) but model expects {self.in_ch}."
            )

        x_pad = self.pad_input(x)
        conv_out = self.conv(x_pad)

        u_next = x[:, 0:1, :] + conv_out if self.residual else conv_out
        u_next = self._apply_dirichlet(u_next)

        return u_next

    # ---------------------------------------------------------------------- #
    # Time-stepping with optional forcing                                      #
    # ---------------------------------------------------------------------- #

    def predict_forward(
        self,
        u0,
        n_steps: int,
        forcing_history=None,
        tBC=None,
        bBC=None,
    ) -> np.ndarray:
        """
        Integrate the learned dynamics forward for n_steps time steps.

        Parameters
        ----------
        u0              : 1-D array or tensor of shape (L,) — initial condition.
        n_steps         : Number of time steps to take.
        forcing_history : None  → zero forcing
                          callable(n) → returns array of shape (L,) at step n
                          array (T, L) or (T,) → indexed by step n (clamped at end)
        tBC, bBC        : Override stored boundary values for this call only.

        Returns
        -------
        u_history : np.ndarray of shape (n_steps + 1, L)
        """
        # Temporarily override BCs if requested
        saved_tBC, saved_bBC = self.tBC, self.bBC
        if tBC is not None:
            self.tBC = tBC
        if bBC is not None:
            self.bBC = bBC

        device = self.device

        if isinstance(u0, np.ndarray):
            u = torch.from_numpy(u0).float().unsqueeze(0).unsqueeze(0).to(device)
        else:
            u = u0.float().unsqueeze(0).unsqueeze(0).to(device)

        # Build a uniform get_forcing(n) interface
        if forcing_history is None:
            def get_forcing(n):
                return np.zeros(u.shape[-1], dtype=np.float32)
        elif callable(forcing_history):
            get_forcing = forcing_history
        else:
            forcing_arr = np.asarray(forcing_history, dtype=np.float32)

            def get_forcing(n):
                idx = min(n, forcing_arr.shape[0] - 1)
                return forcing_arr[idx]

        self.eval()
        u_history = [u[0, 0, :].cpu().numpy().copy()]

        with torch.no_grad():
            for n in range(n_steps):
                if self.in_ch == 1:
                    x_in = u
                elif self.in_ch == 2:
                    u_sq = 0.5 * u * u
                    x_in = torch.cat([u, u_sq], dim=1)
                elif self.in_ch == 3:
                    u_sq = 0.5 * u * u
                    f_n = (
                        torch.from_numpy(get_forcing(n))
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )
                    x_in = torch.cat([u, u_sq, f_n], dim=1)
                else:
                    raise ValueError(f"Unsupported in_ch={self.in_ch}")

                u = self(x_in)
                u = torch.clamp(u, min=-100.0, max=100.0)
                u_history.append(u[0, 0, :].cpu().numpy().copy())

        # Restore BCs
        self.tBC, self.bBC = saved_tBC, saved_bBC

        return np.array(u_history)

class TransportScalarKernelLearner(nn.Module):
    """
    Scalar-coefficient CNN kernel learner for surface transport:

        Gamma_t + d(Gamma*u_s)/dx = D_s * Gamma_xx

    Geometric term = 0, source term = 0.

    Input to forward:
        x shape = (B, 2, Nx)
        x[:, 0, :] = Gamma^n
        x[:, 1, :] = u_s^n

    Internal physical coupling:
        q = Gamma * u_s

    Explicit update:
        Gamma^{n+1} = Gamma^n
                     + alpha * [1, -2, 1] Gamma^n
                     + beta  * [1,  0, -1] q^n

    where
        alpha = D_s * dt / dx^2
        beta  is learned and initialized as dt/(2*dx).

    Time-dependent Dirichlet boundary conditions:
        left_bc_pad/right_bc_pad are used for padding Gamma^n.
        left_bc_out/right_bc_out are used to pin Gamma^{n+1}.
    """

    def __init__(
            self,
            dx,
            dt,
            Ds_init=0.01,
            adv_init=None,
            residual=True,
            periodic=False,
            device=None,
    ):
        super().__init__()

        self.dx = float(dx)
        self.dt = float(dt)
        self.residual = residual
        self.periodic = periodic
        self.N = 3
        self.pad = 1
        self.in_ch = 2

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None else device
        )

        alpha0 = float(Ds_init) * self.dt / (self.dx ** 2)
        beta0 = self.dt / (2.0 * self.dx) if adv_init is None else float(adv_init)

        self.alpha = nn.Parameter(torch.tensor(alpha0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta0, dtype=torch.float32))

        self.register_buffer(
            "laplace_stencil",
            torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32).view(1, 1, 3),
        )
        self.register_buffer(
            "adv_stencil",
            torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32).view(1, 1, 3),
        )

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def mirror_pad(self, u, left_bc=None, right_bc=None):
        """Dirichlet mirror padding for the concentration channel."""
        if self.periodic:
            return F.pad(u, (1, 1), mode="circular")

        B, C, L = u.shape

        if left_bc is None:
            left_bc = u[:, :, 0]
        if right_bc is None:
            right_bc = u[:, :, -1]

        left_bc = left_bc.view(B, 1)
        right_bc = right_bc.view(B, 1)

        u_ext = torch.empty((B, C, L + 2), device=u.device, dtype=u.dtype)
        u_ext[:, :, 1:1 + L] = u
        u_ext[:, :, 0] = 2.0 * left_bc - u[:, :, 0]
        u_ext[:, :, -1] = 2.0 * right_bc - u[:, :, -1]
        return u_ext

    def forward(
            self,
            x,
            left_bc_pad=None,
            right_bc_pad=None,
            left_bc_out=None,
            right_bc_out=None,
    ):
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"Expected x shape (B, 2, Nx), got {tuple(x.shape)}")

        gamma = x[:, 0:1, :]
        u = x[:, 1:2, :]
        q = gamma * u

        gamma_pad = self.mirror_pad(
            gamma,
            left_bc=left_bc_pad,
            right_bc=right_bc_pad,
        )

        # q is a flux, not concentration, so do not impose concentration BCs on q.
        q_pad = F.pad(q, (1, 1), mode="replicate") if not self.periodic else F.pad(q, (1, 1), mode="circular")

        diff_term = F.conv1d(gamma_pad, self.alpha * self.laplace_stencil)
        adv_term = F.conv1d(q_pad, self.beta * self.adv_stencil)

        gamma_next = gamma + diff_term + adv_term if self.residual else diff_term + adv_term

        if not self.periodic:
            if left_bc_out is not None:
                gamma_next[:, :, 0] = left_bc_out.view(-1, 1)
            if right_bc_out is not None:
                gamma_next[:, :, -1] = right_bc_out.view(-1, 1)

        return gamma_next

    def get_kernels(self):
        diff_kernel = (self.alpha * self.laplace_stencil).detach().cpu().numpy()[0, 0]
        adv_kernel = (self.beta * self.adv_stencil).detach().cpu().numpy()[0, 0]
        return diff_kernel, adv_kernel

    def get_Ds(self):
        return float(self.alpha.detach().cpu()) * self.dx ** 2 / self.dt

    def constraint_loss(self, *_args, **_kwargs):
        """
        Compatibility with existing training utilities.
        The stencil shapes are hard-constrained, so no additional constraint is needed.
        """
        zero = self.alpha * 0.0
        return zero, zero, zero

if __name__ == "__main__":

    from numerical_solver import torch_diffusion_solver_from_kernel, DiffusionSolverFE

    mu = 0.02  # Diffusion coefficient
    dt = 0.1  # Time step size
    nx = 20  # Number of spatial points
    Lx = 4.0  # Length of the spatial domain
    Nt = 200  # Number of time steps

    dx = Lx/float(nx-1)
    x = np.linspace(0, Lx, nx)  # Spatial grid
    CFL = mu * dt / dx**2

    # boundary and initial condition
    tBC, bBC = 1.0, 0.0
    B = 1.0; a = 0.625; b = 0.0
    u0 = B*np.sin(a * np.pi * x - b)  # Initial condition

    C = mu * dt / dx**2

    #Check 3rd order kernel against numerical solution
    w = [1, -2, 1]
    initial_weights = [w[0]*C, 1+w[1]*C, w[2]*C]
    model = CNNKernelLearner(kernel_size=3, 
                             initial_weights=initial_weights)
    udiff = torch_diffusion_solver_from_kernel(initial_weights, mu, dt, dx, tBC, bBC, u0)
    umodel = model.call_model(u0, 1, include_initial=False)
    for i in range(udiff.shape[0]):
        print("Error Kernel vs numerical solver", i, umodel[0,i] - udiff[i])
        assert np.abs(umodel[0,i] - udiff[i]) < 1e-9


    #Check 5th order kernel against numerical solution
    solver5 = DiffusionSolverFE(mu, dx, p=5)
    u5_test = solver5.solve(u0, dt, 1, bBC, tBC)
    w = [-1/12, 	4/3, 	-5/2, 	4/3, 	-1/12]
    initial_weights = [w[0]*C, w[1]*C, 1+w[2]*C, w[3]*C,  w[4]*C]
    model = CNNKernelLearner(kernel_size=5, 
                             initial_weights=initial_weights)
    umodel = model.call_model(u0, 1, include_initial=True)
    print("5-point max error:", np.max(np.abs(umodel - u5_test)))
    assert np.max(np.abs(umodel - u5_test)) < 1e-6

    #Check 7th order kernel against numerical solution
    solver7 = DiffusionSolverFE(mu, dx, p=7)
    u7_test = solver7.solve(u0, dt, 1, bBC, tBC)
    w = [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
    initial_weights = [w[0]*C, w[1]*C, w[2]*C, 1+w[3]*C, w[4]*C, w[5]*C,  w[6]*C]
    model = CNNKernelLearner(kernel_size=7, 
                             initial_weights=initial_weights)
    umodel = model.call_model(u0, 1, include_initial=True)
    print("7-point max error:", np.max(np.abs(umodel - u7_test)))
    assert np.max(np.abs(umodel - u7_test)) < 1e-6



