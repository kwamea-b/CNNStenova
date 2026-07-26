#-----This code is to contain all numerical solvers used:
#1) Forward euler finite difference 3-point
#2) Forward euler finite difference 5-point
#3) Adams Bashforth 3-point scheme
#4) Adams Bashforth 5-point scheme

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math

                    # --------3-point Forward Euler finite difference-------------
def fd_diffusion_3point_FE(mu, dt, dx, tBC, bBC, u0, time_steps):
    u = u0.copy()
    nx = len(u)
    uhist = [u.copy()]
    for _ in range(time_steps):
        unew = u.copy()
        # enforce Dirichlet BC
        unew[0], unew[-1] = bBC, tBC
        # interior update
        lap = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        unew[1:-1] = u[1:-1] + mu * dt * lap
        u = unew
        uhist.append(u.copy())
    return np.array(uhist)



                #-------5-point Forward Euler finite difference with mirror ghost cells---------
def fd_diffusion_5point_FE_mirror(mu, dt, dx, tBC, bBC, u0, time_steps):
    u = u0.copy()
    nx = len(u)
    uhist = [u.copy()]
    for _ in range(time_steps):
        # build extended array for ghost cells
        u_ext = np.empty(nx + 4)
        u_ext[2:2 + nx] = u
        # mirror ghost cells
        u_ext[1] = 2 * bBC - u_ext[2]
        u_ext[0] = 2 * bBC - u_ext[3]
        u_ext[-2] = 2 * tBC - u_ext[-3]
        u_ext[-1] = 2 * tBC - u_ext[-4]
        # update
        unew = u.copy()
        coeff = dt * mu / (12 * dx ** 2)
        for i in range(nx):
            unew[i] = u[i] + coeff * (
                    -u_ext[i] + 16 * u_ext[i + 1] - 30 * u_ext[i + 2] +
                    16 * u_ext[i + 3] - u_ext[i + 4]
            )
        # enforce BC
        unew[0], unew[-1] = bBC, tBC
        u = unew
        uhist.append(u.copy())
    return np.array(uhist)



def fd_diffusion_7point_FE_mirror(mu, dt, dx, tBC, bBC, u0, time_steps):
    # coeffs: [2, -27, 270, -490, 270, -27, 2]/180
    c = np.array([2, -27, 270, -490, 270, -27, 2]) / 180.0
    u = u0.copy(); nx = len(u); uhist = [u.copy()]
    for _ in range(time_steps):
        # 3 ghosts each side
        u_ext = np.empty(nx + 6)
        u_ext[3:3+nx] = u
        # mirror fill
        u_ext[2] = 2*bBC - u_ext[3]
        u_ext[1] = 2*bBC - u_ext[4]
        u_ext[0] = 2*bBC - u_ext[5]
        u_ext[-3] = 2*tBC - u_ext[-4]
        u_ext[-2] = 2*tBC - u_ext[-5]
        u_ext[-1] = 2*tBC - u_ext[-6]
        # update
        unew = u.copy(); coeff = dt*mu/(dx**2)
        for i in range(nx):
            unew[i] = u[i] + coeff * np.dot(c, u_ext[i:i+7])
        unew[0], unew[-1] = bBC, tBC
        u = unew; uhist.append(u.copy())
    return np.array(uhist)



#This general version should work for 3,5 and 7 versions of code above
class DiffusionSolverFE:
    def __init__(self, mu, dx, p=3):
        if p % 2 == 0:
            raise ValueError("Stencil size p must be odd (3,5,7,...)")
        self.mu = mu
        self.dx = dx
        self.p = p
        self.m = (p - 1) // 2
        self.coeffs = self._compute_coeffs(p)


    def _compute_coeffs(self, p):
        """
        Return weights w_j for approximating f''(0) ≈ sum_j w_j f(x_j),
        using p odd stencil points centered at 0 (x_j = -m..+m, integers).
        Note: these weights are for unit spacing (dx = 1). When applying
        on a grid with spacing dx, divide by dx**2 (or scale later).
        """
        if p % 2 == 0:
            raise ValueError("p must be odd (3,5,7,...)")
        m = self.m
        # stencil node offsets: x_j = -m, ..., 0, ..., +m
        x = np.arange(-m, m+1, dtype=float)   # length p

        # Build matrix M with rows k=0..p-1 and cols j=0..p-1: M[k,j] = x_j**k
        K = p
        M = np.zeros((K, K), dtype=float)
        for k in range(K):
            M[k, :] = x**k

        # RHS: b[k] = k! if k == 2 else 0
        b = np.zeros(K, dtype=float)
        b[2] = math.factorial(2)   # 2! = 2 for second derivative

        # Solve M @ w = b  -> w are the weights corresponding to nodes x_j
        w = np.linalg.solve(M, b)
        return w

    def step(self, u, dt, bBC, tBC):
        nx = len(u)
        unew = u.copy()

        u_ext = np.empty(nx + 2*self.m)
        u_ext[self.m:self.m+nx] = u
        for j in range(self.m):
            u_ext[self.m-1-j] = 2*bBC - u_ext[self.m+j]
            u_ext[self.m+nx+j] = 2*tBC - u_ext[self.m+nx-1-j]
        for i in range(nx):
            stencil = u_ext[i:i+self.p]
            lap = np.dot(self.coeffs, stencil) / self.dx**2
            unew[i] = u[i] + self.mu * dt * lap

        unew[0], unew[-1] = bBC, tBC
        return unew

    def solve(self, u0, dt, time_steps, bBC, tBC):
        u = u0.copy()
        uhist = [u.copy()]
        for _ in range(time_steps):
            u = self.step(u, dt, bBC, tBC)
            uhist.append(u.copy())
        return np.array(uhist)




#--------------------------Adams Bashforth Numerical schemes-----------

#-----------3-point Adams–Bashforth 2 finite difference-----------------
def fd_diffusion_3point_AB2(mu, dt, dx, tBC, bBC, u0, time_steps):
    # bootstrap with one FE step
    uhist_fe = fd_diffusion_3point_FE(mu, dt, dx, tBC, bBC, u0, 1)
    u_prev, u_curr = uhist_fe[0], uhist_fe[1]
    nx = len(u0)
    uhist = [u_prev.copy(), u_curr.copy()]
    for _ in range(time_steps - 1):
        lap_curr = (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]) / dx**2
        lap_prev = (u_prev[2:] - 2*u_prev[1:-1] + u_prev[:-2]) / dx**2
        unew = u_curr.copy()
        unew[1:-1] = u_curr[1:-1] + mu*dt*(1.5*lap_curr - 0.5*lap_prev)
        # enforce BC
        unew[0], unew[-1] = bBC, tBC
        uhist.append(unew.copy())
        u_prev, u_curr = u_curr, unew
    return np.array(uhist)



#---------------5-point Adams–Bashforth 2 finite difference with mirror ghost cells--------------
def fd_diffusion_5point_AB2_mirror(mu, dt, dx, tBC, bBC, u0, time_steps):
    # bootstrap with one 5-point FE step
    uhist_fe = fd_diffusion_5point_FE_mirror(mu, dt, dx, tBC, bBC, u0, 1)
    u_prev, u_curr = uhist_fe[0], uhist_fe[1]
    nx = len(u0)
    uhist = [u_prev.copy(), u_curr.copy()]
    coeff_base = dt * mu / (12 * dx ** 2)
    for _ in range(time_steps - 1):
        # mirror pad helper
        def mirror_pad(arr):
            ext = np.empty(nx + 4)
            ext[2:2 + nx] = arr
            ext[1] = 2 * bBC - ext[2]
            ext[0] = 2 * bBC - ext[3]
            ext[-2] = 2 * tBC - ext[-3]
            ext[-1] = 2 * tBC - ext[-4]
            return ext

        ext_curr = mirror_pad(u_curr)
        ext_prev = mirror_pad(u_prev)
        lap_curr = np.array([(-ext_curr[i] + 16 * ext_curr[i + 1] - 30 * ext_curr[i + 2]
                              + 16 * ext_curr[i + 3] - ext_curr[i + 4])
                             for i in range(nx)])
        lap_prev = np.array([(-ext_prev[i] + 16 * ext_prev[i + 1] - 30 * ext_prev[i + 2]
                              + 16 * ext_prev[i + 3] - ext_prev[i + 4])
                             for i in range(nx)])
        unew = u_curr + coeff_base * (1.5 * lap_curr - 0.5 * lap_prev)
        unew[0], unew[-1] = bBC, tBC
        uhist.append(unew.copy())
        u_prev, u_curr = u_curr, unew
    return np.array(uhist)


def torch_diffusion_solver_from_kernel(kernel, mu, dt, dx, tBC, bBC, u_t, dtype=torch.float32):
    """
    Compute one diffusion step u(t+dt) from u(t) using a user-specified kernel.
    Runs on GPU if available.

    Parameters:
        kernel : np.ndarray of form [C*w[0], 1-C*w[1], C*w[2]]
        mu : float
        dt : float
        dx : float
        tBC, bBC : float
        u_t : np.ndarray (1D)

    Returns:
        u_next : np.ndarray (1D)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move tensors to the selected device
    u = torch.tensor(u_t, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
    kernel_t = torch.tensor(kernel, dtype=dtype, device=device).view(1, 1, -1)
    u_next = F.conv1d(u, kernel_t, padding=len(kernel)//2)

    # Dirichlet boundary conditions
    u_next[:, :, 0] = bBC
    u_next[:, :, -1] = tBC

    # Move result back to CPU before converting to NumPy
    return u_next.squeeze().detach().cpu().numpy()


##Finite difference scheme to solve the diffusion equation
#def finite_difference(mu, dt, dx, tBC, bBC, u0, time_steps, weights=[1, -2, 1]):
#    """
#    Solve the diffusion equation using a finite difference scheme over a
#    given number of time_steps.
#    """
#    u = u0.copy()
#    uhist = []
#    for t in range(time_steps):
#        # Enforce boundary conditions
#        u[0] = bBC
#        u[-1] = tBC
#        uhist.append(u.copy())
#        # Compute Laplacian with periodic boundary handling via np.roll
#        laplacian = (weights[0] * np.roll(u, -1) +
#                     weights[1] * u +
#                     weights[2] * np.roll(u, 1)) / dx ** 2
#        u = u + mu * dt * laplacian
#    return np.array(uhist)

#def generate_couette_flow(mu, dt, dx, nx, Nt, bBC, tBC):
#    """
#    Generate a Couette flow case with specific boundary conditions.
#    Couette flow is the flow of a viscous fluid in the space between two surfaces,
#    one of which is moving relative to the other.
#    """
#    Lx = (nx - 1) * dx
#    x = np.linspace(0, Lx, nx)
#    u0 = np.sin(np.pi * x)  # Initial condition: fluid at rest

#    return finite_difference(mu, dt, dx, tBC, bBC, u0, Nt)



            #Finite difference scheme to solve stokes second problem
def generate_stokes_second_problem(mu, dt, dx, nx, Nt, frequency):
    """
    Generate a Stokes second problem case with a specified oscillation frequency.
    This is the flow of a viscous fluid near an oscillating plate.
    """
    Lx = (nx - 1) * dx
    x = np.linspace(0, Lx, nx)
    u0 = np.zeros_like(x)  # Initial condition: fluid at rest

    # Create time-varying boundary condition function
    def bottom_bc(t):
        return np.sin(frequency * t * dt)

    # Solve with time-varying bottom boundary condition
    u = u0.copy()
    uhist = []
    for t in range(Nt):
        bBC = bottom_bc(t)
        tBC = 0.0  # Top boundary condition fixed at 0

        # Enforce boundary conditions
        u[0] = bBC
        u[-1] = tBC
        uhist.append(u.copy())

        # Compute Laplacian
        laplacian = np.zeros_like(u)
        # Internal points
        for i in range(1, len(u) - 1):
            laplacian[i] = (u[i - 1] - 2 * u[i] + u[i + 1]) / (dx ** 2)

        u = u + mu * dt * laplacian

    return np.array(uhist)


                        #1D Burgers Equation
def torch_burgers_upwind(mu, dt, dx, bBC, tBC, u0, time_steps, device='cpu', periodic=False, forcing=None):
    device = torch.device(device)
    u = torch.tensor(u0, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Diffusion kernel
    k_id = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device).view(1, 1, 3)
    k_diff_scaled = (dt * mu / (dx ** 2)) * torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32, device=device).view(1, 1, 3)
    kernel_diff_combined = k_id + k_diff_scaled

    # Forcing setup
    if forcing is None:
        def get_f(n):
            return torch.zeros((1, 1, u.shape[-1]), dtype=torch.float32, device=device)
    elif callable(forcing):
        def get_f(n):
            f_np = np.asarray(forcing(n), dtype=np.float32)
            return torch.from_numpy(f_np).float().unsqueeze(0).unsqueeze(0).to(device)
    else:
        f_arr = np.asarray(forcing, dtype=np.float32)
        def get_f(n):
            idx = min(n, f_arr.shape[0] - 1)
            return torch.from_numpy(f_arr[idx]).float().unsqueeze(0).unsqueeze(0).to(device)

    u_hist = []

    for n in range(time_steps + 1):
        # Apply boundary conditions
        if not periodic:
            u[:, :, 0] = float(bBC)
            u[:, :, -1] = float(tBC)

        u_hist.append(u[0, 0, :].detach().cpu().numpy().copy())

        if n == time_steps:
            break

        # Padding for convolution
        if periodic:
            u_padded = F.pad(u, (1, 1), mode='circular')
        else:
            left_pad = torch.full((1, 1, 1), float(bBC), dtype=torch.float32, device=device)
            right_pad = torch.full((1, 1, 1), float(tBC), dtype=torch.float32, device=device)
            u_padded = torch.cat([left_pad, u, right_pad], dim=2)

        # Diffusion step
        u_after_diff = F.conv1d(u_padded, kernel_diff_combined, padding=0)

        # Upwind convection
        f_padded = 0.5 * (u_padded ** 2)
        kernel_flux = torch.tensor([-1.0/dx, 1.0/dx, 0.0], dtype=torch.float32, device=device).view(1, 1, 3)
        df_dx = F.conv1d(f_padded, kernel_flux, padding=0)

        # Time update with forcing
        f_n = get_f(n)
        u = u_after_diff - dt * df_dx + dt * f_n

    return np.array(u_hist)


# ---- plot some testing data  ----
if __name__ == "__main__":

    nx = 20
    dx = 1.0 / (nx - 1)
    x = np.linspace(0, 1, nx)
    u0 = np.sin(np.pi * x)   # initial condition
    mu = 0.1
    dt = 1e-4
    steps = 50
    bBC, tBC = 0.0, 0.0

    # Run reference solvers
    u3_ref = fd_diffusion_3point_FE(mu, dt, dx, tBC, bBC, u0, steps)
    u5_ref = fd_diffusion_5point_FE_mirror(mu, dt, dx, tBC, bBC, u0, steps)
    u7_ref = fd_diffusion_7point_FE_mirror(mu, dt, dx, tBC, bBC, u0, steps)

    # Run general solver with p=3 and p=5
    solver3 = DiffusionSolverFE(mu, dx, p=3)
    u3_test = solver3.solve(u0, dt, steps, bBC, tBC)
    print("3-point max error:", np.max(np.abs(u3_ref - u3_test)))
    assert np.max(np.abs(u3_ref - u3_test)) < 1e-14

    solver5 = DiffusionSolverFE(mu, dx, p=5)
    u5_test = solver5.solve(u0, dt, steps, bBC, tBC)
    print("5-point max error:", np.max(np.abs(u5_ref - u5_test)))
    assert np.max(np.abs(u5_ref - u5_test)) < 1e-14

    solver7 = DiffusionSolverFE(mu, dx, p=7)
    u7_test = solver7.solve(u0, dt, steps, bBC, tBC)
    print("7-point max error:", np.max(np.abs(u7_ref - u7_test)))
    assert np.max(np.abs(u7_ref - u7_test)) < 1e-14

    #Check kernel version matches finite difference
    C = mu * dt / dx**2
    w = [1, -2, 1]
    initial_weights = [w[0]*C, 1+w[1]*C, w[2]*C]
    for t in range(u3_ref.shape[0]-1):
        u3_test_kernel = torch_diffusion_solver_from_kernel(initial_weights, mu, dt, dx, tBC, bBC, u3_ref[t,:])
        print("3-point vs. kernel 32 bit max error at time t:", np.max(np.abs(u3_ref[t+1,:] - u3_test_kernel[:])))
        assert np.max(np.abs(u3_ref[t+1,:] - u3_test_kernel[:])) < 1e-6

        #Check high precision works too
        u3_test_kernel = torch_diffusion_solver_from_kernel(initial_weights, mu, dt, dx, tBC, bBC, u3_ref[t,:], dtype=torch.float64)
        print("3-point vs. kernel 64bit max error at time t:", np.max(np.abs(u3_ref[t+1,:] - u3_test_kernel[:])))
        assert np.max(np.abs(u3_ref[t+1,:] - u3_test_kernel[:])) < 1e-14

    #u5_test_kernel = torch_diffusion_solver_from_kernel(initial_weights, mu, dt, dx, tBC, bBC, u0)
    #u7_test_kernel = torch_diffusion_solver_from_kernel(initial_weights, mu, dt, dx, tBC, bBC, u0)
    #print("5-point vs. kernel max error:", np.max(np.abs(u5_ref - u5_test_kernel)))
    #print("7-point vs. kernel max error:", np.max(np.abs(u7_ref - u7_test_kernel)))

