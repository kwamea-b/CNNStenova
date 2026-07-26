                    #This code contains both analytical solutions of couette flow and stokes second problem.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import fsolve

def get_initial_sine_parameters_to_match_BC(bBC, tBC, L, wavelength=3):

    """
    Solves for a, b, and B given boundary conditions bBC, tBC, and length L.

    Parameters:
        bBC: Boundary condition at x = 0.
        tBC: Boundary condition at x = L.
        L: Domain length.

    Returns:
        a: Value of a.
        b: Value of b.
        B: Value of B.
    """

    if wavelength < 1e-5:
        return 0., 0., 0.

    # Define the equations to solve
    def equations(vars):
        a, b, B = vars
        eq1 = B * np.sin(-b) - bBC  # Boundary at x = 0
        eq2 = B * np.sin(a * np.pi * L - b) - tBC  # Boundary at x = L
        eq3 = B * np.sin(b)**2 + B * np.cos(b)**2 - B  # Ensure Pythagorean identity
        return [eq1, eq2, eq3]

    # Initial guesses for a, b, and B
    initial_guess = [wavelength/L, 0.1, 1.0]

    # Solve the equations
    solution = fsolve(equations, initial_guess)
    a, b, B = solution

    return a, b, B

#------Analytical solution for diffusion equation (couette flow)-------------

def analytical_solution_diffusion_dirichlet(x, t, mu, L, bBC, tBC, u0_func, nmodes=100):
    """
    Compute the analytical solution for the diffusion equation with Dirichlet BCs:
      u_t = μ u_xx,   u(0,t)=bBC,   u(L,t)=tBC,   u(x,0)=u0_func(x).
    """
    if t == 0:
        return u0_func(x)

    # Steady‐state (linear) between boundaries
    u_steady = bBC + (tBC - bBC) * x / L

    # Transient via Fourier series
    u_transient = np.zeros_like(x)
    for n in range(1, nmodes + 1):
        x_fine = np.linspace(0, L, 1000)
        u0_minus_ss = u0_func(x_fine) - (bBC + (tBC - bBC) * x_fine / L)
        integrand = u0_minus_ss * np.sin(n * np.pi * x_fine / L)
        A_n = (2 / L) * np.trapz(integrand, x_fine)

        eigenvalue = (n * np.pi / L) ** 2
        u_transient += A_n * np.sin(n * np.pi * x / L) * np.exp(-mu * eigenvalue * t)

    return u_steady + u_transient
#
#     return np.array(time_history)
def generate_couette_flow_analytical_slow(mu, dt, dx, nx, Nt, bBC, tBC, u0_func, nmodes=50):
    Lx = (nx - 1) * dx
    x  = np.linspace(0, Lx, nx)
    time_history = []

    for n in range(Nt):
        t = n * dt
        u_t = analytical_solution_diffusion_dirichlet(
            x, t, mu, Lx, bBC, tBC, u0_func, nmodes
        )
        time_history.append(u_t)

    return np.array(time_history)



def modal_unsteady_couette_analytical(x, t, mu, L, bBC, tBC, u0,
                                       nmodes=100, x_loc_for_plotmodes=None):

    """
    Compute the analytical solution of the diffusion equation with specified initial and boundary conditions:
        ∂u/∂t = mu * ∂²u/∂x²
    with:
    - u(0, t) = u0,
    - u(L, t) = uL,
    - u(x, 0) = u0_func(x).
 
    Parameters:
    - x: array of spatial points
    - t: time at which the solution is computed
    - mu: diffusion coefficient
    - L: domain length (x in [0, L])
    - u0: boundary condition at x=0
    - uL: boundary condition at x=L
    - u0_func: function defining the initial condition u(x, 0)
    - nmodes: number of Fourier modes to include
  
    Returns:
    - u: array of solution values at x for time t
    """
 
    if type(t) is float:
        if t==0:
            return u0
 
    # Steady-state solution
    u_steady = bBC + (tBC - bBC) * x / L
  
    # Transient initial condition
    u_transient_0 = u0 - u_steady
 
    # Compute Fourier coefficients A_n
    def fourier_coeff(n, x):
        integrand = u_transient_0 * np.sin(n * np.pi * x / L)
        return (2 / L) * np.trapz(integrand, x)
  
    # Transient solution
    u_transient = np.zeros_like(x)
    for n in range(1, nmodes + 1):
        A_n = np.mean(fourier_coeff(n, x))
        l = n * np.pi / L
        u_ = A_n * np.sin(l * x) * np.exp(-mu * l**2 * t)
        u_transient += u_
        if x_loc_for_plotmodes:
            plt.plot(u_[:,x_loc_for_plotmodes], label="modes ="+str(n))
 
            uhattime = []
            for time in range(u_.shape[0]):
                f, uhat = get_fft(u_[time,:])
                uhattime.append(np.max(uhat))
            print(n, uhattime[0])
            plt.plot(uhattime, '--', label="modes ="+str(n))
            #plt.plot(np.fft.fft(u_[:,x_loc_for_plotmodes],1), '--',
            #         label="modes ="+str(n))
    if x_loc_for_plotmodes:
        plt.legend()
        plt.show()

    # Complete solution
    return u_steady + u_transient


def precompute_fourier_coeffs(mu, L, bBC, tBC, u0_func, nmodes=100, quad_points=2000):
    """
    Precompute steady state function, Fourier coefficients A_n, eigenvalues, and sine basis.
    """
    # Steady-state profile
    u_steady_func = lambda x: bBC + (tBC - bBC) * x / L

    # Integration grid
    x_fine = np.linspace(0, L, quad_points)
    u0_minus_ss = u0_func(x_fine) - u_steady_func(x_fine)

    # Mode numbers
    n_vals = np.arange(1, nmodes + 1)[:, None]  # shape (nmodes, 1)

    # Sine terms for integration
    sine_terms = np.sin(n_vals * np.pi * x_fine / L)  # shape (nmodes, quad_points)

    # Fourier coefficients A_n
    integrals = np.trapz(u0_minus_ss * sine_terms, x_fine, axis=1)  # shape (nmodes,)
    A_n = (2 / L) * integrals

    # Eigenvalues λ_n
    eigenvalues = (n_vals[:, 0] * np.pi / L) ** 2

    return u_steady_func, A_n, eigenvalues


def generate_couette_flow_analytical(mu, dt, dx, nx, Nt, bBC, tBC, u0_func, nmodes=50):
    """
    Fully vectorized analytical Couette flow with Dirichlet BCs.
    """
    Lx = (nx - 1) * dx
    x  = np.linspace(0, Lx, nx)

    # Precompute steady state, coefficients, eigenvalues
    u_steady_func, A_n, eigenvalues = precompute_fourier_coeffs(
        mu, Lx, bBC, tBC, u0_func, nmodes
    )

    # Precompute sine basis for all modes at all x
    n_vals = np.arange(1, nmodes + 1)[:, None]
    sine_basis = np.sin(n_vals * np.pi * x / Lx)  # (nmodes, nx)

    # Precompute exponential decay for all modes at all t (excluding t=0 for now)
    t_vals = np.arange(1, Nt) * dt
    decay = np.exp(-mu * eigenvalues[:, None] * t_vals[None, :])  # (nmodes, Nt-1)

    # Transient part for t>0
    transient = np.einsum("m,mi,mk->ki", A_n, sine_basis, decay)  # (Nt-1, nx)

    # Assemble time history
    time_history = np.empty((Nt, nx))
    time_history[0] = u0_func(x)  # exact initial condition
    time_history[1:] = u_steady_func(x)[None, :] + transient

    return time_history


#----Analytical solution for stokes second problem--------
def analytical_stokes_second_problem(x, t, mu, frequency, amplitude=1.0):
    """
    Analytical solution for Stokes’ second problem (oscillating plate).

    u(x,t) = A * exp(-x * sqrt(ω/(2μ))) * sin(ω t - x * sqrt(ω/(2μ)))
    with u(0,t) = A sin(ω t),  u(∞,t) → 0.
    """
    omega = frequency
    delta = np.sqrt(2 * mu / omega)
    beta = x / delta
    u = amplitude * np.exp(-beta) * np.sin(omega * t - beta)
    return u

def generate_stokes_second_problem_analytical(mu, dt, dx, nx, Nt, frequency):
    """
    Generate analytical Stokes second problem solution (u(0,t)=sin, u(L, t)=0).
    """
    Lx = (nx - 1) * dx
    x = np.linspace(0, Lx, nx)

    time_history = []
    for t_step in range(Nt):
        t = t_step * dt
        u_t = analytical_stokes_second_problem(x, t, mu, frequency)

        # Enforce Dirichlet BCs: bottom oscillates, top = 0
        u_t[0] = np.sin(frequency * t)
        u_t[-1] = 0.0
        time_history.append(u_t)

    return np.array(time_history)

def generate_stokes_swapped_bc(mu, dt, dx, nx, Nt, frequency):
    """
    Generate analytical Stokes second problem with swapped BCs:
    u(0,t) = 0, u(L,t) = sin(ω t).
    Achieved by reflecting x → Lx - x in the standard solution.
    """
    Lx = (nx - 1) * dx
    x = np.linspace(0, Lx, nx)

    time_history = []
    for t_step in range(Nt):
        t = t_step * dt
        # Reflect coordinate: x_rev = Lx - x
        x_rev = Lx - x
        u_temp = analytical_stokes_second_problem(x_rev, t, mu, frequency)

        # Enforce swapped BCs manually:
        u_temp[0] = 0.0  # u(0,t) = 0
        u_temp[-1] = np.sin(frequency * t)  # u(L,t) = sin(ω t)
        time_history.append(u_temp)

    return np.array(time_history)


