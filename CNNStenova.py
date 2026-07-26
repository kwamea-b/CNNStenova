"""
CNNStenova
===========================
An interactive educational and research software tool for learning
finite-difference stencils and PDE update operators as interpretable
convolutional neural-network kernels.

Run with:
    streamlit run CNNStenova.py

This app reuses your existing scripts:
    - numerical_solver.py
    - analytical_solution.py
    - kernels.py
    - trainers.py

It demonstrates how a finite-difference stencil can be interpreted as:
    1. a numerical operator,
    2. a convolution kernel,
    3. a learnable CNN layer.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
import torch

from analytical_solution import (
    generate_couette_flow_analytical,
    get_initial_sine_parameters_to_match_BC,
)

from numerical_solver import (
    DiffusionSolverFE,
    torch_diffusion_solver_from_kernel,
    torch_burgers_upwind,
    fd_diffusion_3point_FE,
    generate_stokes_second_problem,
)

from kernels import (
    CNNKernelLearner,
    BurgersCNNKernelLearner,
)

from trainers import train_cnn_kernel



# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CNNStenova",
    page_icon="🧪",
    layout="wide",
)

# ---------------------------------------------------------------------------
# App logo
# ---------------------------------------------------------------------------
LOGO_PATH = Path("CNNStenova_logo.png")

if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width="stretch")
else:
    st.warning("CNNStenova logo not found. Check CNNStenova_logo.png")


st.markdown(
    """
    **CNNStenova** is an interactive educational tool for exploring how
    finite-difference update rules can be represented, learned and inspected
    as convolutional neural-network kernels.

    It reuses the analytical solvers, finite-difference solvers, CNN kernel
    learners and training routines from the
    [CNN numerical schemes repository](https://github.com/kwamea-b/CNN_numerical_schemes).
    """
)

st.divider()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def initial_condition_factory(name, bBC, tBC, L, B=1.0, a=1.0, b=0.0):
    """
    Return an initial condition function compatible with the analytical
    Couette-flow diffusion solver.

    The main run.py-style initial condition is:

        u0(x) = B sin(a pi x - b)

    with the first and last values forced to satisfy the Dirichlet
    boundary conditions.
    """

    if name == "Custom sine: B sin(aπx - b)":
        def u0_func(xv):
            vals = B * np.sin(a * np.pi * xv - b)
            vals[0] = bBC
            vals[-1] = tBC
            return vals.astype(np.float32)

        return u0_func

    if name == "Auto sine matching BC":
        a_auto, b_auto, B_auto = get_initial_sine_parameters_to_match_BC(
            bBC=bBC,
            tBC=tBC,
            L=L,
            wavelength=3,
        )

        def u0_func(xv):
            vals = B_auto * np.sin(a_auto * np.pi * xv - b_auto)
            vals[0] = bBC
            vals[-1] = tBC
            return vals.astype(np.float32)

        return u0_func

    if name == "Zero field":
        def u0_func(xv):
            vals = np.zeros_like(xv)
            vals[0] = bBC
            vals[-1] = tBC
            return vals.astype(np.float32)

        return u0_func

    if name == "Gaussian pulse":
        def u0_func(xv):
            vals = np.exp(-80.0 * (xv - 0.5 * L) ** 2)
            vals[0] = bBC
            vals[-1] = tBC
            return vals.astype(np.float32)

        return u0_func

    def default_u0_func(xv):
        vals = B * np.sin(a * np.pi * xv - b)
        vals[0] = bBC
        vals[-1] = tBC
        return vals.astype(np.float32)

    return default_u0_func

def make_heatmap(data, x, t, title, cmap="RdYlBu_r"):
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap=cmap,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("time")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="u")
    fig.tight_layout()
    return fig


def scale_update_kernel_to_laplacian(update_kernel, cfl):
    """
    Convert a learned update kernel into a Laplacian-style stencil.

    For explicit diffusion, the update kernel is approximately:
        [C, 1 - 2C, C]

    The corresponding Laplacian stencil is:
        [(w_-1)/C, (w_0 - 1)/C, (w_+1)/C]
    """
    update_kernel = np.asarray(update_kernel, dtype=float).copy()

    if abs(cfl) < 1e-12:
        return np.full_like(update_kernel, np.nan)

    scaled = update_kernel / cfl
    mid = len(update_kernel) // 2
    scaled[mid] = (update_kernel[mid] - 1.0) / cfl

    return scaled

def von_neumann_amplification(kernel, num_points=1000):
    """
    Compute Von Neumann amplification factor for a 1D full update kernel.

    For the CNNKernelLearner used in CNNStenova:

        u^{n+1} = K * u^n

    so the learned CNN weights are already the full update stencil.

    Parameters
    ----------
    kernel : array-like
        Learned full update kernel, e.g. [w_-1, w_0, w_+1].
    num_points : int
        Number of wavenumber samples.

    Returns
    -------
    theta : ndarray
        Wavenumber values in [0, pi].
    G : ndarray
        Complex amplification factor.
    amp : ndarray
        Amplification magnitude |G(theta)|.
    max_amp : float
        Maximum amplification magnitude.
    stable : bool
        True if max |G(theta)| <= 1.
    """

    kernel = np.asarray(kernel, dtype=float).reshape(-1)

    if kernel.size % 2 != 1:
        raise ValueError("Kernel size must be odd for centred Von Neumann analysis.")

    m = kernel.size // 2
    j_values = np.arange(-m, m + 1)

    theta = np.linspace(0.0, np.pi, num_points)

    G = np.zeros_like(theta, dtype=np.complex128)

    for w_j, j in zip(kernel, j_values):
        G += w_j * np.exp(1j * j * theta)

    amp = np.abs(G)
    max_amp = float(np.max(amp))
    stable = bool(max_amp <= 1.0 + 1e-12)

    return theta, G, amp, max_amp, stable

def make_von_neumann_plot(theta, amp, max_amp):
    """
    Plot the Von Neumann amplification spectrum |G(theta)|.
    """

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(theta, amp, linewidth=2, label=r"$|G(\theta)|$")
    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=2,
        label="stability limit",
    )

    ax.set_xlabel(r"Wavenumber $\theta = k\Delta x$")
    ax.set_ylabel(r"Amplification factor $|G(\theta)|$")
    ax.set_title(f"Von Neumann stability spectrum, max |G| = {max_amp:.4f}")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    return fig

def make_fbeq_forcing_history(x, dt, time_steps, L, forcing_scale=1.0, seed=42):
    """
    Multi-mode forcing used for the forced Burgers equation example.

    f(x,t) = sum_i A_i sin(omega_i t + 2π l_i x/L + phi_i)

    Returns
    -------
    forcing_history : ndarray
        Shape (time_steps + 1, nx)
    """

    N = 6

    A = forcing_scale * np.array(
        [0.12, 0.08, 0.06, 0.04, 0.03, 0.02],
        dtype=np.float32,
    )

    omega = np.array(
        [0.6, 0.5, 0.3, 0.2, 0.15, 0.1],
        dtype=np.float32,
    )

    l = np.array(
        [1, 2, 3, 4, 6, 8],
        dtype=np.int32,
    )

    rng = np.random.default_rng(seed=seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=N).astype(np.float32)

    times = np.arange(time_steps + 1, dtype=np.float32) * dt

    t_col = times[:, None, None]
    omega_row = omega[None, :, None]
    l_row = l[None, :, None]
    phi_row = phi[None, :, None]
    x_row = x[None, None, :]
    A_row = A[None, :, None]

    k_row = 2.0 * np.pi * l_row / float(L)

    modes = A_row * np.sin(
        omega_row * t_col + k_row * x_row + phi_row
    )

    forcing_history = np.sum(modes, axis=1).astype(np.float32)

    return forcing_history


@st.cache_data(show_spinner=False)
def cached_diffusion_solution(
    mu,
    dt,
    dx,
    nx,
    Nt,
    bBC,
    tBC,
    initial_condition_name,
    nmodes,
    scheme_order,
    B,
    a,
    b,
):
    L = (nx - 1) * dx

    u0_func = initial_condition_factory(
        initial_condition_name,
        bBC,
        tBC,
        L,
        B=B,
        a=a,
        b=b,
    )

    x = np.linspace(0.0, L, nx)
    t = np.arange(Nt) * dt

    u0 = u0_func(x)
    u0[0], u0[-1] = bBC, tBC

    solver = DiffusionSolverFE(mu=mu, dx=dx, p=scheme_order)

    numerical = np.array(
        solver.solve(u0, dt, Nt - 1, bBC, tBC),
        dtype=np.float32,
    )

    stencil = np.array(solver.coeffs, dtype=np.float32)

    analytical = generate_couette_flow_analytical(
        mu=mu,
        dt=dt,
        dx=dx,
        nx=nx,
        Nt=Nt,
        bBC=bBC,
        tBC=tBC,
        u0_func=u0_func,
        nmodes=nmodes,
    )

    analytical[:, 0] = bBC
    analytical[:, -1] = tBC

    return x, t, u0, numerical, analytical, stencil


@st.cache_data(show_spinner=False)
def cached_fbeq_burgers_reference(
    L,
    nx,
    dt,
    T_final,
    mu,
    periodic,
    forcing_scale,
):
    """
    Generate the FBEQ reference solution used in the Burgers CNN lab.
    """

    dx = L / float(nx - 1)
    time_steps = int(T_final / dt)

    x = np.linspace(0.0, L, nx)

    # FBEQ initial condition
    u0 = 0.5 * np.sin(2.0 * np.pi * x / L) * np.exp(-x)
    u0 = u0.astype(np.float32)

    if periodic:
        bBC, tBC = None, None
    else:
        bBC, tBC = 0.0, 1.0
        u0[0] = bBC
        u0[-1] = tBC

    forcing_history = make_fbeq_forcing_history(
        x=x,
        dt=dt,
        time_steps=time_steps,
        L=L,
        forcing_scale=forcing_scale,
        seed=42,
    )

    solution = torch_burgers_upwind(
        mu=mu,
        dt=dt,
        dx=dx,
        bBC=bBC,
        tBC=tBC,
        u0=u0,
        time_steps=time_steps,
        device="cpu",
        periodic=periodic,
        forcing=forcing_history,
    )

    solution = np.asarray(solution, dtype=np.float32)
    t = np.arange(solution.shape[0]) * dt

    C_diff = mu * dt / dx**2
    C_conv_up = dt / dx
    CFL_conv = dt * np.max(np.abs(u0)) / dx

    return (
        x,
        t,
        u0,
        solution,
        forcing_history,
        dx,
        time_steps,
        C_diff,
        C_conv_up,
        CFL_conv,
        bBC,
        tBC,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("CNNStenova Controls")

mode = st.sidebar.radio(
    "Choose lab",
    [
        "Diffusion stencil lab",
        "Learnable CNN kernel lab",
        "Forced Burgers CNN lab",
    ],
)

st.sidebar.divider()
mu = st.sidebar.number_input(
    "Diffusivity / viscosity μ",
    min_value=0.0001,
    max_value=10.0,
    value=0.02,
    step=0.001,
    format="%.4f",
)

dt = st.sidebar.number_input(
    "Time step Δt",
    min_value=0.00001,
    max_value=1.0,
    value=0.1,
    step=0.001,
    format="%.5f",
)

nx = st.sidebar.number_input(
    "Spatial points nx",
    min_value=3,
    max_value=1000,
    value=20,
    step=1,
)

L = st.sidebar.number_input(
    "Domain length L",
    min_value=0.1,
    max_value=100.0,
    value=4.0,
    step=0.1,
    format="%.4f",
)

Nt = st.sidebar.number_input(
    "Number of time frames Nt",
    min_value=2,
    max_value=50000,
    value=200,
    step=1,
)

nx = int(nx)
Nt = int(Nt)

dx = L / float(nx - 1)

CFL = mu * dt / dx**2

st.sidebar.metric(
    "Diffusion number μΔt/Δx²",
    f"{CFL:.4f}",
)

if CFL <= 0.5:
    st.sidebar.success("Stable for explicit 3-point diffusion.")
else:
    st.sidebar.error("Likely unstable for explicit 3-point diffusion.")

# ---------------------------------------------------------------------------
# Shared flow configuration
# ---------------------------------------------------------------------------
if "flow_config" not in st.session_state:
    st.session_state.flow_config = {}

def update_flow_config(
    mu,
    dt,
    dx,
    nx,
    Nt,
    L,
    bBC,
    tBC,
    initial_condition_name,
    nmodes,
    scheme_order,
    B,
    a,
    b,
):
    """
    Store the currently described flow so all labs use the same configuration.
    """
    st.session_state.flow_config = {
        "mu": float(mu),
        "dt": float(dt),
        "dx": float(dx),
        "nx": int(nx),
        "Nt": int(Nt),
        "L": float(L),
        "bBC": float(bBC),
        "tBC": float(tBC),
        "initial_condition_name": initial_condition_name,
        "nmodes": int(nmodes),
        "scheme_order": int(scheme_order),
        "B": float(B),
        "a": float(a),
        "b": float(b),
    }


# ---------------------------------------------------------------------------
# Lab 1: Diffusion stencil visualiser
# ---------------------------------------------------------------------------
if mode == "Diffusion stencil lab":
    st.header("1. Diffusion stencil lab")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        bBC = st.number_input(
            "Bottom/left boundary bBC",
            value=0.0,
            step=0.1,
        )

    with col_b:
        tBC = st.number_input(
            "Top/right boundary tBC",
            value=0.0,
            step=0.1,
        )

    with col_c:
        scheme_order = st.selectbox(
            "Stencil order",
            [3, 5, 7],
            index=0,
        )

    initial_condition_name = st.selectbox(
        "Initial condition",
        [
            "Custom sine: B sin(aπx - b)",
            "Auto sine matching BC",
            "Zero field",
            "Gaussian pulse",
        ],
        index=0,
    )

    col_ic1, col_ic2, col_ic3 = st.columns(3)

    with col_ic1:
        B = st.number_input(
            "Amplitude B",
            value=1.0,
            step=0.1,
            format="%.4f",
        )

    with col_ic2:
        a = st.number_input(
            "Frequency parameter a",
            value=1.0,
            step=0.1,
            format="%.4f",
        )

    with col_ic3:
        b = st.number_input(
            "Phase shift b",
            value=0.0,
            step=0.1,
            format="%.4f",
        )

    nmodes = st.slider(
        "Fourier modes for analytical solution",
        10,
        200,
        50,
        step=10,
    )

    # Save this described flow for the CNN lab
    update_flow_config(
        mu=mu,
        dt=dt,
        dx=dx,
        nx=nx,
        Nt=Nt,
        L=L,
        bBC=bBC,
        tBC=tBC,
        initial_condition_name=initial_condition_name,
        nmodes=nmodes,
        scheme_order=scheme_order,
        B=B,
        a=a,
        b=b,
    )

    x, t, u0, numerical, analytical, stencil = cached_diffusion_solution(
        mu,
        dt,
        dx,
        nx,
        Nt,
        bBC,
        tBC,
        initial_condition_name,
        nmodes,
        scheme_order,
        B,
        a,
        b,
    )

    error = np.abs(numerical - analytical)
    rmse_time = np.sqrt(np.mean(error**2, axis=1))

    st.markdown("### Classical stencil")
    st.code(
        np.array2string(stencil, precision=6),
        language="text",
    )

    st.markdown(
        """
        The diffusion stencil is a curvature detector. In the 3-point case, the classical
        second-derivative stencil is `[1, -2, 1]`. The centre loses intensity while
        neighbouring points receive intensity, producing smoothing over time.
        """
    )

    frame = st.slider(
        "Frame to inspect",
        0,
        numerical.shape[0] - 1,
        numerical.shape[0] // 2,
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, analytical[frame], label="Analytical")
        ax.plot(x, numerical[frame], "--", label="Numerical")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(f"Analytical vs numerical at frame {frame}")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, rmse_time)
        ax.set_xlabel("time")
        ax.set_ylabel("RMSE")
        ax.set_title("Numerical error against analytical solution")
        ax.grid(True)
        fig.tight_layout()
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.pyplot(
            make_heatmap(
                numerical,
                x,
                t,
                "Numerical diffusion solution",
            )
        )

    with col4:
        st.pyplot(
            make_heatmap(
                error,
                x,
                t,
                "Absolute error",
                cmap="viridis",
            )
        )

    st.markdown("### Kernel interpretation")

    st.latex(
        r"u_i^{n+1}=u_i^n + \mu\frac{\Delta t}{\Delta x^2}"
        r"\left(u_{i+1}^n-2u_i^n+u_{i-1}^n\right)"
    )

    st.markdown(
        """
        This is also a convolution operation. For the 3-point explicit diffusion scheme,
        the full update kernel is approximately:
        """
    )

    update_kernel = np.array(
        [
            CFL,
            1.0 - 2.0 * CFL,
            CFL,
        ]
    )

    st.code(
        np.array2string(update_kernel, precision=6),
        language="text",
    )


# ---------------------------------------------------------------------------
# Lab 2: Learnable CNN kernel
# ---------------------------------------------------------------------------
elif mode == "Learnable CNN kernel lab":
    st.header("2. Learnable CNN kernel lab")

    flow_config = st.session_state.get("flow_config", None)

    if not flow_config:
        st.warning(
            "No saved flow description found. Go to the Diffusion stencil lab first, "
            "set the flow parameters, then return here to train the CNN kernel."
        )
        st.stop()

    mu_flow = flow_config["mu"]
    dt_flow = flow_config["dt"]
    dx_flow = flow_config["dx"]
    nx_flow = flow_config["nx"]
    Nt_flow = flow_config["Nt"]
    L_flow = flow_config["L"]
    bBC_flow = flow_config["bBC"]
    tBC_flow = flow_config["tBC"]
    initial_condition_name_flow = flow_config["initial_condition_name"]
    nmodes_flow = flow_config["nmodes"]
    scheme_order_flow = flow_config["scheme_order"]
    B_flow = flow_config["B"]
    a_flow = flow_config["a"]
    b_flow = flow_config["b"]

    CFL_flow = mu_flow * dt_flow / dx_flow ** 2

    st.markdown("### Flow description used for CNN training")

    st.code(
        f"""
    mu = {mu_flow}
    dt = {dt_flow}
    nx = {nx_flow}
    L = {L_flow}
    Nt = {Nt_flow}
    dx = {dx_flow}
    bBC = {bBC_flow}
    tBC = {tBC_flow}
    initial_condition = {initial_condition_name_flow}
    B = {B_flow}
    a = {a_flow}
    b = {b_flow}
    scheme_order = {scheme_order_flow}
    diffusion_number = {CFL_flow}
    """,
        language="text",
    )

    st.markdown(
        """
        This lab trains your existing `CNNKernelLearner` on diffusion data generated by your
        existing numerical/analytical scripts. The goal is to show that a CNN kernel can
        recover a finite-difference stencil.
        """
    )

    source = st.radio(
        "Training data source",
        [
            "Numerical",
            "Analytical",
        ],
        horizontal=True,
    )

    epochs = st.slider(
        "Training epochs",
        min_value=50,
        max_value=60000,
        value=30000,
        step=50,
    )

    st.caption(
        "Higher epochs give the CNN kernel more time to converge, but training may take longer inside Streamlit."
    )

    learning_rate = st.select_slider(
        "Learning rate",
        options=[
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            1e-2,
        ],
        value=1e-3,
    )

    st.markdown("### PINN / moment constraints")

    use_constraints = st.checkbox(
        "Enforce stencil moment constraints",
        value=False,
    )

    if use_constraints:
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            lambda_1 = st.number_input(
                "λ₁: sum(w)=1",
                value=0.1,
                min_value=0.0,
                step=0.1,
                format="%.4f",
            )

        with col_c2:
            lambda_2 = st.number_input(
                "λ₂: first moment=0",
                value=0.1,
                min_value=0.0,
                step=0.1,
                format="%.4f",
            )

        with col_c3:
            lambda_3 = st.number_input(
                "λ₃: second moment=2C",
                value=0.1,
                min_value=0.0,
                step=0.1,
                format="%.4f",
            )

        l123 = [lambda_1, lambda_2, lambda_3]
    else:
        l123 = False

    train_button = st.button("Train CNN kernel")

    # bBC = 0.0
    # tBC = 0.0
    # initial_condition_name = "Sine wave"
    # B = 1.0
    # a = 1.0
    # b = 0.0

    x, t, u0, numerical, analytical, stencil = cached_diffusion_solution(
        mu_flow,
        dt_flow,
        dx_flow,
        nx_flow,
        Nt_flow,
        bBC_flow,
        tBC_flow,
        initial_condition_name_flow,
        nmodes_flow,
        scheme_order_flow,
        B_flow,
        a_flow,
        b_flow,
    )

    training_data = numerical if source == "Numerical" else analytical
    training_data = np.array(training_data, dtype=np.float32)

    st.markdown("### Reference data")

    st.pyplot(
        make_heatmap(
            training_data,
            x,
            t,
            f"{source} training data",
        )
    )

    if train_button:
        with st.spinner("Training CNNKernelLearner using the saved flow configuration..."):

            device = "cpu"
            kernel_size = 3
            multistep = 1

            if source == "Numerical":
                tol = 1e-13
            else:
                tol = 1e-10

            utrain = np.array(training_data, dtype=np.float32)

            # Same diffusion number as run.py
            CFL_train = mu_flow * dt_flow / dx_flow ** 2

            # Same initialisation style as run.py
            w = np.random.random(3)
            initial_weights = [
                w[0] * CFL_train,
                1.0 + w[1] * CFL_train,
                w[2] * CFL_train,
            ]

            model = CNNKernelLearner(
                kernel_size=kernel_size,
                initial_weights=initial_weights,
                device=device,
            )

            start_weight = model.conv.weight[:].detach().numpy().copy()

            trained_model, conv_hist = train_cnn_kernel(
                model,
                utrain,
                dt_flow,
                dx_flow,
                mu_flow,
                added_constraints=l123,
                epochs=epochs,
                multistep=multistep,
                learning_rate=learning_rate,
                conv_stats=True,
                breaktol=tol,
            )

            model = trained_model

        learned_update_kernel = model.conv.weight.detach().cpu().numpy()[0, 0]

        learned_laplacian_stencil = scale_update_kernel_to_laplacian(
            learned_update_kernel,
            CFL_train,
        )

        cnn_solution = model.call_model(
            u0,
            time_steps=Nt_flow - 1,
            tBC=tBC_flow,
            bBC=bBC_flow,
            include_initial=True,
        )

        # ------------------------------------------------------------------
        # Compare CNN solution against the selected reference dataset
        # ------------------------------------------------------------------
        solution = training_data  # selected dataset: numerical or analytical

        T_min = min(cnn_solution.shape[0], solution.shape[0])

        cnn_solution_subset = cnn_solution[:T_min]
        solution_subset = solution[:T_min]

        cnn_error = np.abs(cnn_solution_subset - solution_subset)

        time_subset = np.arange(T_min) * dt_flow

        st.success("Training complete.")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Final loss",
            f"{conv_hist[-1, 0]:.3e}",
        )

        col2.metric(
            "Diffusion number",
            f"{CFL_train:.4f}",
        )

        col3.metric(
            "Frames",
            str(Nt_flow),
        )

        st.markdown("### Learned CNN update kernel")

        st.code(
            np.array2string(learned_update_kernel, precision=8),
            language="text",
        )
        # ------------------------------------------------------------------
        # Von Neumann stability analysis of the learned CNN update kernel
        # ------------------------------------------------------------------
        theta_vn, G_vn, amp_vn, max_amp_vn, stable_vn = von_neumann_amplification(
            learned_update_kernel,
            num_points=1000,
        )

        st.markdown("### Von Neumann stability analysis")

        col_vn1, col_vn2, col_vn3 = st.columns(3)

        with col_vn1:
            st.metric(
                "max |G(θ)|",
                f"{max_amp_vn:.6f}",
            )

        with col_vn2:
            st.metric(
                "Stability condition",
                "Satisfied" if stable_vn else "Violated",
            )

        with col_vn3:
            st.metric(
                "Criterion",
                "≤ 1",
            )

        if stable_vn:
            st.success(
                "The learned CNN update kernel satisfies the Von Neumann condition "
                "max |G(θ)| ≤ 1 for the sampled wavenumbers."
            )
        else:
            st.error(
                "The learned CNN update kernel violates the Von Neumann condition. "
                "Some Fourier modes may be amplified during rollout."
            )

        st.pyplot(
            make_von_neumann_plot(
                theta_vn,
                amp_vn,
                max_amp_vn,
            )
        )

        st.markdown(
            """
            The learned CNN kernel is analysed as a full update operator:
            """
        )

        st.latex(
            r"u_i^{n+1} = \sum_j w_j u_{i+j}^{n}"
        )

        st.markdown(
            """
            Substituting a Fourier mode gives the amplification factor:
            """
        )

        st.latex(
            r"G(\theta)=\sum_j w_j e^{ij\theta}"
        )

        st.markdown(
            """
            Stability requires:
            """
        )

        st.latex(
            r"\max_{\theta} |G(\theta)| \leq 1"
        )

        st.markdown(
            """
            This directly addresses whether the learned convolution operator amplifies
            or damps resolvable Fourier modes.
            """
        )

        st.markdown("### Scaled finite-difference stencil")

        st.code(
            np.array2string(learned_laplacian_stencil, precision=6),
            language="text",
        )

        st.markdown(
            """
            If the CNN has learned the numerical diffusion update correctly, the scaled stencil
            should approach the classical finite-difference stencil `[1, -2, 1]`.
            """
        )

        col4, col5 = st.columns(2)

        with col4:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(conv_hist[:, 0])
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.set_title("Training loss")
            ax.grid(True)
            fig.tight_layout()
            st.pyplot(fig)

        with col5:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(conv_hist[:, 1], label=r"$w_{-1}$")
            ax.plot(conv_hist[:, 2], label=r"$w_0$")
            ax.plot(conv_hist[:, 3], label=r"$w_{+1}$")
            ax.set_xlabel("epoch")
            ax.set_ylabel("weight")
            ax.set_title("Kernel coefficient evolution")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            st.pyplot(fig)

        st.markdown("### CNN comparison against selected reference solution")

        col_true, col_cnn, col_err = st.columns(3)

        with col_true:
            st.pyplot(
                make_heatmap(
                    solution_subset,
                    x,
                    time_subset,
                    f"True Solution ({source})",
                )
            )

        with col_cnn:
            st.pyplot(
                make_heatmap(
                    cnn_solution_subset,
                    x,
                    time_subset,
                    "CNN Solution",
                )
            )

        with col_err:
            st.pyplot(
                make_heatmap(
                    cnn_error,
                    x,
                    time_subset,
                    "Absolute Error",
                    cmap="viridis",
                )
            )
    else:
        st.info(
            "Click **Train CNN kernel** to train on the saved flow from the Diffusion stencil lab."
        )


# ---------------------------------------------------------------------------
# Lab 3: Forced Burgers Equation CNN kernel lab
# ---------------------------------------------------------------------------
else:
    st.header("3. Forced Burgers Equation (FBEQ) CNN kernel lab")

    st.markdown(
        """
        This lab uses the **forced Burgers equation (FBEQ)** setup. It generates a
        forced viscous Burgers reference solution, trains the `BurgersCNNKernelLearner`,
        and compares the CNN rollout against the reference solution.

        The Burgers CNN uses three convolutional input channels:

        1. velocity field `u`,
        2. nonlinear flux channel `u²/2`,
        3. forcing channel `f(x,t)`.
        """
    )

    st.markdown("### FBEQ numerical settings")

    with st.expander("Edit FBEQ settings", expanded=False):
        col_b1, col_b2, col_b3 = st.columns(3)

        with col_b1:
            L_b = st.number_input(
                "FBEQ domain length L",
                min_value=0.1,
                max_value=20.0,
                value=4.0,
                step=0.1,
                format="%.4f",
                key="fbeq_L",
            )

        with col_b2:
            nx_b = st.number_input(
                "FBEQ spatial points nx",
                min_value=20,
                max_value=1000,
                value=250,
                step=10,
                key="fbeq_nx",
            )

        with col_b3:
            mu_b = st.number_input(
                "FBEQ viscosity μ",
                min_value=0.0001,
                max_value=1.0,
                value=0.05,
                step=0.001,
                format="%.4f",
                key="fbeq_mu",
            )

        col_b4, col_b5, col_b6 = st.columns(3)

        with col_b4:
            dt_b = st.number_input(
                "FBEQ time step Δt",
                min_value=0.0001,
                max_value=0.1,
                value=0.002,
                step=0.0005,
                format="%.5f",
                key="fbeq_dt",
            )

        with col_b5:
            T_final_b = st.number_input(
                "FBEQ final time",
                min_value=0.1,
                max_value=50.0,
                value=2.0,
                step=0.5,
                format="%.3f",
                key="fbeq_T_final",
            )

        with col_b6:
            forcing_scale = st.slider(
                "Forcing scale",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.05,
                key="fbeq_forcing_scale",
            )

    periodic = st.checkbox(
        "Periodic boundary condition",
        value=True,
        key="fbeq_periodic",
    )

    burgers_epochs = st.slider(
        "Burgers CNN training epochs",
        min_value=100,
        max_value=50000,
        value=3000,
        step=100,
        key="fbeq_epochs",
    )

    burgers_lr = st.select_slider(
        "Burgers learning rate",
        options=[1e-5, 3e-5, 7e-5, 1e-4, 3e-4, 1e-3],
        value=7e-5,
        key="fbeq_lr",
    )

    # -----------------------------------------------------------------------
    # Generate/train controls
    # -----------------------------------------------------------------------
    generate_reference_button = st.button(
        "Generate FBEQ reference solution",
        key="generate_fbeq_reference",
    )

    train_burgers_button = st.button(
        "Train FBEQ Burgers CNN kernel",
        key="train_fbeq_burgers",
    )

    # A signature records the current numerical settings.
    # If the user changes these settings after generating the reference,
    # we ask them to regenerate the reference solution.
    fbeq_signature = (
        float(L_b),
        int(nx_b),
        float(dt_b),
        float(T_final_b),
        float(mu_b),
        bool(periodic),
        float(forcing_scale),
    )

    if "fbeq_reference_data" not in st.session_state:
        st.session_state.fbeq_reference_data = None

    if "fbeq_reference_signature" not in st.session_state:
        st.session_state.fbeq_reference_signature = None

    # Only generate the expensive FBEQ reference solution when the button is clicked.
    if generate_reference_button:
        with st.spinner("Generating FBEQ reference solution..."):
            st.session_state.fbeq_reference_data = cached_fbeq_burgers_reference(
                L=L_b,
                nx=int(nx_b),
                dt=dt_b,
                T_final=T_final_b,
                mu=mu_b,
                periodic=periodic,
                forcing_scale=forcing_scale,
            )

            st.session_state.fbeq_reference_signature = fbeq_signature

        st.success("FBEQ reference solution generated.")

    # Stop here if no reference solution has been generated yet.
    if st.session_state.fbeq_reference_data is None:
        st.info(
            "Click **Generate FBEQ reference solution** first. "
            "This avoids running the Burgers solver automatically when the page loads."
        )
        st.stop()

    # If settings changed after reference generation, ask user to regenerate.
    if st.session_state.fbeq_reference_signature != fbeq_signature:
        st.warning(
            "The FBEQ settings have changed since the reference solution was generated. "
            "Click **Generate FBEQ reference solution** again before training."
        )
        st.stop()

    # Retrieve the generated reference solution from session state.
    (
        x_b,
        t_b,
        u0_b,
        burgers_ref,
        forcing_history,
        dx_b,
        time_steps_b,
        C_diff_b,
        C_conv_up_b,
        CFL_conv_b,
        bBC_b,
        tBC_b,
    ) = st.session_state.fbeq_reference_data

    st.markdown("### FBEQ stability and setup summary")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        st.metric("Spatial step Δx", f"{dx_b:.5f}")

    with col_s2:
        st.metric("Time steps", str(time_steps_b))

    with col_s3:
        st.metric("Diffusion number", f"{C_diff_b:.4f}")

    with col_s4:
        st.metric("Convective CFL", f"{CFL_conv_b:.4f}")

    if C_diff_b <= 0.5:
        st.success("Diffusion number is within the 1D explicit diffusion bound.")
    else:
        st.warning(
            "Diffusion number is large. The reference Burgers solver may become unstable."
        )

    if time_steps_b > 10000 or burgers_epochs > 50000:
        st.warning(
            "This is a heavy FBEQ run. The full FBEQ setting uses many time steps and "
            "up to 200,000 epochs, so training inside Streamlit may take a long time."
        )

    frame = st.slider(
        "FBEQ frame",
        0,
        burgers_ref.shape[0] - 1,
        burgers_ref.shape[0] // 2,
        key="fbeq_frame",
    )

    st.markdown("### Reference forced Burgers solution")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_b, burgers_ref[0], label="Initial")
        ax.plot(x_b, burgers_ref[frame], label=f"Frame {frame}")
        ax.plot(x_b, burgers_ref[-1], label="Final")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title("Forced viscous Burgers profiles")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        # Downsample only for visual display if the time history is large
        stride = max(1, burgers_ref.shape[0] // 800)
        st.pyplot(
            make_heatmap(
                burgers_ref[::stride],
                x_b,
                t_b[::stride],
                "Reference FBEQ Burgers solution",
            )
        )

    st.markdown("### Forced Burgers equation")

    st.latex(
        r"u_t + u u_x = \mu u_{xx} + f(x,t)"
    )

    st.markdown(
        """
        The CNN model learns a residual update of the form:
        """
    )

    st.latex(
        r"u^{n+1} = u^n + \mathrm{Conv}\left[u,\frac{u^2}{2},f\right]"
    )

    st.markdown(
        """
        This means the CNN is not a generic fully connected model. It is a
        convolutional update operator that acts locally on velocity, nonlinear
        flux, and forcing channels.
        """
    )

    # -----------------------------------------------------------------------
    # Train Burgers CNN kernel
    # -----------------------------------------------------------------------
    if train_burgers_button:
        with st.spinner("Training FBEQ BurgersCNNKernelLearner..."):

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = BurgersCNNKernelLearner(
                bBC=bBC_b,
                tBC=tBC_b,
                kernel_size=3,
                in_channels=3,
                residual=True,
                device=device,
                pad_mode="mirror",
                bias=False,
                physical_init=True,
                noise_factor=0.0001,
                periodic=periodic,
                C_diff_init=C_diff_b,
                C_conv_init=C_conv_up_b,
            ).to(device)

            trained_model, conv_hist = train_cnn_kernel(
                model,
                burgers_ref,
                dt_b,
                dx_b,
                mu_b,
                epochs=burgers_epochs,
                learning_rate=burgers_lr,
                breaktol=1e-14,
                scheduler=True,
                forcing_history=forcing_history,
                grad_clip=1.0,
                conv_stats=True,
            )

            pred_steps = time_steps_b

            cnn_solution = trained_model.predict_forward(
                u0_b,
                pred_steps,
                forcing_history=forcing_history,
            )

            cnn_solution = np.asarray(cnn_solution, dtype=np.float32)

        T_min = min(cnn_solution.shape[0], burgers_ref.shape[0])

        burgers_ref_subset = burgers_ref[:T_min]
        cnn_solution_subset = cnn_solution[:T_min]

        abs_error = np.abs(cnn_solution_subset - burgers_ref_subset)
        t_subset = t_b[:T_min]

        st.success("FBEQ Burgers CNN training complete.")

        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            st.metric(
                "Final loss",
                f"{conv_hist[-1, 0]:.3e}",
            )

        with col_m2:
            st.metric(
                "Max absolute error",
                f"{np.max(abs_error):.3e}",
            )

        with col_m3:
            st.metric(
                "Mean absolute error",
                f"{np.mean(abs_error):.3e}",
            )

        # -------------------------------------------------------------------
        # Learned Burgers kernels
        # -------------------------------------------------------------------
        weights = trained_model.conv.weight.detach().cpu().numpy()[0]

        st.markdown("### Learned FBEQ Burgers CNN kernels")

        st.markdown("**Velocity channel kernel**")
        st.code(
            np.array2string(weights[0], precision=8),
            language="text",
        )

        st.markdown("**Nonlinear flux channel kernel**")
        st.code(
            np.array2string(weights[1], precision=8),
            language="text",
        )

        if weights.shape[0] >= 3:
            st.markdown("**Forcing channel kernel**")
            st.code(
                np.array2string(weights[2], precision=8),
                language="text",
            )

        st.markdown("### Scaled physical interpretation")

        if C_diff_b > 1e-12:
            st.markdown(
                "**Scaled diffusion channel**; expected residual shape close to `[1, -2, 1]`"
            )
            st.code(
                np.array2string(weights[0] / C_diff_b, precision=6),
                language="text",
            )

        if C_conv_up_b > 1e-12:
            st.markdown(
                "**Scaled convection/flux channel**; expected upwind-like residual shape close to `[-1, 1, 0]`"
            )
            st.code(
                np.array2string(-weights[1] / C_conv_up_b, precision=6),
                language="text",
            )

        # -------------------------------------------------------------------
        # Training loss
        # -------------------------------------------------------------------
        st.markdown("### FBEQ Burgers CNN training loss")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(conv_hist[:, 0] + 1e-20)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("FBEQ Burgers CNN training loss")
        ax.grid(True)
        fig.tight_layout()
        st.pyplot(fig)

        # -------------------------------------------------------------------
        # CNN rollout comparison
        # -------------------------------------------------------------------
        st.markdown("### CNN rollout comparison")

        stride = max(1, T_min // 800)

        col_ref, col_cnn, col_err = st.columns(3)

        with col_ref:
            st.pyplot(
                make_heatmap(
                    burgers_ref_subset[::stride],
                    x_b,
                    t_subset[::stride],
                    "Reference FBEQ solution",
                )
            )

        with col_cnn:
            st.pyplot(
                make_heatmap(
                    cnn_solution_subset[::stride],
                    x_b,
                    t_subset[::stride],
                    "CNN FBEQ rollout",
                )
            )

        with col_err:
            st.pyplot(
                make_heatmap(
                    abs_error[::stride],
                    x_b,
                    t_subset[::stride],
                    "Absolute error",
                    cmap="viridis",
                )
            )

    else:
        st.info(
            "Click **Train FBEQ Burgers CNN kernel** to train the Burgers model." 
            "For the online demo, the default FBEQ settings are reduced for speed."
    "Use longer final times and more epochs locally for publication-quality runs."
        )