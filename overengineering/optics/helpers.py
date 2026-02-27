from ..general import *
from scipy.constants import h, c
from .params import LaserBeam, Telescope

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm


# paulis
_I2  = np.eye(2, dtype=complex)
_sX  = np.array([[ 0,  1 ], [ 1,  0 ]], dtype=complex)   # σ₁ → D/A basis
_sY  = np.array([[ 0, -1j], [ 1j, 0 ]], dtype=complex)   # σ₂ → R/L basis
_sZ  = np.array([[ 1,  0 ], [ 0, -1 ]], dtype=complex)   # σ₃ → H/V basis
_PAULI = [_I2, _sX, _sY, _sZ]

# projection kets
_KET = {
    'H': np.array([ 1,  0 ], dtype=complex),
    'V': np.array([ 0,  1 ], dtype=complex),
    'D': np.array([ 1,  1 ], dtype=complex) / np.sqrt(2),
    'A': np.array([ 1, -1 ], dtype=complex) / np.sqrt(2),
    'R': np.array([ 1,  1j ], dtype=complex) / np.sqrt(2),
    'L': np.array([ 1, -1j ], dtype=complex) / np.sqrt(2),
}
_LABELS_1Q = ['H', 'V', 'D', 'A', 'R', 'L']

# quED-TOM waveplate settings:
_SETTINGS = {
    'H': ( 0, 0 ),
    'V': ( 0, np.pi/4 ),
    'D': ( 0, np.pi/8 ),
    'A': ( 0, -np.pi/8 ),
    'R': ( np.pi/4, 0 ),
    'L': ( -np.pi/4, 0 ),
}


def _stokes_contrast(n_plus, n_minus) -> float:
    """(N+ - N-) / (N+ + N-).  Returns 0 if both are zero."""
    total = n_plus + n_minus
    return float((n_plus - n_minus) / total) if total > 0 else 0.0

def _analysis_circuit(qwp_angle: float, hwp_angle: float,
                      wavelength: float = 810e-9):
    """
    Build the QWP - HWP - PBS analysis circuit for one measurement setting.
    """
    from .simulators import OpticalCircuit
    circ = OpticalCircuit(wavelength=wavelength, name="Analysis Circuit")
    circ.add_qwp(fast_axis_angle=qwp_angle)
    circ.add_hwp(fast_axis_angle=hwp_angle)
    circ.add_pbs(port='transmitted')
    return circ

def density_matrix_1photon(counts) -> Qobj:
    
    N = np.asarray(counts, dtype=float)
    N_H, N_V, N_D, N_A, N_R, N_L = N

    SZ = _stokes_contrast(N_H, N_V)
    SX = _stokes_contrast(N_D, N_A)
    SY = _stokes_contrast(N_R, N_L)

    rho = 0.5 * (_I2 + SX * _sX + SY * _sY + SZ * _sZ)
    return Qobj(rho)

def density_matrix_2photon(counts_36) -> Qobj:
    
    C = np.asarray(counts_36, dtype=float)
    assert C.shape == (6, 6), "counts_36 must be shape (6, 6)"

    # basis ordering 0=HV, 1=DA, 2=RL maps to Pauli indices 3, 1, 2
    _B2P  = [3, 1, 2]
    _SIGN = np.array([[1, -1], [-1, 1]])

    S = np.zeros((4, 4), dtype=float)
    S[0, 0] = 1.0

    for bi in range(3):
        pi = _B2P[bi]
        for bj in range(3):
            pj = _B2P[bj]
            block = C[2*bi:2*bi+2, 2*bj:2*bj+2]
            total = block.sum()
            S[pi, pj] = (_SIGN * block).sum() / total if total > 0 else 0.0

        row = C[2*bi:2*bi+2, :].sum(axis=1)
        S[pi, 0] = _stokes_contrast(row[0], row[1])

    for bj in range(3):
        pj = _B2P[bj]
        col = C[:, 2*bj:2*bj+2].sum(axis=0)
        S[0, pj] = _stokes_contrast(col[0], col[1])

    rho = sum(S[i, j] * np.kron(_PAULI[i], _PAULI[j])
              for i in range(4) for j in range(4)) / 4.0
    return Qobj(rho)

def rho_properties(rho) -> dict:
    
    arr = rho.full() if isinstance(rho, Qobj) else np.asarray(rho)
    return {
        'trace':  float(np.trace(arr).real),
        'purity': float(np.trace(arr @ arr).real),
    }

def rho_eigensystem(rho) -> Tuple[np.ndarray, List[Qobj]]:
    
    arr = rho.full() if isinstance(rho, Qobj) else np.asarray(rho)
    vals, vecs = np.linalg.eigh(arr)
    order = np.argsort(vals)[::-1]
    vals  = vals[order].real
    vecs  = vecs[:, order]
    return vals, [Qobj(vecs[:, k].reshape(-1, 1)) for k in range(len(vals))]

def compose_abcd(*matrices: np.ndarray) -> np.ndarray:
    """
    Compose multiple ABCD matrices (right to left).
    """
    result = np.eye(2)
    for M in reversed(matrices):
        result = M @ result
    return result

def _fiber_jones(theta: float, delta: float) -> np.ndarray:
    """
    Jones matrix for a birefringent fiber with fast axis at angle theta and retardance delta.
    """
    c, s = np.cos(theta), np.sin(theta)
    R     = np.array([[c, -s], [s, c]])
    W     = np.array([[1, 0], [0, np.exp(-1j * delta)]])
    return R @ W @ R.T

def sweep_focal_lengths(
    laser:        LaserBeam,
    f_collimator: float,
    w_fiber:      float,
    d0:           float,
    d_12:         float,
    top_n:        int = 10,
) -> List[Dict]:
    """
    Available focal lengths
    -----------------------
    50, 75, 100, 150, 200, 300 mm

    Parameters
    ----------
    laser        : LaserBeam with w0 and wavelength set [m]
    f_collimator : collimator focal length [m]
    w_fiber      : target fiber mode-field radius (MFD/2) [m]
    d0           : fixed laser-waist-to-L1 distance [m]
    d_12         : fixed L1-to-L2 distance [m]
    top_n        : number of top results to return and print

    Returns
    -------
    List of result dicts sorted by coupling_eff descending, each containing
    all keys from FiberModeMatchOptimizer.optimize() plus 'f1' and 'f2' [m].
    """
    from .analyzers import FiberModeMatchOptimizer

    available_f = [0.050, 0.075, 0.100, 0.150, 0.200, 0.300]  # meters

    all_results = []
    total = len(available_f) ** 2
    niceprint(f"Sweeping {total} focal-length combinations …", 5)

    for f1 in available_f:
        for f2 in available_f:
            tel = Telescope(f1=f1, f2=f2)
            opt = FiberModeMatchOptimizer(
                laser        = laser,
                telescope    = tel,
                f_collimator = f_collimator,
                w_fiber      = w_fiber,
                d0           = d0,
                d_12         = d_12,
            )
            r = opt.optimize()
            all_results.append({'f1': f1, 'f2': f2, **r})

    all_results.sort(key=lambda x: x["coupling_eff"], reverse=True)
    top = all_results[:top_n]

    niceprint("---")
    niceprint(f"**Focal Length Sweep — Top {top_n} Configurations**", 3)
    niceprint(
        f"{'Rank':<5} {'f1 (mm)':<10} {'f2 (mm)':<10} {'η (%)':<10} "
        f"{'L2→coll (cm)':<15} {'w_fiber (μm)':<14} {'Δw (%)'}",
        5,
    )
    for i, r in enumerate(top, 1):
        dw = abs(r['w_at_fiber'] - r['w_fiber_target']) / r['w_fiber_target'] * 100
        niceprint(
            f"{i:<5} {r['f1'] * 1e3:<10.0f} {r['f2'] * 1e3:<10.0f} "
            f"{r['coupling_eff'] * 100:<10.2f} "
            f"{r['d_L2coll'] * 1e2:<15.2f} "
            f"{r['w_at_fiber'] * 1e6:<14.3f} {dw:.2f}%"
        )

    return top

def sweep_telescope_focal_lengths(
    laser:        LaserBeam,
    f_collimator: float,
    w_fiber:      float,
    d0:           float,
    d_12:         float,
    top_n:        int = 10,
) -> List[Dict]:
    """
    Available focal lengths
    -----------------------
    50, 75, 100, 150, 200, 300 mm

    Parameters
    ----------
    laser        : LaserBeam dataclass [m]
    f_collimator : collimator focal length [m]
    w_fiber      : target fiber mode-field radius (MFD/2) [m]
    d0           : fixed laser-waist-to-L1 distance [m]
    d_12         : fixed L1-to-L2 distance [m]
    top_n        : number of top results to print

    Returns
    -------
    List of result dicts sorted by coupling_eff descending, each containing
    all keys from FiberModeMatchOptimizer.optimize() plus 'f1_mm' and 'f2_mm'.
    """
    from .analyzers import FiberModeMatchOptimizer

    FOCAL_LENGTHS_M = [f * 1e-3 for f in [50, 75, 100, 150, 200, 300]]

    all_results = []
    n_total = len(FOCAL_LENGTHS_M) ** 2

    print(f"Sweeping {n_total} focal-length combinations...")
    for f1 in tqdm(FOCAL_LENGTHS_M, leave=False):
        for f2 in FOCAL_LENGTHS_M:
            tel = Telescope(f1=f1, f2=f2)
            opt = FiberModeMatchOptimizer(
                laser        = laser,
                telescope    = tel,
                f_collimator = f_collimator,
                w_fiber      = w_fiber,
                d0           = d0,
                d_12         = d_12,
            )
            r = opt.optimize()
            all_results.append({'f1_mm': f1 * 1e3, 'f2_mm': f2 * 1e3, **r})

    all_results.sort(key=lambda x: x['coupling_eff'], reverse=True)

    header = (
        f"{'Rank':<5} {'f1 (mm)':<10} {'f2 (mm)':<10} {'η (%)':<9}"
        f" {'d_12 (cm)':<11} {'L2→coll (cm)':<14} {'w_fiber (μm)':<14} {'mismatch':<10}"
    )
    print("\n" + "=" * len(header))
    print(f"  Top {top_n} telescope configurations by coupling efficiency")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for rank, r in enumerate(all_results[:top_n], 1):
        print(
            f"{rank:<5} {r['f1_mm']:<10.0f} {r['f2_mm']:<10.0f}"
            f" {r['coupling_eff'] * 100:<9.2f}"
            f" {r['d_12'] * 1e2:<11.2f}"
            f" {r['d_L2coll'] * 1e2:<14.2f}"
            f" {r['w_at_fiber'] * 1e6:<14.3f}"
            f" {r['mismatch']:<10.5f}"
        )

    print("=" * len(header))
    print(
        f"\nBest: f1 = {all_results[0]['f1_mm']:.0f} mm, "
        f"f2 = {all_results[0]['f2_mm']:.0f} mm"
        f"  →  η = {all_results[0]['coupling_eff'] * 100:.2f}%\n"
    )

    return all_results

def _pbs_transmitted_power(jones_state: np.ndarray) -> float:
    """
    Power transmitted through a horizontal PBS port.
    """
    E_H = jones_state[0]
    E_V = jones_state[1]
    P_tot = np.abs(E_H)**2 + np.abs(E_V)**2
    if P_tot < 1e-30:
        return 0.0
    return float(np.abs(E_H)**2 / P_tot)

def plot_beam_propagation(positions: np.ndarray, widths: np.ndarray,
                          title: str = "Beam Propagation",
                          save_path: Optional[str] = None):
    """
    Plot beam width evolution.
    
    Args:
        positions: z positions (meters)
        widths: Beam widths (meters)
        title: Plot title
        save_path: If provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to mm for easier reading
    ax.plot(positions * 1e2, widths * 1e3, 'b-', linewidth=2)
    ax.fill_between(positions * 1e2, -widths * 1e3, widths * 1e3, 
                     alpha=0.3, color='blue')
    
    ax.set_xlabel('Position (cm)', fontsize=12)
    ax.set_ylabel('Beam Width (mm)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_cavity_stability_map(R: float, reflectivity: float,
                              W_range: Tuple[float, float],
                              H_range: Tuple[float, float],
                              W_design: float, H_design: float,
                              wavelength: float, num_points: int = 250):
    """
    Plot cavity stability and FSR as function of width and height.
    
    Args:
        R: Mirror radius of curvature (m)
        reflectivity: Product of all mirror reflectivities
        W_range: (W_min, W_max) in meters
        H_range: (H_min, H_max) in meters
        W_design: Design width (m)
        H_design: Design height (m)
        wavelength: Wavelength (m)
        num_points: Grid resolution
    """
    from .analyzers import CavityAnalyzer
    analyzer = CavityAnalyzer(wavelength)
    
    W_arr = np.linspace(W_range[0], W_range[1], num_points)
    H_arr = np.linspace(H_range[0], H_range[1], num_points)
    W_grid, H_grid = np.meshgrid(W_arr, H_arr)
    
    FSR_grid = np.zeros_like(W_grid)
    stability_grid = np.zeros_like(W_grid)
    
    # Calculate properties over parameter space
    for i in range(num_points):
        for j in range(num_points):
            W, H = W_grid[i, j], H_grid[i, j]
            d_diag = np.sqrt(W**2 + H**2)
            L1 = d_diag + W
            L2 = d_diag + W
            
            results = analyzer.full_analysis(L1, L2, R, R, reflectivity)
            FSR_grid[i, j] = results['FSR'] if results['stable'] else 0
            stability_grid[i, j] = results['g_parameter']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ----- FSR Plot -----
    FSR_GHz = FSR_grid / 1e9  # Convert to GHz
    FSR_GHz[FSR_GHz == 0] = np.nan  # Don't show unstable regions
    
    levels_fsr = np.linspace(np.nanmin(FSR_GHz), np.nanmax(FSR_GHz), 25)
    contour_fsr = ax1.contourf(W_grid*100, H_grid*100, FSR_GHz,
                              levels=levels_fsr, cmap='plasma')
    contour_lines = ax1.contour(W_grid*100, H_grid*100, FSR_GHz,
                               levels=15, colors='white', linewidths=0.5, alpha=0.4)
    ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    
    # Stability boundary
    stability_boundary = ax1.contour(W_grid*100, H_grid*100, stability_grid,
                                    levels=[1.0], colors='cyan', linewidths=3)
    ax1.clabel(stability_boundary, inline=True, fontsize=10, fmt='Stability Limit')
    unstable_mask = stability_grid > 1
    ax1.contourf(W_grid*100, H_grid*100, unstable_mask.astype(float),
                levels=[0.5, 1.5], colors='black', alpha=0.2)
    
    # Mark design point
    ax1.plot(W_design*100, H_design*100, 'r*', markersize=20,
            markeredgecolor='white', markeredgewidth=2,
            label=f'W = {W_design*100:.1f} cm, H = {H_design*100:.1f} cm')
    
    plt.colorbar(contour_fsr, ax=ax1, label='FSR (GHz)', pad=0.02)
    ax1.set_xlabel('Width W (cm)', fontsize=11)
    ax1.set_ylabel('Height H (cm)', fontsize=11)
    ax1.set_title('FSR - Full Parameter Space', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ----- Stability Plot -----
    levels_stab = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    colors_stab = plt.cm.RdYlGn_r(np.linspace(0, 1, len(levels_stab)-1))
    
    contour_stab = ax2.contourf(W_grid*100, H_grid*100, stability_grid,
                               levels=levels_stab, colors=colors_stab, alpha=0.8)
    contour_lines_stab = ax2.contour(W_grid*100, H_grid*100, stability_grid,
                                     levels=levels_stab, colors='black',
                                     linewidths=1, alpha=0.5)
    ax2.clabel(contour_lines_stab, inline=True, fontsize=9, fmt='%.1f')
    
    # Stability boundary
    boundary = ax2.contour(W_grid*100, H_grid*100, stability_grid,
                          levels=[1.0], colors='red', linewidths=4)
    ax2.clabel(boundary, inline=True, fontsize=12, fmt='Stability Limit')
    ax2.plot(W_design*100, H_design*100, 'r*', markersize=20,
            markeredgecolor='white', markeredgewidth=2,
            label=f'W = {W_design*100:.1f} cm, H = {H_design*100:.1f} cm')
    
    cbar2 = plt.colorbar(contour_stab, ax=ax2,
                        label='Stability Parameter (Stable: 0 < g < 1)', pad=0.02)
    ax2.set_xlabel('Width W (cm)', fontsize=11)
    ax2.set_ylabel('Height H (cm)', fontsize=11)
    ax2.set_title(r'Stability Parameter $\frac{(A+D)^2}{4}$',
                 fontweight='bold', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    return ax1, ax2

def photon_energy(wavelength: float) -> float:
    """Energy [J] of one photon at the given wavelength [m]."""
    return h * c / wavelength

def visibility(R_max: float, R_min: float) -> float:
    return (R_max - R_min) / (R_max + R_min)



