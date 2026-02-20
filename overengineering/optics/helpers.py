from ..general import *
from ..constants import *
from .helpers import *

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class CavityGeometry:
    """Bow-tie cavity geometry parameters (all in cm for compatibility)"""
    W: float  # Width of bow-tie cavity in cm
    H: float  # Height of bow-tie cavity in cm
    fconcave: float  # Focal length of concave mirror in cm
    R_mirrors: List[float]  # Reflectivities [R_input, R_output, R_flat1, R_flat2]
    wavelength: float  # Wavelength in cm


@dataclass
class LaserBeam:
    """Laser beam parameters (in cm for compatibility)"""
    w0: float  # Beam waist in cm
    z0_location: float  # Distance from reference point to waist in cm
    wavelength: float  # Wavelength in cm


@dataclass
class Telescope:
    """Telescope lens parameters (in cm)"""
    f1: float  # Focal length of first lens in cm
    f2: float  # Focal length of second lens in cm


@dataclass
class PiezoActuator:
    """Piezo actuator parameters"""
    displacement_per_volt: float  # nm/V
    voltage_amplitude: float  # V (peak amplitude)
    frequency: float  # Hz
    offset_voltage: float  # V (DC offset)


@dataclass
class Photodetector:
    """Photodetector parameters"""
    responsivity: float  # A/W
    load_resistance: float  # Ohms
    gain: float  # Additional amplifier gain

@dataclass
class SPDC:
    """
    SPDC parameters for the quED entanglement demonstrator

    Parameters
    ----------
    lambda_pump    : pump wavelength [m], default 405 nm (Blu-Ray diode)
    spdc_efficiency: probability per pump photon of producing one pair, default 1e-11
    eta_1, eta_2   : end-to-end detection efficiencies for arm 1 and arm 2
    P_max          : pump power at operating current [W], default 18 mW
    I_threshold    : laser diode threshold current [mA], default 26 mA
    I_operating    : laser diode operating current [mA], default 41 mA (quED-3)
    """
    lambda_pump:     float = 405e-9
    spdc_efficiency: float = 1e-11
    eta_1:           float = 0.20
    eta_2:           float = 0.20
    P_max:           float = 18e-3
    I_threshold:     float = 26.0
    I_operating:     float = 41.0



def compose_abcd(*matrices: np.ndarray) -> np.ndarray:
    """
    Compose multiple ABCD matrices (right to left).
    
    Math:
        M_total = M_n @ M_{n-1} @ ... @ M_2 @ M_1
    
    Example:
        M_total = compose_abcd(M_lens2, M_space, M_lens1)
    """
    result = np.eye(2)
    for M in reversed(matrices):
        result = M @ result
    return result


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
    from .tools import CavityAnalyzer
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
    
    plt.tight_layout()
    plt.show()
    
    return fig, (ax1, ax2)

@staticmethod
def photon_energy(wavelength: float) -> float:
    """Energy [J] of one photon at the given wavelength [m]."""
    return h * c / wavelength

@staticmethod
def visibility(R_max: float, R_min: float) -> float:
    return (R_max - R_min) / (R_max + R_min)

