from ..general import *

import numpy as np
from scipy.signal import sawtooth
from scipy.optimize import minimize, minimize_scalar, differential_evolution

import matplotlib.pyplot as plt

from qutip import *

from typing import Literal, Tuple, Dict, Optional, List
from dataclasses import dataclass

c = 299792458  # speed of light (m/s)

@dataclass
class CavityGeometry:
    """Basic cavity geometry parameters"""
    W: float  # Width of bow-tie cavity in cm
    H: float  # Height of bow-tie cavity in cm
    f_concave: float  # Focal length of concave mirror in cm
    R_mirrors: list  # Reflectivities [R_input, R_output, R_flat1, R_flat2]
    wavelength: float  # Wavelength in cm
    
    def __post_init__(self):
        """Validate inputs"""
        if self.W <= 0 or self.H <= 0:
            raise ValueError("W and H must be positive")
        if self.f_concave <= 0:
            raise ValueError("Focal length must be positive")
        if len(self.R_mirrors) < 2:
            raise ValueError("Need at least 2 mirror reflectivities")

@dataclass
class LaserBeam:
    """Laser beam parameters (measured in lab)"""
    w0: float  # Beam waist in cm
    z0_location: float  # Distance from reference point to waist in cm
    wavelength: float  # Wavelength in cm

@dataclass
class Telescope:
    """Telescope lens parameters"""
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
    """DET36A2 photodetector parameters"""
    responsivity: float  # A/W 
    load_resistance: float  # Ohms
    gain: float  # Additional amplifier gain if any


class GaussianBeam:
    """Gaussian beam propagation using ABCD matrices"""
    
    def __init__(self, wavelength: float):
        self.lam = wavelength
        self.z_R_factor = np.pi / wavelength  # for calculating Rayleigh range
    
    def q_from_waist(self, w0: float, z: float, z0: float = 0) -> complex:
        """Calculate q-parameter from beam waist and position"""
        z_R = self.z_R_factor * w0**2  # Rayleigh range
        return (z - z0) + 1j * z_R
    
    def waist_from_q(self, q: complex) -> float:
        """Extract beam waist from q-parameter"""
        if q.imag <= 0:
            raise ValueError("Invalid q-parameter")
        return np.sqrt(abs(q.imag) * self.lam / np.pi)
    
    def R_from_q(self, q: complex) -> float:
        """Extract radius of curvature from q-parameter"""
        if abs(q.real) < 1e-10:
            return np.inf
        return q.real * (1 + (q.imag / q.real)**2)
    
    def propagate_q(self, q: complex, ABCD: np.ndarray) -> complex:
        """Propagate q through ABCD matrix"""
        A, B, C, D = ABCD[0, 0], ABCD[0, 1], ABCD[1, 0], ABCD[1, 1]
        return (A * q + B) / (C * q + D)
    
    @staticmethod
    def free_space(d: float) -> np.ndarray:
        """ABCD matrix for free space"""
        return np.array([[1, d], [0, 1]])
    
    @staticmethod
    def thin_lens(f: float) -> np.ndarray:
        """ABCD matrix for thin lens"""
        return np.array([[1, 0], [-1/f, 1]])
    
    @staticmethod
    def curved_mirror(R: float) -> np.ndarray:
        """ABCD matrix for curved mirror (R is radius of curvature)"""
        # For a curved mirror: f = R/2, so we use -2/R for the ray matrix
        return np.array([[1, 0], [-2/R, 1]])
    
    @staticmethod
    def flat_mirror() -> np.ndarray:
        """ABCD matrix for flat mirror"""
        return np.array([[1, 0], [0, 1]])
    
    @staticmethod
    def compose(*matrices) -> np.ndarray:
        """Compose multiple ABCD matrices (right to left)"""
        result = np.eye(2)
        for M in reversed(matrices):
            result = M @ result
        return result
    
    def get_beam_width(self, q: complex) -> float:
        """
        Calculate beam width w(z) from q-parameter.
        
        Math: w(z) = w₀√(1 + (z/z_R)²)
        
        where:
            w₀ = beam waist
            z = Re(q) = distance from waist  
            z_R = Im(q) = Rayleigh range
        
        Parameters:
        -----------
        q : complex
            Complex beam parameter
        
        Returns:
        --------
        float
            Beam width at position z (same units as wavelength)
        """
        w0 = self.waist_from_q(q)
        z = q.real
        z_R = q.imag
        
        if abs(z_R) < 1e-12:
            return w0
        
        return w0 * np.sqrt(1 + (z / z_R)**2)
    
    def get_divergence_angle(self, w0: float) -> float:
        """
        Calculate far-field divergence angle θ.
        
        Math: θ = λ/(π·w₀) [radians]
        
        Parameters:
        -----------
        w0 : float
            Beam waist size (same units as wavelength)
        
        Returns:
        --------
        float
            Half-angle divergence in radians
        """
        return self.lam / (np.pi * w0)
    
    def get_rayleigh_range(self, w0: float) -> float:
        """
        Calculate Rayleigh range z_R.
        
        Math: z_R = π·w₀²/λ
        
        Parameters:
        -----------
        w0 : float
            Beam waist size (same units as wavelength)
        
        Returns:
        --------
        float
            Rayleigh range (same units as wavelength)
        """
        return self.z_R_factor * w0**2

class OpticalCavity:
    """
    Optical cavity (resonator) with two mirrors.
    
    ELI5: A cavity is like a room for light where it bounces back and forth
    between two mirrors. If the round-trip distance matches the wavelength
    perfectly, the light builds up (resonates). Otherwise, it cancels out.
    
    Math: For a cavity with mirrors M1, M2 separated by lengths L1, L2,
    the round-trip ABCD matrix is:
      M_rt = M_M2 · M_L2 · M_M1 · M_L1
    
    Stability condition: |A + D| < 2 where A, D are elements of M_rt
    
    The cavity is stable when: 0 < (1 - L1/R1)(1 - L2/R2) < 1
    
    Source: Optical Cavities lecture notes, stability analysis
    """
    
    def __init__(self, L1: float, L2: float, R1: float, R2: float,
                 reflectivity: float, wavelength: float):
        """
        Args:
            L1, L2: distances between mirrors (m)
            R1, R2: radii of curvature (m), use np.inf for flat mirror
            reflectivity: power reflectivity of mirrors (0 to 1)
            wavelength: wavelength (m)
        """
        self.L1 = L1
        self.L2 = L2
        self.R1 = R1
        self.R2 = R2
        self.rho = reflectivity
        self.wavelength = wavelength
        self.L_rt = L1 + L2
        
        # Calculate ABCD matrix
        self.M_rt = self._calculate_round_trip_matrix()
        self.A = self.M_rt[0, 0]
        self.B = self.M_rt[0, 1]
        self.C = self.M_rt[1, 0]
        self.D = self.M_rt[1, 1]
        
        # Calculate cavity properties
        self.stability_param = (self.A + self.D)**2 / 4
        self.is_stable = self.stability_param <= 1
        
        if self.is_stable:
            self._calculate_mode_parameters()
            self._calculate_resonance_properties()
        else:
            print(f"Unstable configuration: stability parameter = {self.stability_param:.3f} (must be <= 1)")
    
    def _calculate_round_trip_matrix(self) -> np.ndarray:
        """
        Calculate round-trip ABCD matrix.
        
        Math: Starting from M1, going around and back:
          M_rt = M_M1 · M_L1 · M_M2 · M_L2
        
        where:
          M_Mj = [[1, 0], [-2/Rj, 1]]  (curved mirror)
          M_Lj = [[1, Lj], [0, 1]]     (free space)
        """
        # Free space propagation
        M_L1 = np.array([[1, self.L1], [0, 1]])
        M_L2 = np.array([[1, self.L2], [0, 1]])
        
        # Mirrors (use -2/R for curved, identity for flat)
        M_M1 = np.array([[1, 0], [-2/self.R1, 1]]) if np.isfinite(self.R1) else np.eye(2)
        M_M2 = np.array([[1, 0], [-2/self.R2, 1]]) if np.isfinite(self.R2) else np.eye(2)
        
        # Round trip: M1 → L1 → M2 → L2 → back to M1
        M_rt = M_L2 @ M_M2 @ M_L1
        
        return M_rt
    
    def _calculate_mode_parameters(self):
        """
        Calculate cavity mode (beam waist location and size).
        
        Math: The eigenmode satisfies q_out = q_in under round-trip propagation.
        This gives: q = (A - D - i√(4 - (A+D)²)) / (2C)
        
        From q = z + i·z_R, we extract:
          z = Re(q) = distance from waist to reference plane
          z_R = Im(q) = Rayleigh range
          w0 = √(λ·z_R/π) = beam waist radius
        
        Source: Optical Cavities lecture notes, eigenmode analysis
        """
        discriminant = 4 - (self.A + self.D)**2
        
        if discriminant < 0:
            raise ValueError("Cavity is unstable (discriminant < 0)")
        
        # Calculate q-parameter of cavity eigenmode
        self.q_res = ((self.A - self.D) - 1j * np.sqrt(discriminant)) / (2 * self.C)
        
        self.z0 = np.real(self.q_res)  # Waist location from M1
        self.b = np.imag(self.q_res)    # Confocal parameter b = z_R
        self.w0 = np.sqrt(self.wavelength * self.b / np.pi)  # Beam waist
    
    def _calculate_resonance_properties(self):
        """
        Calculate cavity resonance properties.
        
        Math:
          FSR = c / L_rt (Free Spectral Range)
          Finesse = π√ρ / (1 - ρ) ≈ π / (1 - ρ) for ρ ≈ 1
          δf = FSR / Finesse (Linewidth)
          N_bounce = -1 / (2·ln(ρ)) (number of bounces before decay)
        
        Source: Optical Cavities lecture notes, Finesse and FSR
        """
        self.FSR = c / self.L_rt
        self.finesse = np.pi * np.sqrt(self.rho) / (1 - self.rho)
        self.n_bounce = -1 / (2 * np.log(self.rho))
        self.delta_f = self.FSR / self.finesse
        
        # Quality factor Q = ν₀ / δf where ν₀ = c/λ
        self.Q = (c / self.wavelength) / self.delta_f
    
    def get_mode_at_position(self, z: float) -> GaussianBeam:
        """
        Get cavity mode parameters at distance z from M1.
        
        Args:
            z: distance from M1 (m)
        
        Returns:
            GaussianBeam object at that position
        """
        if not self.is_stable:
            raise ValueError("Cannot get mode for unstable cavity")
        
        # Create beam at waist, then propagate to position z
        beam = GaussianBeam(self.w0, self.wavelength, z=0)
        beam.z = z - self.z0  # Shift reference to waist
        beam.q = beam.z + 1j * beam.zR
        
        return beam
    
    def __repr__(self):
        if not self.is_stable:
            return f"OpticalCavity(UNSTABLE, L_rt={self.L_rt*100:.2f}cm)"
        
        return (f"OpticalCavity(L_rt={self.L_rt*100:.2f}cm, "
                f"w0={self.w0*1e6:.1f}um, "
                f"FSR={self.FSR/1e9:.2f}GHz, "
                f"F={self.finesse:.1f})")



class BowTieCavity:
    """Analyzes bow-tie cavity and calculates cavity mode"""
    
    def __init__(self, geom: CavityGeometry):
        self.geom = geom
        self.beam = GaussianBeam(geom.wavelength)
        
        # Calculated properties (filled by analyze_cavity)
        self.is_stable = False
        self.stability_param = None
        self.roundtrip_matrix = None
        self.q_at_input = None
        self.cavity_waist = None
        self.waist_location = None
        self.cavity_length = None
        self.FSR = None
        self.finesse = None
        self.linewidth = None
        
    def build_roundtrip_matrix(self) -> np.ndarray:
        """
        Build round-trip ABCD matrix for bow-tie cavity.
        
        Cavity configuration (looking down from above):
        
                M2 (flat)
                /      \
               /        \
        M1 (input)    M3 (concave)
               \        /
                \      /
                 M4 (flat)
        
        Starting from M1 (input mirror), going counterclockwise.
        """
        W = self.geom.W
        H = self.geom.H
        f = self.geom.f_concave
        R_concave = 2 * f  # Radius of curvature
        
        # Calculate segment lengths for bow-tie
        # The bow-tie has two diagonal segments and two horizontal segments
        # Diagonal length (assuming symmetric bow-tie)
        d_diagonal = np.sqrt(W**2 + (H/2)**2)
        d_horizontal = W
        
        # Round-trip starting from input mirror (M1)
        # Path: M1 -> M2 -> M3 (concave) -> M4 -> back to M1
        
        M = self.beam.compose(
            # Segment 1: M1 to M2 (diagonal)
            self.beam.free_space(d_diagonal),
            self.beam.flat_mirror(),
            
            # Segment 2: M2 to M3 (horizontal)
            self.beam.free_space(d_horizontal),
            self.beam.curved_mirror(R_concave),
            
            # Segment 3: M3 to M4 (diagonal)
            self.beam.free_space(d_diagonal),
            self.beam.flat_mirror(),
            
            # Segment 4: M4 back to M1 (horizontal)
            self.beam.free_space(d_horizontal),
            self.beam.flat_mirror()  # M1 (input mirror, assumed flat)
        )
        
        self.roundtrip_matrix = M
        self.cavity_length = 2 * (d_diagonal + d_horizontal)
        
        return M
    
    def check_stability(self, M: np.ndarray) -> Tuple[bool, float]:
        """Check if cavity is stable"""
        A, D = M[0, 0], M[1, 1]
        stability = abs((A + D) / 2)
        is_stable = stability < 1.0
        return is_stable, stability
    
    def find_eigenmode(self, M: np.ndarray) -> Optional[complex]:
        """
        Find the eigenmode q-parameter of the cavity.
        
        Eigenmode satisfies: q = (A*q + B) / (C*q + D)
        Rearranging: C*q^2 + (D-A)*q - B = 0
        """
        A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        
        if abs(C) < 1e-12:
            # Special case: nearly confocal or degenerate
            if abs(D - A) < 1e-12:
                # Highly degenerate - try alternative formulation
                # For stable cavity, use B / (1 - (A+D)/2) as estimate
                if abs(1 - (A + D)/2) > 1e-10:
                    z_R = np.sqrt(abs(B * (1 - (A + D)/2)))
                    return 1j * z_R
                return None
            q = -B / (D - A)
            # Make sure imaginary part is positive
            if q.imag <= 0:
                q = -B / (D - A) * 1j
        else:
            # Solve quadratic equation: C*q^2 + (D-A)*q - B = 0
            # Use: q = [-(D-A) ± sqrt((D-A)^2 + 4BC)] / (2C)
            discriminant = (D - A)**2 + 4*B*C
            
            # For a stable cavity, discriminant should be negative (giving complex q)
            # If positive, we have an issue
            if discriminant >= 0:
                # Try to construct a physical solution anyway
                sqrt_disc = np.sqrt(abs(discriminant))
                if discriminant < 0:
                    sqrt_disc = 1j * sqrt_disc
                
                q1 = (-(D - A) + sqrt_disc) / (2*C)
                q2 = (-(D - A) - sqrt_disc) / (2*C)
                
                # Choose solution with positive imaginary part
                if abs(q1.imag) > abs(q2.imag):
                    q = q1 if q1.imag > 0 else -np.conj(q1)
                else:
                    q = q2 if q2.imag > 0 else -np.conj(q2)
            else:
                # discriminant < 0 (normal case for stable cavity)
                sqrt_disc = np.sqrt(discriminant + 0j)  # Force complex
                q1 = (-(D - A) + sqrt_disc) / (2*C)
                q2 = (-(D - A) - sqrt_disc) / (2*C)
                
                # Choose solution with positive imaginary part (physical beam)
                if q1.imag > 0:
                    q = q1
                elif q2.imag > 0:
                    q = q2
                else:
                    # Force positive imaginary part
                    q = q1 if abs(q1.imag) > abs(q2.imag) else q2
                    if q.imag < 0:
                        q = -np.conj(q)
        
        # Ensure imaginary part is positive
        if q.imag <= 0:
            q = q.real + 1j * abs(q.imag)
            if q.imag == 0:
                # Last resort: create small positive imaginary part
                q = q.real + 1j * 1e-6
        
        # Verify this is actually an eigenmode
        q_check = self.beam.propagate_q(q, M)
        error = abs(q_check - q) / abs(q)
        if error > 0.01:  # 1% relative error
            warnings.warn(f"Eigenmode verification: relative error = {error:.6f}")
        
        return q
    
    def find_waist_from_q(self, q_at_ref: complex) -> Tuple[float, float]:
        """
        Find location and size of beam waist from q-parameter.
        
        Returns:
            z_to_waist: distance from reference point to waist (cm)
            w0: beam waist size (cm)
        """
        # Waist occurs where Re(q) = 0
        z_to_waist = -q_at_ref.real
        
        # Propagate to waist
        q_at_waist = self.beam.propagate_q(q_at_ref, self.beam.free_space(z_to_waist))
        
        # Extract waist size
        w0 = self.beam.waist_from_q(q_at_waist)
        
        return z_to_waist, w0
    
    def calculate_cavity_properties(self):
        """Calculate FSR, finesse, linewidth, etc."""
        # Free Spectral Range
        c = 3e10  # cm/s
        self.FSR = c / (2 * self.cavity_length)  # Hz
        
        # Finesse from mirror reflectivities
        R = self.geom.R_mirrors
        if len(R) >= 2:
            # Product of all mirror reflectivities
            R_total = np.prod(R)
            
            # For a cavity with losses (transmission + other losses)
            # Finesse ≈ π√(R_total) / (1 - R_total)
            if R_total > 0 and R_total < 1:
                self.finesse = np.pi * np.sqrt(R_total) / (1 - R_total)
            else:
                self.finesse = None
        else:
            self.finesse = None
        
        # Linewidth (FWHM)
        if self.finesse is not None:
            self.linewidth = self.FSR / self.finesse
        else:
            self.linewidth = None
    
    def analyze_cavity(self) -> Dict:
        """
        Complete cavity analysis.
        
        Returns dictionary with all cavity properties.
        """
        niceprint('---')
        niceprint(f"**Cavity Analysis**", 3)
        
        # Build round-trip matrix
        M = self.build_roundtrip_matrix()
        
        # Check stability
        self.is_stable, self.stability_param = self.check_stability(M)
        if not self.is_stable:
            return {
                'stable': False,
                'stability_param': self.stability_param
            }
        
        # Find eigenmode
        q_eigenmode = self.find_eigenmode(M)
        if q_eigenmode is None:
            print("Could not find cavity eigenmode.")
            return {
                'stable': False,
                'error': 'No eigenmode found'
            }
        
        self.q_at_input = q_eigenmode
        
        # Find waist location and size
        z_to_waist, w0 = self.find_waist_from_q(q_eigenmode)
        self.waist_location = z_to_waist
        self.cavity_waist = w0
        
        # Calculate other properties
        self.calculate_cavity_properties()
        
        # Get beam parameters at input
        w_at_input = self.beam.waist_from_q(q_eigenmode)
        R_at_input = self.beam.R_from_q(q_eigenmode)
        
        
        niceprint(f"<u> Cavity Geometry </u>",5)
        niceprint(f"width: {self.geom.W:.2f} cm, height: {self.geom.H:.2f} cm <br>" +
                  f"concave mirror focal length: {self.geom.f_concave:.2f} cm <br>" +
                  f"round-trip length: {self.cavity_length:.2f} cm"
                  )
        
        niceprint(f"<u> Cavity Stability </u>",5)
        niceprint("Goal: |g| < 1 for stable cavity <br>" +
                  f"Stability parameter |g|: {self.stability_param:.4f} <br>" +
                  f"Status: {'STABLE' if self.is_stable else 'UNSTABLE'}"
                  )
        
        niceprint(fr"<u> Cavity Mode ($\text{{TEM}}_{{00}}$) </u>", 5)
        niceprint(fr"Waist size $w_0$: {self.cavity_waist*1e4:.1f} $\mu m$ <br>" +
                  fr"Waist location from input mirror M1: {self.waist_location:.2f} cm <br>" +
                  fr"Beam waist at input mirror: {w_at_input*1e4:.1f} $\mu m$ <br>" +
                  fr"Radius of curvature at input mirror M1: {R_at_input:.1f} cm"
                  )
        
        niceprint(f"<u> Cavity Properties </u>", 5)
        niceprint(f"Free Spectral Range (FSR): {self.FSR/1e9:.2f} GHz <br>" +
                  f"Finesse: {self.finesse:7.1f} <br>" if self.finesse is not None else "N/A <br>" +
                  f"Linewidth (FWHM): {self.linewidth/1e6:7.2f} MHz" if self.linewidth is not None else "Linewidth (FWHM): N/A"
                  )
        
        return {
            'stable': True,
            'stability_param': self.stability_param,
            'cavity_waist': self.cavity_waist,
            'waist_location': self.waist_location,
            'q_at_input': self.q_at_input,
            'cavity_length': self.cavity_length,
            'FSR': self.FSR,
            'finesse': self.finesse,
            'linewidth': self.linewidth
        }


class ModeMatchOptimizer:
    """Optimize telescope to match laser into cavity mode"""
    
    def __init__(self, cavity: BowTieCavity, laser: LaserBeam, telescope: Telescope, 
                 d_laser_to_L1: float):
        """
        Parameters:
        -----------
        cavity : BowTieCavity
            Analyzed cavity object
        laser : LaserBeam  
            Measured laser beam parameters
        telescope : Telescope
            Telescope lens focal lengths
        d_laser_to_L1 : float
            Distance from laser waist to first lens (cm)
        """
        if not cavity.is_stable:
            raise ValueError("Cavity is unstable.")
        
        self.cavity = cavity
        self.laser = laser
        self.telescope = telescope
        self.d_laser_to_L1 = d_laser_to_L1
        self.beam = GaussianBeam(laser.wavelength)
    
    def propagate_laser_to_cavity(self, d_L1_L2: float, d_L2_cavity: float) -> complex:
        """Propagate laser through telescope to cavity input"""
        
        # Start at laser waist
        q = self.beam.q_from_waist(self.laser.w0, 0, self.laser.z0_location)
        
        # Build total ABCD matrix from laser to cavity
        M_total = self.beam.compose(
            self.beam.free_space(self.d_laser_to_L1),
            self.beam.thin_lens(self.telescope.f1),
            self.beam.free_space(d_L1_L2),
            self.beam.thin_lens(self.telescope.f2),
            self.beam.free_space(d_L2_cavity)
        )
        
        # Propagate
        return self.beam.propagate_q(q, M_total)
    
    def match_quality(self, q_laser: complex, q_cavity: complex) -> float:
        """Calculate mismatch metric"""
        # Normalize by cavity q magnitude
        q_cav_mag = abs(q_cavity)
        if q_cav_mag < 1e-10:
            return 1e10
        mismatch = abs(q_laser - q_cavity) / q_cav_mag
        return mismatch
    
    def coupling_efficiency(self, q_laser: complex, q_cavity: complex) -> float:
        """Calculate mode overlap/coupling efficiency"""
        w_laser = self.beam.waist_from_q(q_laser)
        w_cavity = self.beam.waist_from_q(q_cavity)
        
        # Gaussian mode overlap
        ratio = w_laser / w_cavity
        eta = 4 / (ratio + 1/ratio)**2
        
        return eta
    
    def objective(self, params: np.ndarray) -> float:
        """Objective function for optimization"""
        d_L1_L2, d_L2_cavity = params
        
        # Physical constraints
        if d_L1_L2 < 0.5 or d_L1_L2 > 150:
            return 1e6
        if d_L2_cavity < 1 or d_L2_cavity > 150:
            return 1e6
        
        # Propagate laser to cavity input
        try:
            q_laser = self.propagate_laser_to_cavity(d_L1_L2, d_L2_cavity)
        except:
            return 1e6
        
        # Calculate mismatch
        return self.match_quality(q_laser, self.cavity.q_at_input)
    
    def optimize(self, method='global') -> Dict:
        """Run optimization"""
        
        # Initial guess
        x0 = [self.telescope.f1 + self.telescope.f2, 30.0]
        
        if method == 'global':
            bounds = [(0.5, 150), (1, 150)]
            result = differential_evolution(
                self.objective,
                bounds,
                maxiter=300,
                popsize=15,
                tol=1e-10,
                seed=42,
                disp=False,
                workers=1
            )
        else:
            result = minimize(
                self.objective,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-10}
            )
        
        # Extract results
        d_L1_L2_opt, d_L2_cavity_opt = result.x
        
        # Calculate final parameters
        q_laser_opt = self.propagate_laser_to_cavity(d_L1_L2_opt, d_L2_cavity_opt)
        
        w_laser = self.beam.waist_from_q(q_laser_opt)
        R_laser = self.beam.R_from_q(q_laser_opt)
        w_cavity = self.beam.waist_from_q(self.cavity.q_at_input)
        R_cavity = self.beam.R_from_q(self.cavity.q_at_input)
        
        efficiency = self.coupling_efficiency(q_laser_opt, self.cavity.q_at_input)
        
        # Propagate to cavity waist
        q_laser_at_waist = self.beam.propagate_q(
            q_laser_opt,
            self.beam.free_space(self.cavity.waist_location)
        )
        w_laser_at_waist = self.beam.waist_from_q(q_laser_at_waist)
        
        results = {
            'd_L1_L2': d_L1_L2_opt,
            'd_L2_cavity': d_L2_cavity_opt,
            'total_distance': self.d_laser_to_L1 + d_L1_L2_opt + d_L2_cavity_opt,
            'mismatch': result.fun,
            'coupling_efficiency': efficiency,
            'w_laser_at_input': w_laser,
            'R_laser_at_input': R_laser,
            'w_cavity_at_input': w_cavity,
            'R_cavity_at_input': R_cavity,
            'w_laser_at_cavity_waist': w_laser_at_waist,
            'w_cavity_waist': self.cavity.cavity_waist,
            'success': result.success
        }
        
        return results
    
    def print_results(self, results: Dict):
        niceprint('---')
        niceprint(f"**Telescope Optimization**", 3)
        
        niceprint(f"<u> Telescope Optimized Distances </u>", 5)
        niceprint(fr"**Given** laser waist to Lens 1: {self.d_laser_to_L1:8.2f} cm <br>" +
                  fr"$\rightarrow$ Lens 1 to Lens 2: {results['d_L1_L2']:8.2f} cm <br>" +
                  fr"$\rightarrow$ Lens 2 to Cavity input: {results['d_L2_cavity']:8.2f} cm <br>" +
                  "───────────────────────── <br>" +
                  fr"Total path length: {results['total_distance']:8.2f} cm (from source to cavity input)"
                  )
        
        if results['coupling_efficiency'] < 0.85:
            CE_warning = 'Less than 85% coupling efficiency. Consider adjusting lens distances or focal lengths.'
        elif results['coupling_efficiency'] > 0.95:
            CE_warning = 'Coupling efficiency is great, but be aware of sensitivity to misalignment.'
        else:
            CE_warning = 'Coupling efficiency is good.'
        
        niceprint(f"<u> Optimization Performance </u>", 5)
        niceprint(f"Coupling efficiency: {results['coupling_efficiency']*100:7.2f} % <br>" +
                  fr"$\quad$ {CE_warning} <br>" +
                  f"Mismatch (normalized): {results['mismatch']:7.4f} <br>" +
                  f"Optimization was {'' if results['success'] else 'not'} successful."
                  )
        
        niceprint(f"<u> Beam Parameters at Cavity Input (M1) </u>", 5)
        niceprint("**Laser**<br>" +
                  fr"$\quad$ beam waist: {results['w_laser_at_input']*1e4:7.1f} $\mu m$ <br>" +
                  fr"$\quad$ radius of curvature: {results['R_laser_at_input']:7.1f} cm <br>" +
                  "**Cavity mode**<br>" +
                  fr"$\quad$ beam waist: {results['w_cavity_at_input']*1e4:7.1f} $\mu m$ <br>" +
                  fr"$\quad$ radius of curvature: {results['R_cavity_at_input']:7.1f} cm <br>"
                  )
        
        waist_diff = abs(results['w_laser_at_cavity_waist'] - results['w_cavity_waist'])/results['w_cavity_waist']*100
        niceprint(f"<u> Beam Parameters at Cavity Waist </u>", 5)
        niceprint(fr"Laser beam waist at cavity waist: {results['w_laser_at_cavity_waist']*1e4:7.1f} $\mu m$ <br>" +
                  fr"Cavity mode waist (target): {results['w_cavity_waist']*1e4:7.1f} $\mu m$ <br>" +
                  f"Match quality: {waist_diff:6.2f} % difference"
                  )


class CavityTransmissionSimulator:
    """
    Simulates cavity transmission signal as piezo scans cavity length.
    
    Models the voltage output from a photodetector as the cavity is scanned
    through resonances using a piezo on one mirror.
    """
    
    def __init__(self, cavity: BowTieCavity, 
                 input_power: float,
                 piezo: PiezoActuator = None,
                 detector: Photodetector = None,
                 mode_match_efficiency: float = 1.0):
        """
        Parameters:
        -----------
        cavity : BowTieCavity
            Analyzed cavity object
        input_power : float
            Input laser power in watts (e.g., 1e-3 for 1 mW)
        piezo : PiezoActuator
            Piezo parameters (uses defaults if None)
        detector : Photodetector
            Photodetector parameters (uses defaults if None)
        mode_match_efficiency : float
            Fraction of light coupled into TEM00 mode (0-1)
        """
        self.cavity = cavity
        self.input_power = input_power
        self.piezo = piezo if piezo else PiezoActuator()
        self.detector = detector if detector else Photodetector()
        self.mode_match_efficiency = mode_match_efficiency
        
        # Calculate cavity parameters
        self._calculate_transmission_parameters()
    
    def _calculate_transmission_parameters(self):
        """Calculate cavity transmission parameters"""
        # Get mirror reflectivities
        R = self.cavity.geom.R_mirrors
        
        # Input and output mirror reflectivities
        R_in = R[0]  # Input coupling mirror
        R_out = R[1]  # Output coupling mirror
        
        # Product of all other mirror reflectivities (round-trip loss)
        R_other = np.prod([R[i] for i in range(2, len(R))]) if len(R) > 2 else 1.0
        
        # Total round-trip intensity reflectivity
        self.R_rt = R_in * R_out * R_other
        
        # Transmission coefficients (intensity)
        self.T_in = 1 - R_in  # Input coupling
        self.T_out = 1 - R_out  # Output coupling
        
        # Finesse coefficient for Airy function
        # F = 4R/(1-R)^2
        if self.R_rt > 0 and self.R_rt < 1:
            self.F_coeff = 4 * self.R_rt / (1 - self.R_rt)**2
        else:
            self.F_coeff = 0
        
        # Maximum transmission (on resonance)
        # T_max = T_in * T_out / (1 - sqrt(R_in * R_out * R_other))^2
        sqrt_R = np.sqrt(self.R_rt)
        if sqrt_R < 1:
            self.T_max = self.T_in * self.T_out / (1 - sqrt_R)**2
        else:
            self.T_max = 0
        
        # Linewidth (FWHM) in Hz
        if self.cavity.finesse and self.cavity.FSR:
            self.linewidth_hz = self.cavity.FSR / self.cavity.finesse
        else:
            self.linewidth_hz = None
    
    def airy_transmission(self, phase_shift: np.ndarray) -> np.ndarray:
        """
        Calculate Airy transmission function.
        
        T(δ) = T_max / (1 + F * sin²(δ/2))
        
        Parameters:
        -----------
        phase_shift : np.ndarray
            Round-trip phase shift in radians
        
        Returns:
        --------
        transmission : np.ndarray
            Intensity transmission coefficient (0-1)
        """
        return self.T_max / (1 + self.F_coeff * np.sin(phase_shift / 2)**2)
    
    def cavity_length_to_phase(self, delta_L: float) -> float:
        """
        Convert cavity length change to round-trip phase shift.
        
        For a change in cavity length ΔL:
        Δδ = 4π ΔL / λ  (factor of 4π for round-trip)
        
        Parameters:
        -----------
        delta_L : float
            Change in cavity length in cm
        
        Returns:
        --------
        phase_shift : float
            Round-trip phase shift in radians
        """
        wavelength = self.cavity.geom.wavelength
        return 4 * np.pi * delta_L / wavelength
    
    def piezo_displacement(self, time: np.ndarray) -> np.ndarray:
        """
        Calculate piezo displacement as function of time.
        
        Triangle wave: displacement = A * sawtooth(2πft, width=0.5)
        where width=0.5 gives symmetric triangle
        
        Parameters:
        -----------
        time : np.ndarray
            Time array in seconds
        
        Returns:
        --------
        displacement : np.ndarray
            Piezo displacement in cm
        """
        # Triangle wave voltage
        voltage = self.piezo.voltage_amplitude * sawtooth(
            2 * np.pi * self.piezo.frequency * time, 
            width=0.5
        ) + self.piezo.offset_voltage
        
        # Convert to displacement (nm/V * V = nm, then convert to cm)
        displacement_nm = self.piezo.displacement_per_volt * voltage
        displacement_cm = displacement_nm * 1e-7  # nm to cm
        
        return displacement_cm
    
    def simulate_transmission(self, duration: float = 0.01, 
                            num_points: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate cavity transmission as function of time.
        
        Parameters:
        -----------
        duration : float
            Simulation duration in seconds (default: 10 ms = 2 cycles at 200 Hz)
        num_points : int
            Number of time points
        
        Returns:
        --------
        time : np.ndarray
            Time array in seconds
        voltage_output : np.ndarray
            Output voltage in volts
        transmitted_power : np.ndarray
            Transmitted optical power in watts
        displacement : np.ndarray
            Piezo displacement in nm
        """
        # Time array
        time = np.linspace(0, duration, num_points)
        
        # Piezo displacement
        displacement = self.piezo_displacement(time)
        
        # Convert displacement to phase shift
        phase_shift = self.cavity_length_to_phase(displacement)
        
        # Calculate transmission
        transmission = self.airy_transmission(phase_shift)
        
        # Transmitted power accounting for mode matching
        transmitted_power = (self.input_power * self.mode_match_efficiency * 
                           transmission)
        
        # Photodetector current (A = W * A/W)
        photocurrent = transmitted_power * self.detector.responsivity
        
        # Voltage across load resistor (V = I * R)
        voltage_output = (photocurrent * self.detector.load_resistance * 
                         self.detector.gain)
        
        # Convert displacement back to nm for output
        displacement_nm = displacement / 1e-7
        
        return time, voltage_output, transmitted_power, displacement_nm
    
    def plot_transmission(self, duration: float = 0.01, 
                         num_points: int = 10000,
                         save_path: str = None):
        """
        Plot simulated transmission signal.
        
        Parameters:
        -----------
        duration : float
            Simulation duration in seconds
        num_points : int
            Number of points
        save_path : str
            Path to save figure (optional)
        """
        time, voltage, power, displacement = self.simulate_transmission(
            duration, num_points
        )
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        # Plot 1: Piezo displacement
        axes[0].plot(time * 1000, displacement, 'b-', linewidth=1.5)
        axes[0].set_ylabel('Piezo Displacement (nm)', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Cavity Transmission Simulation', fontsize=12, fontweight='bold')
        
        # Plot 2: Detector voltage
        axes[1].plot(time * 1000, voltage * 1000, 'g-', linewidth=1.5)
        axes[1].set_ylabel('Detector Voltage (mV)', fontsize=11)
        axes[1].set_xlabel('Time (ms)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    def print_summary(self):
        niceprint('---')
        niceprint(f"**Cavity Transmission Simulation**", 3)
        
        niceprint("<u> Input Parameters </u>", 5)
        niceprint(fr"Input laser power: {self.input_power*1e3:7.2f} mW <br>" +
                  fr"Mode-matching efficiency: {self.mode_match_efficiency*100:7.2f} % <br>" +
                  fr"Coupled power into cavity: {self.input_power*self.mode_match_efficiency*1e3:7.2f} mW <br>"
                  )
        
        niceprint("<u> Piezo Parameters </u>", 5)
        niceprint(f"Displacement per volt: {self.piezo.displacement_per_volt:7.1f} nm/V <br>" +
                  f"Voltage amplitude: {self.piezo.voltage_amplitude:7.2f} V <br>" +
                  f"Peak to Peak: {2*self.piezo.voltage_amplitude*self.piezo.displacement_per_volt:7.1f} nm <br>" +
                  f"Scan frequency: {self.piezo.frequency:7.1f} Hz <br>" +
                  f"Scan period: {1000/self.piezo.frequency:7.2f} ms"
                  )
        
        niceprint("<u> Cavity Transmission </u>", 5)
        niceprint(f"Input coupling (T_in): {self.T_in*100:7.2f} % <br>" +
                  f"Output coupling (T_out): {self.T_out*100:7.2f} % <br>" +
                  f"Round-trip reflectivity: {self.R_rt*100:7.2f} % <br>" +
                  f"Peak transmission (T_max): {self.T_max*100:7.3f} % <br>" +
                  f"Finesse: {self.cavity.finesse:7.1f} <br>" +
                  (f"Linewidth (FWHM): {self.linewidth_hz/1e6:7.2f} MHz <br>" if self.linewidth_hz else "") +
                  f"Free Spectral Range: {self.cavity.FSR/1e9:7.2f} GHz"
                  )
        
        # Calculate expected output
        max_transmitted_power = self.input_power * self.mode_match_efficiency * self.T_max
        max_photocurrent = max_transmitted_power * self.detector.responsivity
        max_voltage = max_photocurrent * self.detector.load_resistance * self.detector.gain
        
        niceprint("<u> Photodetector Output </u>", 5)
        niceprint(f"Responsivity at 632.8 nm:       {self.detector.responsivity:7.3f} A/W <br>" +
                  fr"Load resistance:                {self.detector.load_resistance/1000:7.1f} $k\Omega$ <br>" +
                  f"Amplifier gain:                 {self.detector.gain:7.1f}x <br>" +
                  fr"Peak transmitted power:         {max_transmitted_power*1e6:7.2f} $\mu W$ <br>" +
                  fr"Peak photocurrent:              {max_photocurrent*1e6:7.2f} $\mu A$ <br>" +
                  f"Peak voltage output:            {max_voltage*1000:7.2f} mV"
                  )
        
        # FSR in terms of displacement
        wavelength_nm = self.cavity.geom.wavelength * 1e7  # cm to nm
        FSR_displacement = wavelength_nm / 2  # Half wavelength shift for FSR
        num_FSR_in_scan = (2 * self.piezo.voltage_amplitude * 
                           self.piezo.displacement_per_volt) / FSR_displacement
        
        niceprint("<u> Analysis </u>", 5)
        niceprint(f"Wavelength: {wavelength_nm:7.1f} nm <br>" +
                  f"FSR in displacement: {FSR_displacement:7.2f} nm <br>" +
                  f"Number of FSRs per scan: {num_FSR_in_scan:7.2f} <br>" +
                  f"Expected resonance peaks: {int(np.ceil(num_FSR_in_scan)):7d}"
                  )
    
    def compare_telescope_effect(self, optimizer, optimizer_results, laser,
                            input_power=1e-3):
        niceprint('---')
        niceprint(f"**Effect of Telescope on Cavity**", 3)
        
        total_distance = optimizer_results['total_distance']
        
        eff_with_tele = optimizer_results['coupling_efficiency']
        
        q_start = optimizer.beam.q_from_waist(laser.w0, 0, laser.z0_location)
        M_free_space = optimizer.beam.free_space(total_distance)
        q_no_tele = optimizer.beam.propagate_q(q_start, M_free_space)
        eff_no_tele = optimizer.coupling_efficiency(q_no_tele, self.cavity.q_at_input)
        
        w_cavity = optimizer.beam.waist_from_q(self.cavity.q_at_input)
        R_cavity = optimizer.beam.R_from_q(self.cavity.q_at_input)
        
        improvement = eff_with_tele / eff_no_tele if eff_no_tele > 0 else np.inf
        
        niceprint('<u> Coupling Efficiency </u>', 5)
        niceprint(f"Without telescope: {eff_no_tele*100:7.2f} % <br>" +
                f"With telescope:    {eff_with_tele*100:7.2f} % <br>" +
                f"Improvement:      {improvement:7.2f}x")
        
        
        # transmission parameters
        R_mirrors = self.cavity.geom.R_mirrors
        R_in, R_out = R_mirrors[0], R_mirrors[1]
        R_rt = np.prod(R_mirrors)
        T_in = 1 - R_in
        T_out = 1 - R_out
        sqrt_R = np.sqrt(R_rt)
        if sqrt_R < 1:
            T_max = T_in * T_out / (1 - sqrt_R)**2
        else:
            T_max = 0
        
        # photodetector params (DET36A2 at 632.8 nm with 10k load)
        responsivity = self.detector.responsivity  # A/W
        load_R = self.detector.load_resistance  # Ohms
        
        # peak transmitted power (on resonance)
        P_trans_no_tele = input_power * eff_no_tele * T_max
        P_trans_with_tele = input_power * eff_with_tele * T_max
        
        # peak transmitted voltage
        V_peak_no_tele = P_trans_no_tele * responsivity * load_R
        V_peak_with_tele = P_trans_with_tele * responsivity * load_R
        
        niceprint('<u> Expected Photodetector Output </u>', 5)
        niceprint(f"Peak without telescope: {V_peak_no_tele*1000:7.2f} mV <br>" +
                f"Peak with telescope:    {V_peak_with_tele*1000:7.2f} mV <br>" +
                f"Improvement:           {improvement:7.2f}x")



class QuantumState:
    """
    Represents a quantum state for interferometry and quantum optics.
    
    Math: A quantum state is represented as |ψ⟩ = Σᵢ cᵢ|i⟩
    where |i⟩ are basis states (e.g., different optical paths)
    and cᵢ are complex probability amplitudes with Σᵢ|cᵢ|² = 1.
    
    Example:
    --------
    # Photon in path 0
    >>> state = QuantumState(0, num_paths=2)
    
    # Apply beam splitter
    >>> bs = BeamSplitter()
    >>> new_state = state.apply_operator(bs.get_quantum_operator())
    >>> new_state.measure(0)  # Probability in path 0
    0.5
    """
    
    def __init__(self, state, num_paths: int = 2):
        """
        Parameters:
        -----------
        state : Qobj or int
            If int: create basis state |i⟩
            If Qobj: use provided quantum state
        num_paths : int
            Number of optical paths/modes
        """
        if isinstance(state, int):
            # Create basis state |i⟩
            self.state = basis(num_paths, state)
        else:
            self.state = state
        self.num_paths = num_paths
    
    def measure(self, path_index: int) -> float:
        """
        Measure probability of finding photon in specified path.
        
        Math: P(path i) = |⟨i|ψ⟩|² = |cᵢ|²
        
        Parameters:
        -----------
        path_index : int
            Index of path to measure (0, 1, 2, ...)
        
        Returns:
        --------
        float
            Probability of finding photon in that path (0 to 1)
        """
        amplitude = self.state[path_index, 0]
        return np.abs(amplitude)**2
    
    def apply_operator(self, operator: Qobj) -> 'QuantumState':
        """
        Apply unitary operator to state.
        
        Math: |ψ_out⟩ = Û|ψ_in⟩
        
        Parameters:
        -----------
        operator : Qobj
            Unitary operator (e.g., from BeamSplitter, PhaseShifter)
        
        Returns:
        --------
        QuantumState
            New quantum state after transformation
        """
        new_state = operator * self.state
        return QuantumState(new_state, self.num_paths)
    
    def get_amplitudes(self) -> np.ndarray:
        """Get complex probability amplitudes for all paths"""
        return np.array([self.state[i, 0] for i in range(self.num_paths)])
    
    def __repr__(self):
        probs = [self.measure(i) for i in range(self.num_paths)]
        return f"QuantumState(probs={[f'{p:.3f}' for p in probs]})"


class BeamSplitter:
    """
    50:50 beam splitter for quantum optics.
    
    Math: Unitary transformation matrix
        Û_BS = (1/√2) [[1,  1],
                       [1, -1]]
    
    Physical meaning:
    - Input photon in path 0 → equal superposition (|0⟩ + |1⟩)/√2
    - Input photon in path 1 → equal superposition (|0⟩ - |1⟩)/√2
    - Relative π phase shift on one path (sign difference)
    
    Example:
    --------
    >>> state_in = QuantumState(0, num_paths=2)  # Photon in path 0
    >>> bs = BeamSplitter()
    >>> state_out = state_in.apply_operator(bs.get_quantum_operator())
    >>> state_out.measure(0), state_out.measure(1)
    (0.5, 0.5)  # Equal probability in both paths
    """
    
    def __init__(self, input_a: int = 0, input_b: int = 1):
        """
        Parameters:
        -----------
        input_a, input_b : int
            Path indices that the beam splitter couples
        """
        self.input_a = input_a
        self.input_b = input_b
    
    def get_quantum_operator(self, num_paths: int = 2) -> Qobj:
        """
        Get quantum operator for beam splitter.
        
        Returns:
        --------
        Qobj
            2×2 unitary matrix representing beam splitter transformation
        """
        data = np.eye(num_paths, dtype=complex)
        data[self.input_a, self.input_a] = 1/np.sqrt(2)
        data[self.input_a, self.input_b] = 1/np.sqrt(2)
        data[self.input_b, self.input_a] = 1/np.sqrt(2)
        data[self.input_b, self.input_b] = -1/np.sqrt(2)
        return Qobj(data)
    
    def __repr__(self):
        return f"BeamSplitter(paths={self.input_a},{self.input_b})"


class PhaseShifter:
    """
    Phase shifter for quantum optics.
    
    Math: Applies phase shift e^(iφ) to one path
        Û_φ = [[e^(iφ), 0    ],
               [0,      1    ]]  (for path 0)
    
    Physical meaning:
    - Adds optical path length Δ = φλ/(2π)
    - Changes relative phase between paths
    - Key for interferometer control
    
    Example:
    --------
    >>> # Create superposition
    >>> state = QuantumState(0, num_paths=2)
    >>> bs = BeamSplitter()
    >>> state = state.apply_operator(bs.get_quantum_operator())
    
    >>> # Add π/2 phase shift to path 1
    >>> ps = PhaseShifter(path_index=1, phase=np.pi/2)
    >>> state = state.apply_operator(ps.get_quantum_operator())
    """
    
    def __init__(self, path_index: int, phase: float):
        """
        Parameters:
        -----------
        path_index : int
            Which path gets the phase shift
        phase : float
            Phase shift in radians
        """
        self.path_index = path_index
        self.phase = phase
    
    def get_quantum_operator(self, num_paths: int = 2) -> Qobj:
        """
        Get quantum operator for phase shifter.
        
        Returns:
        --------
        Qobj
            Diagonal matrix with e^(iφ) on specified path
        """
        data = np.eye(num_paths, dtype=complex)
        data[self.path_index, self.path_index] = np.exp(1j * self.phase)
        return Qobj(data)
    
    def __repr__(self):
        return f"PhaseShifter(path={self.path_index}, φ={self.phase:.3f}rad)"


class Mirror:
    """
    Mirror for quantum optics (introduces π phase shift).
    
    Math: Reflection introduces π phase shift
        Û_mirror = -I = [[-1, 0 ],
                         [0, -1]]
    
    Physical meaning:
    - Reflection causes π phase shift (flips sign)
    - Important for interferometer analysis
    
    Note: Can specify which path gets reflected (for asymmetric setups)
    """
    
    def __init__(self, path_index: int = None):
        """
        Parameters:
        -----------
        path_index : int or None
            If specified, only that path gets π shift
            If None, all paths get π shift
        """
        self.path_index = path_index
    
    def get_quantum_operator(self, num_paths: int = 2) -> Qobj:
        """Get quantum operator for mirror"""
        data = np.eye(num_paths, dtype=complex)
        if self.path_index is None:
            # All paths get π shift
            data = -data
        else:
            # Only specified path gets π shift
            data[self.path_index, self.path_index] = -1
        return Qobj(data)
    
    def __repr__(self):
        if self.path_index is None:
            return "Mirror(all paths)"
        return f"Mirror(path={self.path_index})"


class MachZehnderInterferometer:
    """
    Mach-Zehnder interferometer with quantum state evolution.
    
    Configuration:
                BS1
               ╱│╲
              ╱ │ ╲
          M0 ╱  │  ╲ M1
            │   │   │
            │   │   │ (path length difference ΔL)
             ╲  │  ╱
              ╲ │ ╱
               ╲│╱
                BS2
    
    Math: Output probabilities depend on path difference
        P₀ = cos²(φ/2)
        P₁ = sin²(φ/2)
    where φ = 2πn·ΔL/λ is the phase difference.
    
    Example:
    --------
    >>> # Constructive interference in output 0
    >>> mz = MachZehnderInterferometer(delta_L=0, wavelength=632.8e-9)
    >>> P0, P1 = mz.get_output_probabilities()
    >>> print(f"P0 = {P0:.3f}, P1 = {P1:.3f}")
    P0 = 1.000, P1 = 0.000
    
    >>> # Destructive interference (π phase shift)
    >>> mz = MachZehnderInterferometer(delta_L=316.4e-9, wavelength=632.8e-9)
    >>> P0, P1 = mz.get_output_probabilities()
    >>> print(f"P0 = {P0:.3f}, P1 = {P1:.3f}")
    P0 = 0.000, P1 = 1.000
    """
    
    def __init__(self, delta_L: float = 0, wavelength: float = 632.8e-9,
                 n: float = 1.0, include_mirrors: bool = True):
        """
        Parameters:
        -----------
        delta_L : float
            Path length difference (meters)
        wavelength : float
            Wavelength (meters)
        n : float
            Refractive index
        include_mirrors : bool
            Whether to include mirror phase shifts in calculation
        """
        self.delta_L = delta_L
        self.wavelength = wavelength
        self.n = n
        self.include_mirrors = include_mirrors
        
        # Calculate phase shift from path difference
        # φ = 2π·n·ΔL/λ
        self.delta_phi = (2 * np.pi / wavelength) * n * delta_L
    
    def get_output_probabilities(self) -> Tuple[float, float]:
        """
        Calculate output probabilities at both ports.
        
        Returns:
        --------
        P0, P1 : float, float
            Probabilities at output ports 0 and 1
        """
        # Start with photon in input port 0
        state = QuantumState(0, num_paths=2)
        
        # First beam splitter
        bs1 = BeamSplitter(0, 1)
        state = state.apply_operator(bs1.get_quantum_operator())
        
        # Mirrors (if included)
        if self.include_mirrors:
            m0 = Mirror(path_index=0)
            m1 = Mirror(path_index=1)
            state = state.apply_operator(m0.get_quantum_operator())
            state = state.apply_operator(m1.get_quantum_operator())
        
        # Phase shift on path 1
        ps = PhaseShifter(path_index=1, phase=self.delta_phi)
        state = state.apply_operator(ps.get_quantum_operator())
        
        # Second beam splitter
        bs2 = BeamSplitter(0, 1)
        state = state.apply_operator(bs2.get_quantum_operator())
        
        # Measure output probabilities
        P0 = state.measure(0)
        P1 = state.measure(1)
        
        return P0, P1
    
    def scan_interference(self, delta_L_range: Tuple[float, float],
                         num_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scan path length difference and calculate interference pattern.
        
        Parameters:
        -----------
        delta_L_range : tuple
            (min_delta_L, max_delta_L) in meters
        num_points : int
            Number of points to sample
        
        Returns:
        --------
        delta_L_array : np.ndarray
            Path differences (meters)
        P0_array : np.ndarray
            Probabilities at output 0
        P1_array : np.ndarray
            Probabilities at output 1
        """
        delta_L_array = np.linspace(delta_L_range[0], delta_L_range[1], num_points)
        P0_array = np.zeros(num_points)
        P1_array = np.zeros(num_points)
        
        for i, dL in enumerate(delta_L_array):
            # Create new interferometer with this path difference
            mz_temp = MachZehnderInterferometer(
                delta_L=dL,
                wavelength=self.wavelength,
                n=self.n,
                include_mirrors=self.include_mirrors
            )
            P0, P1 = mz_temp.get_output_probabilities()
            P0_array[i] = P0
            P1_array[i] = P1
        
        return delta_L_array, P0_array, P1_array
    
    def __repr__(self):
        return (f"MachZehnderInterferometer(ΔL={self.delta_L*1e9:.1f}nm, "
                f"λ={self.wavelength*1e9:.1f}nm, φ={self.delta_phi:.3f}rad)")



def plot_beam_propagation(beam: GaussianBeam, w0: float, z0: float,
                         z_range: Tuple[float, float],
                         components: List[Tuple[str, float, float]] = None,
                         num_points: int = 500,
                         save_path: str = None):
    """
    Plot Gaussian beam propagation showing how beam width evolves.
    
    Math: w(z) = w₀√(1 + (z/z_R)²) where z_R = πw₀²/λ
    
    Parameters:
    -----------
    beam : GaussianBeam
        GaussianBeam object with wavelength
    w0 : float
        Initial beam waist (cm)
    z0 : float
        Initial waist location (cm)
    z_range : tuple
        (z_min, z_max) range to plot (cm)
    components : list, optional
        List of (type, position, parameter) for optical elements
        type can be 'lens' (parameter=focal length) or 'space'
    num_points : int
        Number of points to plot
    save_path : str, optional
        Path to save figure
    
    Example:
    --------
    >>> beam = GaussianBeam(wavelength=632.8e-7)  # HeNe in cm
    >>> plot_beam_propagation(
    ...     beam, w0=0.05, z0=0,
    ...     z_range=(0, 100),
    ...     components=[('lens', 20, 30), ('space', 50, None)]
    ... )
    """
    z_array = np.linspace(z_range[0], z_range[1], num_points)
    w_array = np.zeros(num_points)
    
    # Calculate initial q-parameter
    q = beam.q_from_waist(w0, 0, z0)
    
    # Track position and q-parameter through system
    z_current = z_range[0]
    q_current = q
    
    if components is None:
        # Simple free space propagation
        for i, z in enumerate(z_array):
            dz = z - z_range[0]
            q_at_z = beam.propagate_q(q, beam.free_space(dz))
            w_array[i] = beam.get_beam_width(q_at_z)
    else:
        # Propagation through optical system
        # This is a simplified version - for full implementation,
        # would need to track through each component
        comp_positions = []
        for comp_type, position, param in components:
            comp_positions.append(position)
            if comp_type == 'lens':
                # Apply lens at this position
                dz = position - z_range[0]
                q_temp = beam.propagate_q(q, beam.free_space(dz))
                M = beam.thin_lens(param)
                q_current = beam.propagate_q(q_temp, M)
        
        # Plot beam width
        for i, z in enumerate(z_array):
            dz = z - z_range[0]
            q_at_z = beam.propagate_q(q, beam.free_space(dz))
            w_array[i] = beam.get_beam_width(q_at_z)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(z_array, w_array * 1e4, 'b-', linewidth=2, label='Beam width w(z)')
    ax.axhline(y=w0 * 1e4, color='r', linestyle='--', alpha=0.5, label=f'Waist w₀ = {w0*1e4:.1f} μm')
    
    # Mark component positions
    if components:
        for comp_type, position, param in components:
            if comp_type == 'lens':
                ax.axvline(x=position, color='g', linestyle=':', alpha=0.7)
                ax.text(position, ax.get_ylim()[1]*0.9, f'Lens\nf={param:.1f}cm',
                       ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Position z (cm)', fontsize=12)
    ax.set_ylabel('Beam Width w(z) (μm)', fontsize=12)
    ax.set_title('Gaussian Beam Propagation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def stability_map(cavity_builder_func,
                             param1_name: str, param1_range: Tuple[float, float],
                             param2_name: str, param2_range: Tuple[float, float],
                             param1_design: float, param2_design: float,
                             num_points: int = 100,
                             save_path: str = None):
    """
    Create contour plot showing cavity stability and FSR as function of parameters.
    
    Parameters:
    -----------
    cavity_builder_func : callable
        Function that takes (param1, param2) and returns BowTieCavity object
    param1_name, param2_name : str
        Names for the two parameters (e.g., "Width W", "Height H")
    param1_range, param2_range : tuple
        (min, max) ranges for parameters
    param1_design, param2_design : float
        Design point to mark on plot
    num_points : int
        Resolution of grid
    save_path : str, optional
        Path to save figure
    
    Example:
    --------
    >>> # Define cavity builder
    >>> def build_cavity(W, H):
    ...     geom = CavityGeometry(W=W, H=H, f_concave=50,
    ...                          R_mirrors=[0.99, 0.99, 1.0, 1.0],
    ...                          wavelength=632.8e-7)
    ...     cavity = BowTieCavity(geom)
    ...     cavity.analyze_cavity()
    ...     return cavity
    >>> 
    >>> stability_map(
    ...     build_cavity,
    ...     "Width W", (10, 30),
    ...     "Height H", (5, 20),
    ...     param1_design=20, param2_design=10
    ... )
    """
    # Create parameter grids
    p1_arr = np.linspace(param1_range[0], param1_range[1], num_points)
    p2_arr = np.linspace(param2_range[0], param2_range[1], num_points)
    P1_grid, P2_grid = np.meshgrid(p1_arr, p2_arr)
    
    FSR_grid = np.zeros_like(P1_grid)
    stability_grid = np.zeros_like(P1_grid)
    
    # Calculate properties over parameter space
    print("Calculating cavity properties over parameter space...")
    for i in range(num_points):
        for j in range(num_points):
            p1, p2 = P1_grid[i, j], P2_grid[i, j]
            try:
                cavity = cavity_builder_func(p1, p2)
                if cavity.is_stable and cavity.FSR:
                    FSR_grid[i, j] = cavity.FSR
                    stability_grid[i, j] = cavity.stability_param
                else:
                    FSR_grid[i, j] = 0
                    stability_grid[i, j] = 2  # Mark as unstable
            except:
                FSR_grid[i, j] = 0
                stability_grid[i, j] = 2
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: FSR contours
    FSR_GHz = FSR_grid / 1e9
    FSR_GHz[FSR_GHz == 0] = np.nan  # Don't show unstable regions
    
    levels_fsr = np.linspace(np.nanmin(FSR_GHz), np.nanmax(FSR_GHz), 25)
    contour_fsr = ax1.contourf(P1_grid, P2_grid, FSR_GHz,
                              levels=levels_fsr, cmap='plasma')
    contour_lines = ax1.contour(P1_grid, P2_grid, FSR_GHz,
                               levels=15, colors='white', linewidths=0.5, alpha=0.4)
    ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    
    # Mark stability boundary
    stability_boundary = ax1.contour(P1_grid, P2_grid, stability_grid,
                                    levels=[1.0], colors='cyan', linewidths=3)
    ax1.clabel(stability_boundary, inline=True, fontsize=10, fmt='Stability Limit')
    
    # Shade unstable region
    unstable_mask = stability_grid > 1
    ax1.contourf(P1_grid, P2_grid, unstable_mask.astype(float),
                levels=[0.5, 1.5], colors='black', alpha=0.2)
    
    # Mark design point
    ax1.plot(param1_design, param2_design, 'r*', markersize=20,
            markeredgecolor='white', markeredgewidth=2,
            label=f'{param1_name}: {param1_design:.1f}, {param2_name}: {param2_design:.1f}')
    
    plt.colorbar(contour_fsr, ax=ax1, label='FSR (GHz)', pad=0.02)
    ax1.set_xlabel(param1_name + ' (cm)', fontsize=11)
    ax1.set_ylabel(param2_name + ' (cm)', fontsize=11)
    ax1.set_title('Free Spectral Range', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Stability parameter
    levels_stab = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    colors_stab = plt.cm.RdYlGn_r(np.linspace(0, 1, len(levels_stab)-1))
    
    contour_stab = ax2.contourf(P1_grid, P2_grid, stability_grid,
                                levels=levels_stab, colors=colors_stab, alpha=0.8)
    contour_lines_stab = ax2.contour(P1_grid, P2_grid, stability_grid,
                                     levels=levels_stab, colors='black',
                                     linewidths=1, alpha=0.5)
    ax2.clabel(contour_lines_stab, inline=True, fontsize=9, fmt='%.1f')
    
    # Mark stability boundary
    boundary = ax2.contour(P1_grid, P2_grid, stability_grid,
                          levels=[1.0], colors='red', linewidths=4)
    ax2.clabel(boundary, inline=True, fontsize=12, fmt='Stability Limit')
    
    # Mark design point
    ax2.plot(param1_design, param2_design, 'r*', markersize=20,
            markeredgecolor='white', markeredgewidth=2,
            label=f'{param1_name}: {param1_design:.1f}, {param2_name}: {param2_design:.1f}')
    
    cbar2 = plt.colorbar(contour_stab, ax=ax2, label='Stability Parameter', pad=0.02)
    ax2.set_xlabel(param1_name + ' (cm)', fontsize=11)
    ax2.set_ylabel(param2_name + ' (cm)', fontsize=11)
    ax2.set_title('Cavity Stability (Stable: 0 < g < 1)', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2)


class PolarizationState:
    """
    Jones vector for polarization state.
    
    Representation: [a_H, a_V]^T where a_H, a_V are complex amplitudes
    Normalization: |a_H|^2 + |a_V|^2 = 1 for unit intensity
    
    Source: Polarization_Notes_10.pdf, Eq. 4 (page 2)
    """
    
    def __init__(self, jones_vector: np.ndarray, normalize: bool = True):
        self.vector = np.array(jones_vector, dtype=complex)
        if self.vector.shape != (2,):
            raise ValueError("Jones vector must be 2D")
        
        if normalize and self.intensity() > 0:
            self.vector = self.vector / np.sqrt(self.intensity())
    
    def intensity(self) -> float:
        """Intensity I ∝ ψ†ψ (Eq. 10, page 4)"""
        return float(np.real(np.vdot(self.vector, self.vector)))
    
    def get_components(self) -> Tuple[complex, complex]:
        """Return (a_H, a_V)"""
        return self.vector[0], self.vector[1]
    
    # Basis states
    @classmethod
    def horizontal(cls):
        """H = [1, 0]^T"""
        return cls(np.array([1.0, 0.0]))
    
    @classmethod
    def vertical(cls):
        """V = [0, 1]^T"""
        return cls(np.array([0.0, 1.0]))
    
    @classmethod
    def diagonal(cls):
        """D = [1, 1]^T/√2 (Eq. 16, page 7)"""
        return cls(np.array([1.0, 1.0]) / np.sqrt(2))
    
    @classmethod
    def antidiagonal(cls):
        """A = [1, -1]^T/√2 (Eq. 17, page 7)"""
        return cls(np.array([1.0, -1.0]) / np.sqrt(2))
    
    @classmethod
    def right_circular(cls):
        """RCP = [1, -i]^T/√2"""
        return cls(np.array([1.0, -1.0j]) / np.sqrt(2))
    
    @classmethod
    def left_circular(cls):
        """LCP = [1, i]^T/√2"""
        return cls(np.array([1.0, 1.0j]) / np.sqrt(2))
    
    @classmethod
    def linear(cls, angle: float):
        """Linear polarization at angle (radians) from H"""
        return cls(np.array([np.cos(angle), np.sin(angle)]))


class PolarizationOptic:
    """
    Base class for Jones matrices.
    
    Transforms polarization: ψ_out = T·ψ_in
    Composition: T_total = T_n ··· T_2·T_1 (right to left)
    
    Source: Eq. 5-6 (pages 3-4)
    """
    
    def __init__(self, jones_matrix: np.ndarray):
        self.matrix = np.array(jones_matrix, dtype=complex)
        if self.matrix.shape != (2, 2):
            raise ValueError("Jones matrix must be 2x2")
    
    def apply(self, state: PolarizationState) -> PolarizationState:
        """Apply T to state"""
        return PolarizationState(self.matrix @ state.vector, normalize=False)
    
    def __matmul__(self, other):
        """Compose optics: T3 = T2 @ T1"""
        if isinstance(other, PolarizationOptic):
            return PolarizationOptic(self.matrix @ other.matrix)
        raise TypeError("Can only compose PolarizationOptic")
    
    @staticmethod
    def rotation_matrix(angle: float) -> np.ndarray:
        """2D rotation by angle (Eq. 11, page 5)"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])


class Polarizer(PolarizationOptic):
    """
    Linear polarizer at angle from H.
    
    Horizontal polarizer: T_H = [[1, 0], [0, 0]]
    Rotated by α: T(α) = R(α)·T_H·R(-α)
    
    Source: Eq. 8, 12 (pages 4-5)
    """
    
    def __init__(self, angle: float = 0.0):
        # Horizontal polarizer base
        T_H = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        
        # Rotate if needed
        if angle != 0:
            R = self.rotation_matrix(angle)
            jones_matrix = R @ T_H @ R.T
        else:
            jones_matrix = T_H
        
        super().__init__(jones_matrix)
    
    @classmethod
    def horizontal(cls):
        return cls(0.0)
    
    @classmethod
    def vertical(cls):
        return cls(np.pi/2)


class WavePlate(PolarizationOptic):
    """
    Half-wave or quarter-wave plate.
    
    HWP (δ=π): T = [[1, 0], [0, -1]] with fast axis along H
    QWP (δ=π/2): T = [[1, 0], [0, -i]] with fast axis along H
    
    Source: Eq. 37 (page 15), Eq. 42 (page 17)
    """
    
    def __init__(self, plate_type: Literal['half', 'quarter'], 
                 fast_axis_angle: float = 0.0):
        # Base matrix (fast axis horizontal)
        if plate_type == 'half':
            T_base = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        elif plate_type == 'quarter':
            T_base = np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=complex)
        else:
            raise ValueError("plate_type must be 'half' or 'quarter'")
        
        # Rotate
        if fast_axis_angle != 0:
            R = self.rotation_matrix(fast_axis_angle)
            jones_matrix = R @ T_base @ R.T
        else:
            jones_matrix = T_base
        
        super().__init__(jones_matrix)
    
    @classmethod
    def half_wave(cls, fast_axis_angle: float = 0.0):
        """HWP rotates linear polarization by 2θ"""
        return cls('half', fast_axis_angle)
    
    @classmethod
    def quarter_wave(cls, fast_axis_angle: float = 0.0):
        """QWP at 45° creates circular from linear"""
        return cls('quarter', fast_axis_angle)


class PolarizingBeamSplitter:
    """
    PBS with two outputs: H (transmitted) and V (reflected).
    
    Not a PolarizationOptic because it has two output ports.
    
    Source: Lab P Section 2, Problem 2
    """
    
    def __init__(self):
        self.T_trans = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        self.T_refl = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    
    def split(self, state: PolarizationState) -> Tuple[PolarizationState, PolarizationState]:
        """Return (transmitted, reflected) states"""
        trans = PolarizationState(self.T_trans @ state.vector, normalize=False)
        refl = PolarizationState(self.T_refl @ state.vector, normalize=False)
        return trans, refl
    
    def get_powers(self, state: PolarizationState) -> Tuple[float, float]:
        """Return (P_transmitted, P_reflected)"""
        trans, refl = self.split(state)
        return trans.intensity(), refl.intensity()


# =============================================================================
# LAB P SECTION 2 - Analytical Solutions
# =============================================================================

def crossed_polarizers_transmission(theta: np.ndarray) -> np.ndarray:
    """
    Problem 1: H-pol → Pol(θ) → V-pol transmission.
    
    Result: T(θ) = (1/4)sin²(2θ)
    Maximum at θ = 45°: T = 0.25
    
    Derivation:
        ψ_in = [1, 0]^T
        After Pol(θ): [cos²θ, cosθ·sinθ]^T  
        After V-pol: [0, cosθ·sinθ]^T
        I ∝ |cosθ·sinθ|² = (1/4)sin²(2θ)
    """
    return 0.25 * np.sin(2 * theta)**2


def hwp_pbs_powers(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Problem 3a: V → HWP(θ) → PBS powers.
    
    Results:
        P_trans = sin²(2θ)
        P_refl = cos²(2θ)
    
    Derivation:
        HWP matrix at θ: [[cos(2θ), sin(2θ)], [sin(2θ), -cos(2θ)]]
        V input [0,1]^T → [sin(2θ), -cos(2θ)]^T
        PBS splits into H and V components
    
    Source: Eq. 44 (page 17)
    """
    return np.sin(2 * theta)**2, np.cos(2 * theta)**2


def qwp_pbs_powers(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Problem 3b: V → QWP(θ) → PBS powers.
    
    Results:
        P_trans = (1/2)sin²(2θ)
        P_refl = 1 - (1/2)sin²(2θ)
    
    At θ=45°: creates circular, so P_trans = P_refl = 0.5
    
    Derivation:
        QWP at θ from Eq. 43 (page 17)
        V input → [(1+i)cosθ·sinθ, sin²θ - i·cos²θ]^T
        |H-component|² = |1+i|²·cos²θ·sin²θ = 2cos²θ·sin²θ = (1/2)sin²(2θ)
    
    Source: Eq. 43 (page 17)
    """
    return 0.5 * np.sin(2 * theta)**2, 1 - 0.5 * np.sin(2 * theta)**2


def plot_polarization_analysis(save_path: Optional[str] = None):
    """
    Create comprehensive plots for all Lab P Section 2 problems.
    """
    angles_deg = np.linspace(0, 180, 1000)
    angles_rad = np.deg2rad(angles_deg)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Crossed polarizers
    ax1 = fig.add_subplot(gs[0, 0])
    T_crossed = crossed_polarizers_transmission(angles_rad)
    ax1.plot(angles_deg, T_crossed, 'b-', linewidth=2)
    ax1.axvline(45, color='r', linestyle='--', alpha=0.5, label='Max at 45°')
    ax1.set_xlabel('Middle Polarizer Angle (degrees)', fontsize=11)
    ax1.set_ylabel('Transmission', fontsize=11)
    ax1.set_title('Problem 1: Crossed Polarizers', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 180)
    
    # Plot 2: HWP + PBS
    ax2 = fig.add_subplot(gs[0, 1])
    P_hwp_t, P_hwp_r = hwp_pbs_powers(angles_rad)
    ax2.plot(angles_deg, P_hwp_t, 'b-', linewidth=2, label='Transmitted (H)')
    ax2.plot(angles_deg, P_hwp_r, 'r-', linewidth=2, label='Reflected (V)')
    ax2.set_xlabel('HWP Fast Axis Angle (degrees)', fontsize=11)
    ax2.set_ylabel('Power', fontsize=11)
    ax2.set_title('Problem 3a: V → HWP → PBS', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 180)
    ax2.set_ylim(-0.05, 1.05)
    
    # Plot 3: QWP + PBS
    ax3 = fig.add_subplot(gs[1, 0])
    P_qwp_t, P_qwp_r = qwp_pbs_powers(angles_rad)
    ax3.plot(angles_deg, P_qwp_t, 'b-', linewidth=2, label='Transmitted (H)')
    ax3.plot(angles_deg, P_qwp_r, 'r-', linewidth=2, label='Reflected (V)')
    ax3.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(45, color='gray', linestyle=':', alpha=0.5, label='Circular @ 45°')
    ax3.set_xlabel('QWP Fast Axis Angle (degrees)', fontsize=11)
    ax3.set_ylabel('Power', fontsize=11)
    ax3.set_title('Problem 3b: V → QWP → PBS', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0, 180)
    ax3.set_ylim(-0.05, 1.05)
    
    # Plot 4: Comparison HWP vs QWP
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(angles_deg, P_hwp_t, 'b-', linewidth=2, label='HWP')
    ax4.plot(angles_deg, P_qwp_t, 'r-', linewidth=2, label='QWP')
    ax4.set_xlabel('Waveplate Fast Axis Angle (degrees)', fontsize=11)
    ax4.set_ylabel('Transmitted Power', fontsize=11)
    ax4.set_title('Comparison: HWP vs QWP Transmission', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim(0, 180)
    ax4.set_ylim(-0.05, 1.05)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig