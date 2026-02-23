from ..general import *
from .params import *
from .simulators import *
from scipy.constants import h, c

import numpy as np
from qutip import *
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict
import warnings



class FiberModeMatchOptimizer:
    """
    Optimize fiber mode-matching for a packaged fiber collimator.

    Assumes a fixed optical layout with a telescope followed by a fiber
    collimator. The telescope and fiber parameters are fixed, and the 
    optimizer adjusts the distance from the second telescope lens to the 
    collimator `d_L2coll` to achieve optimal mode-matching.
    
    The optimizer minimizes the normalized q-mismatch at the fiber face:

        mismatch = |q_beam - q_fiber| / |q_fiber|

    where q_fiber = i * pi * w_fiber^2 / lambda  (ideal flat-wavefront Gaussian).
    """

    def __init__(
        self,
        laser:            LaserBeam,
        telescope:        Telescope,
        f_collimator:     float,
        w_fiber:          float,
        d0:               float,
        d_12:  float,   # fixed total L1-to-L2 distance [m]
    ):
        self.laser            = laser
        self.telescope        = telescope
        self.f_coll           = f_collimator
        self.w_fiber          = w_fiber
        self.d0               = d0
        self.d_12  = d_12
        self.wavelength       = laser.wavelength
        self.beam             = GaussianBeamTool(laser.wavelength)
        self.results: Optional[Dict] = None

        # target q
        self._q_fiber = 1j * (np.pi * w_fiber**2 / laser.wavelength)

    def _propagate_to_fiber(self, d_L2coll: float) -> complex:
        """
        Propagate q from the laser waist through the full train to the fiber face.

        The fiber face is the back focal plane of the collimator, so q is
        evaluated immediately after the collimator lens with no further propagation.

        Returns
        -------
        q_at_fiber : complex
        """
        q0 = self.beam.q_from_waist(
            self.laser.w0,
            z=0.0,
            z0=self.laser.z0_location,
        )

        circ = OpticalCircuit(wavelength=self.wavelength)
        circ.add_free_space(self.d0) # laser waist to L1
        circ.add_thin_lens(self.telescope.f1) # L1
        circ.add_free_space(self.d_12) # L1 to L2 (fixed)
        circ.add_thin_lens(self.telescope.f2) # L2
        circ.add_free_space(d_L2coll) # L2 to collimator front
        circ.add_thin_lens(self.f_coll) # collimator lens; fiber is here

        M = circ.get_total_abcd_matrix()
        return self.beam.propagate_q(q0, M)

    def _cost(self, params: np.ndarray) -> float:
        """Normalized |Delta q|/|q_fiber| at the fiber face."""
        d_L2coll = params[0]

        if d_L2coll < 1e-3 or d_L2coll > 1.0:
            return 1e6

        try:
            q = self._propagate_to_fiber(d_L2coll)
        except Exception:
            return 1e6

        if q.imag <= 0:
            return 1e6

        return float(abs(q - self._q_fiber) / abs(self._q_fiber))

    def _coupling_efficiency(self, q_at_fiber: complex) -> float:
        """
        Gaussian mode-overlap coupling efficiency.
        """
        w_b  = self.beam.waist_from_q(q_at_fiber)
        w_f  = self.w_fiber
        R_b  = self.beam.R_from_q(q_at_fiber)

        ratio     = w_b / w_f
        curv_term = 0.0 if np.isinf(R_b) else (self.wavelength * R_b) / (np.pi * w_b * w_f)

        return 4.0 / ((ratio + 1.0 / ratio)**2 + curv_term**2)

    def optimize(self) -> Dict:
        """
        Global differential-evolution search over d_L2coll in [1 mm, 1 m].

        Returns
        -------
        dict with keys:
            d_12  – fixed L1-to-L2 spacing [m]
            d_L2coll         – optimized L2-to-collimator distance [m]
            mismatch         – normalized |Delta q|/|q_fiber|
            coupling_eff     – Gaussian mode overlap η  (0 to 1)
            w_at_fiber       – achieved beam waist at fiber face [m]
            R_at_fiber       – wavefront radius of curvature at fiber face [m]
            w_fiber_target   – target fiber mode-field radius [m]
            success          – scipy optimizer success flag
        """
        result = differential_evolution(
            self._cost,
            bounds=[(1e-3, 1.0)],
            maxiter=1000,
            popsize=20,
            tol=1e-14,
            seed=42,
            disp=False,
            workers=1,
        )

        d_L2coll_opt  = result.x[0]
        q_at_fiber    = self._propagate_to_fiber(d_L2coll_opt)

        self.results = {
            "d_12": self.d_12,
            "d_L2coll":        d_L2coll_opt,
            "mismatch":        result.fun,
            "coupling_eff":    self._coupling_efficiency(q_at_fiber),
            "w_at_fiber":      self.beam.waist_from_q(q_at_fiber),
            "R_at_fiber":      self.beam.R_from_q(q_at_fiber),
            "w_fiber_target":  self.w_fiber,
            "success":         result.success,
        }
        return self.results

    def print_summary(self):
        r = self.results

        niceprint("---")
        niceprint("**Fiber Mode-Matching Optimization**", 3)

        niceprint("<u> Fixed System Parameters </u>", 5)
        niceprint(
            f"Laser waist $w_0$: {self.laser.w0 * 1e3:.3f} mm <br>"
            f"Laser to L1 ($d_0$): {self.d0 * 1e2:.1f} cm <br>"
            f"Telescope: $f_1$ = {self.telescope.f1 * 1e3:.1f} mm, "
            f"$f_2$ = {self.telescope.f2 * 1e3:.1f} mm <br>"
            f"L1-to-L2 separation ($d_{{12}}$): {self.d_12 * 1e2:.1f} cm <br>"
            f"Collimator focal length: {self.f_coll * 1e3:.1f} mm <br>"
            f"Target fiber waist (MFD/2): {self.w_fiber * 1e6:.2f} μm"
        )

        total = self.d0 + r['d_12'] + r['d_L2coll']
        niceprint("<u> Optimized Distances </u>", 5)
        niceprint(
            f"**Given** laser waist → L1: {self.d0 * 1e2:.1f} cm <br>"
            f"**Fixed** L1 → L2 ($d_{{12}}$): {r['d_12'] * 1e2:.2f} cm <br>"
            f"$\\rightarrow$ L2 to collimator ($d_{{2c}}$): {r['d_L2coll'] * 1e2:.2f} cm <br>"
            "───────────────────────── <br>"
            f"Total path length: {total * 1e2:.1f} cm"
        )

        eta = r["coupling_eff"]
        if eta < 0.85:
            eta_note = 'Less than 85% — consider adjusting telescope focal lengths or $d_{12}$.'
        elif eta > 0.95:
            eta_note = 'Excellent — note sensitivity to alignment at this coupling level.'
        else:
            eta_note = 'Good coupling efficiency.'

        niceprint("<u> Optimization Performance </u>", 5)
        niceprint(
            f"Coupling efficiency $\\eta$: {eta * 100:.2f} % <br>"
            f"$\\quad$ {eta_note} <br>"
            f"$\\quad$ Optimization was {'**successful**' if r['success'] else '**not** successful'} <br>"
            f"Normalized mismatch $|\\Delta q|/|q_\\text{{fiber}}|$: {r['mismatch']:.6f}"
        )

        niceprint("<u> Beam Parameters at Fiber Face </u>", 5)
        niceprint(
            f"Achieved beam waist: {r['w_at_fiber'] * 1e6:.3f} μm <br>"
            f"Target fiber waist: {r['w_fiber_target'] * 1e6:.3f} μm <br>"
            f"Waist mismatch: {abs(r['w_at_fiber'] - r['w_fiber_target']) / r['w_fiber_target'] * 100:.2f} % <br>"
            f"Wavefront $R$ at fiber face: "
            + (
                "$\\infty$ (flat – ideal)" if np.isinf(r["R_at_fiber"])
                else f"{r['R_at_fiber'] * 1e3:.2f} mm"
            )
        )

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
            Distance from laser waist to first lens (m)
        """
        if not cavity.results['stable']:
            raise ValueError("Cavity is unstable.")
        
        self.cavity = cavity
        self.laser = laser
        self.telescope = telescope
        self.d_laser_to_L1 = d_laser_to_L1
        self.beam = GaussianBeamTool(laser.wavelength)
        self.wavelength = laser.wavelength
    
    def propagate_laser_to_cavity(self, d_L1_L2: float, d_L2_cavity: float) -> complex:
        """Propagate laser through telescope to cavity input"""
        
        q = self.beam.q_from_waist(self.laser.w0, 0, self.laser.z0_location)
        
        self.circuit = OpticalCircuit(wavelength=self.wavelength)
        self.circuit.add_free_space(self.d_laser_to_L1)
        self.circuit.add_thin_lens(self.telescope.f1)
        self.circuit.add_free_space(d_L1_L2)
        self.circuit.add_thin_lens(self.telescope.f2)
        self.circuit.add_free_space(d_L2_cavity)
        
        M_total = self.circuit.get_total_abcd_matrix()
        
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
        
        # Physical constraints (meters)
        if d_L1_L2 < 0.005 or d_L1_L2 > 1.5:
            return 1e6
        if d_L2_cavity < 0.01 or d_L2_cavity > 1.5:
            return 1e6
        
        # Propagate laser to cavity input
        try:
            q_laser = self.propagate_laser_to_cavity(d_L1_L2, d_L2_cavity)
        except:
            return 1e6
        
        # Calculate mismatch
        return self.match_quality(q_laser, self.cavity.q_at_input)
    
    def find_waist_from_q(self, q_at_ref: complex) -> Tuple[float, float]:
        """
        Find location and size of beam waist from q-parameter.
        
        Returns:
            z_to_waist: distance from reference point to waist (m)
            w0: beam waist size (m)
        """
        # Waist occurs where Re(q) = 0
        z_to_waist = -q_at_ref.real
        
        # Propagate to waist
        q_at_waist = self.beam.propagate_q(q_at_ref, self.beam.free_space(z_to_waist))
        
        # Extract waist size
        w0 = self.beam.waist_from_q(q_at_waist)
        
        return z_to_waist, w0
    
    def optimize(self, method='global') -> Dict:
        """Run optimization"""
        
        # Initial guess (meters)
        x0 = [self.telescope.f1 + self.telescope.f2, 0.30]
        
        if method == 'global':
            bounds = [(0.005, 1.5), (0.01, 1.5)]
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
        self.waist_location = self.cavity.waist_location
        w_cavity_waist = self.cavity.cavity_waist
        circuit = OpticalCircuit(wavelength=self.wavelength)
        circuit.add_free_space(self.waist_location)
        M_total = circuit.get_total_abcd_matrix()
        q_laser_at_waist = self.beam.propagate_q(q_laser_opt, M_total)
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
            'w_cavity_waist': w_cavity_waist,
            'success': result.success
        }
        
        return results
    
    def print_results(self, results: Dict):
        niceprint('---')
        niceprint(f"**Telescope Optimization**", 3)
        
        niceprint(f"<u> Telescope Optimized Distances </u>", 5)
        niceprint(fr"**Given** laser waist to Lens 1: {self.d_laser_to_L1 * 100:8.2f} cm <br>" +
                  fr"$\rightarrow$ Lens 1 to Lens 2: {results['d_L1_L2'] * 100:8.2f} cm <br>" +
                  fr"$\rightarrow$ Lens 2 to Cavity input: {results['d_L2_cavity'] * 100:8.2f} cm <br>" +
                  "───────────────────────── <br>" +
                  fr"Total path length: {results['total_distance'] * 100:8.2f} cm (from source to cavity input)"
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
                  fr"$\quad$ beam waist: {results['w_laser_at_input'] * 1e6:7.1f} $\mu m$ <br>" +
                  fr"$\quad$ radius of curvature: {results['R_laser_at_input'] * 100:7.1f} cm <br>" +
                  "**Cavity mode**<br>" +
                  fr"$\quad$ beam waist: {results['w_cavity_at_input'] * 1e6:7.1f} $\mu m$ <br>" +
                  fr"$\quad$ radius of curvature: {results['R_cavity_at_input'] * 100:7.1f} cm <br>"
                  )
        
        waist_diff = abs(results['w_laser_at_cavity_waist'] - results['w_cavity_waist'])/results['w_cavity_waist']*100
        niceprint(f"<u> Beam Parameters at Cavity Waist </u>", 5)
        niceprint(fr"Laser beam waist at cavity waist: {results['w_laser_at_cavity_waist'] * 1e6:7.1f} $\mu m$ <br>" +
                  fr"Cavity mode waist (target): {results['w_cavity_waist'] * 1e6:7.1f} $\mu m$ <br>" +
                  f"Match quality: {waist_diff:6.2f} % difference"
                  )

class TelescopeOptimizer:
    """
    Optimize telescope designs for mode matching and fiber coupling.
    """
    
    def __init__(self, wavelength: float):
        self.wavelength = wavelength
        self.beam = GaussianBeamTool(wavelength)
    
    def telescope_abcd(self, f1: float, f2: float, separation: float) -> np.ndarray:
        """
        ABCD matrix for a telescope.
        
        Args:
            f1: First lens focal length
            f2: Second lens focal length
            separation: Lens separation
        
        Returns:
            2x2 ABCD matrix
        """
        M1 = np.array([[1, 0], [-1/f1, 1]])
        M_prop = np.array([[1, separation], [0, 1]])
        M2 = np.array([[1, 0], [-1/f2, 1]])
        return M2 @ M_prop @ M1
    
    def optimize_for_target_waist(self, 
                                  q_in: complex,
                                  w_target: float,
                                  available_lenses: List[float],
                                  max_separation: float) -> dict:
        """
        Find best telescope to achieve target waist.
        
        Uses differential evolution to search lens combinations.
        
        Args:
            q_in: Input beam parameter
            w_target: Target waist size (meters)
            available_lenses: List of available focal lengths (meters)
            max_separation: Maximum lens separation (meters)
        
        Returns:
            Dictionary with optimized parameters
        """
        def cost_function(params):
            f1_idx, f2_idx, sep = params
            f1 = available_lenses[int(f1_idx)]
            f2 = available_lenses[int(f2_idx)]
            
            M = self.telescope_abcd(f1, f2, sep)
            q_out = self.beam.propagate_q(q_in, M)
            w_out = self.beam.waist_from_q(q_out)
            
            return abs(w_out - w_target)
        
        # Bounds: (f1_index, f2_index, separation)
        bounds = [(0, len(available_lenses) - 1),
                 (0, len(available_lenses) - 1),
                 (0.01, max_separation)]
        
        result = differential_evolution(cost_function, bounds, seed=42)
        
        f1_idx = int(result.x[0])
        f2_idx = int(result.x[1])
        separation = result.x[2]
        
        f1 = available_lenses[f1_idx]
        f2 = available_lenses[f2_idx]
        M = self.telescope_abcd(f1, f2, separation)
        q_out = self.beam.propagate_q(q_in, M)
        
        return {
            'f1': f1,
            'f2': f2,
            'separation': separation,
            'magnification': -f2/f1,
            'q_out': q_out,
            'w_out': self.beam.waist_from_q(q_out),
            'error': result.fun
        }

class CavityAnalyzer:
    """
    Analyze optical cavity stability and modes.
    
    A cavity is stable when the round-trip beam reproduces itself.
    This requires: 0 < g₁*g₂ < 1
    
    where g_i = 1 - L/R_i for each mirror.
    """
    
    def __init__(self, wavelength: float):
        self.wavelength = wavelength
        self.beam = GaussianBeamTool(wavelength)
    
    def stability_parameter(self, L1: float, L2: float, 
                          R1: float, R2: float) -> float:
        """
        Calculate stability through g parameter.
        """
        g1 = 1 - L1/R1 if not np.isinf(R1) else 1
        g2 = 1 - L2/R2 if not np.isinf(R2) else 1
        return g1 * g2
    
    def cavity_mode_waist(self, L1: float, L2: float,
                         R1: float, R2: float) -> Tuple[float, float]:
        """
        Calculate cavity mode waist size and position.
        
        Returns:
            (w0, z0) - waist size and position from first mirror
        """
        g1 = 1 - L1/R1 if not np.isinf(R1) else 1
        g2 = 1 - L2/R2 if not np.isinf(R2) else 1
        
        L_total = L1 + L2
        
        # Waist size
        w0_sq = (self.wavelength * L_total / np.pi) * np.sqrt(g1 * g2 * (1 - g1*g2)) / abs(g1 + g2 - 2*g1*g2)
        w0 = np.sqrt(w0_sq)
        
        # Waist position from first mirror
        z0 = L1 * g2 * (1 - g1) / (g1 + g2 - 2*g1*g2)
        
        return w0, z0
    
    def free_spectral_range(self, L_total: float) -> float:
        """
        Calculate free spectral range.
        
        Args:
            L_total: Total round-trip cavity length (m)
        
        Returns:
            FSR in Hz
        """
        return c / (2 * L_total)
    
    def finesse(self, reflectivity: float) -> float:
        """
        Calculate cavity finesse.
        
        Args:
            reflectivity: Total power reflectivity (product of all mirrors)
        
        Returns:
            Finesse (dimensionless)
        """
        R = reflectivity
        return np.pi * np.sqrt(R) / (1 - R)
    
    def linewidth(self, FSR: float, finesse: float) -> float:
        """
        Calculate cavity linewidth (FWHM).
        
        Args:
            FSR: Free spectral range (Hz)
            finesse: Cavity finesse
        
        Returns:
            Linewidth in Hz
        """
        return FSR / finesse
    
    def quality_factor(self, frequency: float, linewidth: float) -> float:
        """
        Calculate cavity quality factor.
        
        Args:
            frequency: Resonance frequency (Hz)
            linewidth: Cavity linewidth (Hz)
        
        Returns:
            Quality factor Q (dimensionless)
        """
        return frequency / linewidth
    
    def bounce_number(self, finesse: float) -> float:
        """
        Average number of round trips photon makes in cavity.
        
        Args:
            finesse: Cavity finesse
        
        Returns:
            Average number of bounces (dimensionless)
        """
        return finesse / np.pi
    
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
        qcheck = self.beam.propagate_q(q, M)
        error = abs(qcheck - q) / abs(q)
        if error > 0.01:  # 1% relative error
            warnings.warn(f"Eigenmode verification: relative error = {error:.6f}")
        
        return q
    
    def full_analysis(self, L1: float, L2: float, R1: float, R2: float,
                     reflectivity: float) -> Dict:
        """
        Complete cavity analysis.
        
        Args:
            L1, L2: Cavity arm lengths (m)
            R1, R2: Mirror radii of curvature (m)
            reflectivity: Product of all mirror reflectivities
        
        Returns:
            Dictionary with all cavity properties
        """
        # Stability
        g_param = self.stability_parameter(L1, L2, R1, R2)
        is_stable = 0 < g_param < 1
        
        results = {
            'stable': is_stable,
            'g_parameter': g_param,
            'L_total': L1 + L2,
        }
        
        if is_stable:
            # Mode parameters
            w0, z0 = self.cavity_mode_waist(L1, L2, R1, R2)
            results['w0'] = w0
            results['z0'] = z0
            
            # Resonance properties
            L_total = L1 + L2
            FSR = self.free_spectral_range(L_total)
            F = self.finesse(reflectivity)
            delta_f = self.linewidth(FSR, F)
            freq = c / self.wavelength
            Q = self.quality_factor(freq, delta_f)
            n_bounce = self.bounce_number(F)
            
            results['FSR'] = FSR
            results['finesse'] = F
            results['linewidth'] = delta_f
            results['Q'] = Q
            results['n_bounce'] = n_bounce
        
        return results



class QuantumStateTomographer:
    """
    Full quantum state tomography simulator for single and two-photon polarization states.

    Parameters
    ----------
    n_counts : int
        Expected photon counts per measurement setting.
    angle_error_rad : float
        Max waveplate angle error (radians).  0 = ideal apparatus.
    wavelength : float
        Photon wavelength (m).  Default 810 nm (quED SPDC output).
    seed : int or None
    """

    def __init__(self, n_counts: int = 10000,
                 angle_error_rad: float = 0.0,
                 wavelength: float = 810e-9,
                 seed=None):
        self.n_counts = n_counts
        self.angle_error_rad = angle_error_rad
        self.wavelength = wavelength
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _to_rho(self, state) -> np.ndarray:
        """Return ideal density matrix as (n,n) numpy array."""
        if isinstance(state, Qobj):
            return (state * state.dag()).full() if state.type == 'ket' else state.full()
        arr = np.asarray(state, dtype=complex)
        if arr.ndim == 1:
            return np.outer(arr, arr.conj())
        return arr

    def _proj_ket(self, qwp_angle: float, hwp_angle: float) -> np.ndarray:
        """
        Build the analysis circuit and extract the effective projection ket.
        """
        from .helpers import _analysis_circuit
        circ = _analysis_circuit(qwp_angle, hwp_angle, self.wavelength)
        J    = circ.get_total_jones_matrix()
        proj = J.conj().T[:, 0]
        proj /= (np.linalg.norm(proj) + 1e-15)
        return proj

    def _noisy_qwp_hwp(self, label: str) -> Tuple[float, float]:
        """Return (QWP, HWP) angles for label, with optional random error."""
        from .helpers import _SETTINGS
        qwp, hwp = _SETTINGS[label]
        if self.angle_error_rad > 0:
            qwp += self._rng.uniform(-self.angle_error_rad, self.angle_error_rad)
            hwp += self._rng.uniform(-self.angle_error_rad, self.angle_error_rad)
        return qwp, hwp

    def run_1photon(self, state) -> dict:
        """
        Simulate single-photon tomography.

        Parameters
        ----------
        state : Qobj (ket or dm) or np.ndarray

        Returns
        -------
        dict
            'rho_ideal'   : Qobj   — ideal density matrix
            'rho_noisy'   : Qobj   — reconstructed from simulated counts
            'counts'      : ndarray shape (6,)  — [H, V, D, A, R, L]
            'stokes'      : dict   — S1, S2, S3 values
            'properties'  : dict   — trace and purity
            'eigenvalues' : ndarray
            'eigenvectors': list[Qobj]
        """
        from .helpers import _LABELS_1Q, density_matrix_1photon, _stokes_contrast, rho_eigensystem, rho_properties
        
        rho_arr   = self._to_rho(state)
        rho_ideal = Qobj(rho_arr)
        counts    = np.zeros(6, dtype=float)

        for k, label in enumerate(_LABELS_1Q):
            qwp, hwp = self._noisy_qwp_hwp(label)
            proj     = self._proj_ket(qwp, hwp)
            p        = float(np.real(proj.conj() @ rho_arr @ proj))
            counts[k]= self._rng.poisson(max(p, 0) * self.n_counts)

        rho_noisy = density_matrix_1photon(counts)
        vals, vecs = rho_eigensystem(rho_noisy)
        props      = rho_properties(rho_noisy)

        stokes = {
            'S1': _stokes_contrast(counts[2], counts[3]),   # D/A
            'S2': _stokes_contrast(counts[4], counts[5]),   # R/L
            'S3': _stokes_contrast(counts[0], counts[1]),   # H/V
        }

        return {
            'rho_ideal':    rho_ideal,
            'rho_noisy':    rho_noisy,
            'counts':       counts,
            'stokes':       stokes,
            'properties':   props,
            'eigenvalues':  vals,
            'eigenvectors': vecs,
        }

    def run_2photon(self, state_2q) -> dict:
        """
        Simulate two-photon tomography.
        
        Parameters
        ----------
        state_2q : Qobj (ket or dm), shape 4x1 or 4x4
            Two-qubit state in {HH, HV, VH, VV} ordering.

        Returns
        -------
        dict
            'rho_ideal'   : Qobj
            'rho_noisy'   : Qobj
            'counts_36'   : ndarray shape (6, 6) — coincidence matrix
            'properties'  : dict
            'eigenvalues' : ndarray
            'eigenvectors': list[Qobj]
        """
        from .helpers import _LABELS_1Q, density_matrix_2photon, _stokes_contrast, rho_eigensystem, rho_properties
        
        rho_arr   = self._to_rho(state_2q)
        rho_ideal = Qobj(rho_arr)
        counts_36 = np.zeros((6, 6), dtype=float)

        for a, la in enumerate(_LABELS_1Q):
            qwp_a, hwp_a = self._noisy_qwp_hwp(la)
            proj_a       = self._proj_ket(qwp_a, hwp_a)

            for b, lb in enumerate(_LABELS_1Q):
                qwp_b, hwp_b = self._noisy_qwp_hwp(lb)
                proj_b       = self._proj_ket(qwp_b, hwp_b)

                kab           = np.kron(proj_a, proj_b)
                p             = float(np.real(kab.conj() @ rho_arr @ kab))
                counts_36[a, b] = self._rng.poisson(max(p, 0) * self.n_counts)

        rho_noisy  = density_matrix_2photon(counts_36)
        vals, vecs = rho_eigensystem(rho_noisy)
        props      = rho_properties(rho_noisy)

        return {
            'rho_ideal':    rho_ideal,
            'rho_noisy':    rho_noisy,
            'counts_36':    counts_36,
            'properties':   props,
            'eigenvalues':  vals,
            'eigenvectors': vecs,
        }

    def print_summary(self, result: dict, label: str = ""):
        
        niceprint('---')
        niceprint(f"**Quantum State Tomography{': ' + label if label else ''}**", 3)

        props = result['properties']
        is_1q = 'counts' in result

        if is_1q:
            N = result['counts'].astype(int)
            niceprint("<u>Measurement Counts</u>", 5)
            niceprint(f"H: {N[0]:6d} &nbsp;&nbsp; V: {N[1]:6d} <br>"
                      f"D: {N[2]:6d} &nbsp;&nbsp; A: {N[3]:6d} <br>"
                      f"R: {N[4]:6d} &nbsp;&nbsp; L: {N[5]:6d}")

            S = result['stokes']
            niceprint("<u>Stokes Parameters</u>", 5)
            niceprint(f"$S_1$ (D/A): {S['S1']:+.4f} <br>"
                      f"$S_2$ (R/L): {S['S2']:+.4f} <br>"
                      f"$S_3$ (H/V): {S['S3']:+.4f}")

        niceprint("<u>Reconstructed $\\rho$</u>", 5)
        niceprint(cleandisp(result['rho_noisy'], return_str='Markdown'))

        niceprint("<u>State Properties</u>", 5)
        niceprint(f"$\\text{{Tr}}{{\\rho}}$ = {props['trace']:.6f} <br>"
                  f"Purity $\\text{{Tr}}{{\\rho^2}}$ = {props['purity']:.6f}")

        niceprint("<u>Eigenvalues</u>", 5)
        niceprint(" &nbsp;&nbsp; ".join(f"{v:.4f}" for v in result['eigenvalues']))

        niceprint('---')

class BellInequalityAnalyzer:
    """
    CHSH Bell inequality class for analysis and simulation.

    Usages (independent):

      1. Data Analysis - load BellMeasurement rows from real measurements:
          analyzer = BellInequalityAnalyzer.from_ex_data(state='triplet')
          analyzer.print_S()

      2. Simulation - Poisson-sampled coincidences scaled by SPDCSimulator:
          analyzer = BellInequalityAnalyzer(state='triplet', simulator=spdc_sim)
          result = analyzer.run_chsh()
          analyzer.print_chsh_result(result)
          analyzer.plot_correlation_sweep()

      3. Photon Statistics - use static methods to analyze g^2 from raw counts:
          BellInequalityAnalyzer.g2_unheralded(N1, N2, N12, tau, T)
          BellInequalityAnalyzer.g2_heralded(N1, N12, N13, N123)
    """

    # Optimal CHSH angles for both Bell states [radians].
    # CHSH formula S = E(a1,b1) + E(a1,b2) + E(a2,b1) - E(a2,b2)
    CHSH_ANGLES = {
        'singlet': {'alpha': (0.0, np.pi / 4), 'beta': (-np.pi / 8,  np.pi / 8)},
        'triplet': {'alpha': (0.0, np.pi / 4), 'beta': ( np.pi / 8, -np.pi / 8)},
    }

    def __init__(
        self,
        state: str = 'triplet',
        visibility: float = 1.0,
        simulator: Optional['SPDCSimulator'] = None,
        measurements: Optional[List[BellMeasurement]] = None,
    ):
        """
        Parameters
        ----------
        state        : 'singlet' for (|HH> - |VV>) / sqrt(2)
                       'triplet' for (|HH> + |VV>) / sqrt(2)
        visibility   : fringe visibility in [0, 1].
                       P_obs = V * P_QM + (1 - V) * 1/4  (uniform background)
        simulator    : SPDCSimulator — required for sample_counts / run_chsh.
                       If None, only data-analysis and static methods work.
        measurements : optional list of BellMeasurement rows for real-data mode.
        """
        if state not in ('singlet', 'triplet'):
            raise ValueError("state must be 'singlet' or 'triplet'")
        self.state      = state
        self.visibility = visibility
        self.sim        = simulator
        self.angles     = self.CHSH_ANGLES[state]
        self._lam       = simulator.lambda_spdc if simulator is not None else 810e-9

        self.measurements = measurements or []
        self._index: Dict[Tuple[float, float], BellMeasurement] = {}
        for m in self.measurements:
            self._index[(round(m.alpha, 4), round(m.beta, 4))] = m

    def add_measurement(self, m: BellMeasurement):
        """Add one BellMeasurement row to the real-data index."""
        self.measurements.append(m)
        self._index[(round(m.alpha, 4), round(m.beta, 4))] = m

    def _get(self, alpha: float, beta: float) -> BellMeasurement:
        key = (round(alpha, 4), round(beta, 4))
        if key not in self._index:
            raise KeyError(f"No measurement at (alpha={alpha} deg, beta={beta} deg)")
        return self._index[key]

    @classmethod
    def from_ex_data(cls, **kwargs) -> 'BellInequalityAnalyzer':
        """
        Return an analyzer from an example dataset.

        Acquisition time T = 15 s, coincidence window tau = 25 ns.
        All 16 combinations of Alice angles {-45, 0, 45, 90} deg and
        Bob angles {-22.5, 22.5, 67.5, 112.5} deg.
        kwargs forwarded to __init__ (e.g. state='triplet', visibility=0.95).
        """
        rows = [
            # alpha,  beta,    N_A,    N_B,     N,   N_ac
            ( -45.0, -22.5,  84525,  80356,   842,  10.0),
            ( -45.0,  22.5,  84607,  82853,   212,  10.3),
            ( -45.0,  67.5,  83874,  82179,   302,  10.1),
            ( -45.0, 112.5,  83769,  77720,   836,   9.5),
            (   0.0, -22.5,  87015,  80948,   891,  10.3),
            (   0.0,  22.5,  86674,  83187,   869,  10.6),
            (   0.0,  67.5,  87086,  81846,   173,  10.5),
            (   0.0, 112.5,  86745,  77700,   261,   9.9),
            (  45.0, -22.5,  87782,  80385,   255,  10.3),
            (  45.0,  22.5,  87932,  83265,   830,  10.7),
            (  45.0,  67.5,  87794,  81824,   814,  10.5),
            (  45.0, 112.5,  88023,  77862,   221,  10.1),
            (  90.0, -22.5,  88416,  80941,   170,  10.5),
            (  90.0,  22.5,  88285,  82924,   259,  10.7),
            (  90.0,  67.5,  88383,  81435,   969,  10.6),
            (  90.0, 112.5,  88226,  77805,   846,  10.1),
        ]
        measurements = [
            BellMeasurement(alpha=r[0], beta=r[1],
                            N_A=r[2], N_B=r[3], N=r[4], N_ac=r[5])
            for r in rows
        ]
        return cls(measurements=measurements, **kwargs)

    # ───────────────────────────────────────────────────────────────────────────
    # Data Analysis Methods
    # ───────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def compute_E(N_pp: float, N_pm: float, N_mp: float, N_mm: float
                  ) -> Tuple[float, float]:
        """
        Correlation E and its Poisson uncertainty from four net coincidence counts.

          N_pp : (alpha,    beta   )  -- Alice +1, Bob +1
          N_pm : (alpha,    beta+90)  -- Alice +1, Bob -1
          N_mp : (alpha+90, beta   )  -- Alice -1, Bob +1
          N_mm : (alpha+90, beta+90)  -- Alice -1, Bob -1

        Formula (general, external to notes):
            E = [(N_pp + N_mm) - (N_pm + N_mp)] / T
            sigma_E = sqrt((1 - E^2) / T)

        Returns
        -------
        (E, sigma_E)
        """
        T = N_pp + N_pm + N_mp + N_mm
        if T == 0:
            return 0.0, 0.0
        E       = ((N_pp + N_mm) - (N_pm + N_mp)) / T
        sigma_E = np.sqrt(max(1.0 - E**2, 0.0) / T)
        return float(E), float(sigma_E)

    def compute_S(self, state: Optional[str] = None) -> Dict:
        """
        Compute <S> and its uncertainty from loaded BellMeasurement rows.

        Each E-value draws from four angle combinations: the base pair and
        their 90-deg-shifted counterparts (all present in the 16-row dataset).

          'triplet':  <S> = E(0,-22.5) + E(0,22.5) + E(45,22.5) - E(45,-22.5)
          'singlet':  <S> = E(45,-22.5) + E(0,-22.5) + E(45,112.5) - E(0,112.5)

        Source: Entanglement notes, Eq. 25 (page 11).

        Parameters
        ----------
        state : override self.state if provided ('singlet' or 'triplet')
        """
        state = state or self.state
        if state == 'triplet':
            # each row: (alpha,beta), (alpha+90,beta+90), (alpha+90,beta), (alpha,beta+90), sign, label
            configs = [
                (( 0.0, -22.5), ( 90.0,  67.5), ( 90.0, -22.5), ( 0.0,  67.5), +1, "E(0°, -22.5°)"),
                (( 0.0,  22.5), ( 90.0, 112.5), ( 90.0,  22.5), ( 0.0, 112.5), +1, "E(0°, +22.5°)"),
                ((45.0,  22.5), (-45.0, 112.5), (-45.0,  22.5), (45.0, 112.5), +1, "E(45°, +22.5°)"),
                ((45.0, -22.5), (-45.0,  67.5), (-45.0, -22.5), (45.0,  67.5), -1, "E(45°, -22.5°)"),
            ]
        elif state == 'singlet':
            configs = [
                ((45.0, -22.5), (-45.0,  67.5), (-45.0, -22.5), (45.0,  67.5), +1, "E(45°, -22.5°)"),
                (( 0.0, -22.5), ( 90.0,  67.5), ( 90.0, -22.5), ( 0.0,  67.5), +1, "E(0°, -22.5°)"),
                ((45.0, 112.5), (-45.0,  22.5), (-45.0, 112.5), (45.0,  22.5), +1, "E(45°, 112.5°)"),
                (( 0.0, 112.5), ( 90.0,  22.5), ( 90.0, 112.5), ( 0.0,  22.5), -1, "E(0°, 112.5°)"),
            ]
        else:
            raise ValueError(f"state must be 'singlet' or 'triplet', got '{state}'")

        E_vals, sigma_E_vals, signs, labels = [], [], [], []
        for (pp, mm, mp, pm, sign, label) in configs:
            m_pp = self._get(*pp); m_mm = self._get(*mm)
            m_mp = self._get(*mp); m_pm = self._get(*pm)
            E, sE = self.compute_E(m_pp.N_net, m_pm.N_net, m_mp.N_net, m_mm.N_net)
            E_vals.append(E); sigma_E_vals.append(sE); signs.append(sign); labels.append(label)

        S       = sum(s * E for s, E in zip(signs, E_vals))
        sigma_S = np.sqrt(sum(sE**2 for sE in sigma_E_vals))
        return {'state': state, 'E_values': list(zip(labels, E_vals, sigma_E_vals, signs)),
                'S': S, 'sigma_S': sigma_S}

    def print_S(self, state: Optional[str] = None):
        
        result = self.compute_S(state)
        label  = 's' if result['state'] == 'singlet' else 't'
        niceprint(f"**CHSH Bell Inequality (data) — $|\\psi_{{{label}}}\\rangle$**", 4)
        for lbl, E, sE, sign in result['E_values']:
            niceprint(f"$\\quad {'+'if sign>0 else'-'}\\; {lbl} = {E:+.4f} \\pm {sE:.4f}$")
        S, sS  = result['S'], result['sigma_S']
        niceprint(
            f"$\\langle\\hat{{S}}\\rangle = {S:.4f} \\pm {sS:.4f}$" + (
                f" — **{(abs(S)-2)/sS:.1f}σ above LHV bound**"
                if abs(S) > 2 else " — No violation detected")
        )

    # ───────────────────────────────────────────────────────────────────────────
    # Simulation Methods
    # ───────────────────────────────────────────────────────────────────────────

    def _arm_amplitudes(self, angle: float) -> Tuple[complex, complex, complex, complex]:
        """
        Transmission (T) and reflection (R) amplitudes for |H> and |V> inputs
        through a PBS at angle [radians].

        Returns
        -------
        (t_H, t_V, r_H, r_V) with analytic values:
          t_H = cos(angle),  t_V = sin(angle)
          r_H = -sin(angle), r_V = cos(angle)
        """
        H_in = np.array([1.0, 0.0], dtype=complex)
        V_in = np.array([0.0, 1.0], dtype=complex)
        H_a  = np.array([ np.cos(angle),  np.sin(angle)], dtype=complex)
        V_a  = np.array([-np.sin(angle),  np.cos(angle)], dtype=complex)

        circ_T = OpticalCircuit(wavelength=self._lam)
        circ_T.add_polarizer(angle=angle)
        t_H = complex(H_a @ circ_T.propagate_polarization(H_in))
        t_V = complex(H_a @ circ_T.propagate_polarization(V_in))

        circ_R = OpticalCircuit(wavelength=self._lam)
        circ_R.add_polarizer(angle=angle + np.pi / 2)
        r_H = complex(V_a @ circ_R.propagate_polarization(H_in))
        r_V = complex(V_a @ circ_R.propagate_polarization(V_in))

        return t_H, t_V, r_H, r_V

    def _joint_probs(self, alpha: float, beta: float) -> Tuple[float, float, float, float]:
        """
        Compute P_HH, P_HV, P_VH, P_VV from OpticalCircuit amplitudes.
        """
        t_aH, t_aV, r_aH, r_aV = self._arm_amplitudes(alpha)
        t_bH, t_bV, r_bH, r_bV = self._arm_amplitudes(beta)
        sign = -1.0 if self.state == 'singlet' else +1.0
        s2   = np.sqrt(2.0)

        A_HH = (t_aH * t_bH + sign * t_aV * t_bV) / s2
        A_HV = (t_aH * r_bH + sign * t_aV * r_bV) / s2
        A_VH = (r_aH * t_bH + sign * r_aV * t_bV) / s2
        A_VV = (r_aH * r_bH + sign * r_aV * r_bV) / s2

        V, bg = self.visibility, 0.25
        return (
            V * abs(A_HH)**2 + (1 - V) * bg,
            V * abs(A_HV)**2 + (1 - V) * bg,
            V * abs(A_VH)**2 + (1 - V) * bg,
            V * abs(A_VV)**2 + (1 - V) * bg,
        )

    # single setting sampling of Poisson counts ─────────────────────────────────
    def sample_counts(
        self,
        alpha: float,
        beta: float,
        P_pump: Optional[float] = None,
        T_acq: float = 15.0,
    ) -> Dict:
        """
        Simulate Poisson coincidence counts at one (alpha, beta) setting.

        Parameters
        ----------
        alpha, beta : polarizer angles [radians]
        P_pump      : pump power [W] (defaults to sim.p.P_max)
        T_acq       : acquisition time per setting [s]

        Returns
        -------
        dict with N_HH, N_HV, N_VH, N_VV, N_tot, E, sigma_E
        """
        if self.sim is None:
            raise RuntimeError("sample_counts requires a SPDCSimulator. pass one to __init__")
        if P_pump is None:
            P_pump = self.sim.p.P_max

        P_HH, P_HV, P_VH, P_VV = self._joint_probs(alpha, beta)
        N_mean = float(self.sim.Rcoin_W(np.array([P_pump]))[0]) * T_acq

        N_HH = int(np.random.poisson(P_HH * N_mean))
        N_HV = int(np.random.poisson(P_HV * N_mean))
        N_VH = int(np.random.poisson(P_VH * N_mean))
        N_VV = int(np.random.poisson(P_VV * N_mean))
        N_tot = N_HH + N_HV + N_VH + N_VV

        if N_tot == 0:
            return dict(N_HH=0, N_HV=0, N_VH=0, N_VV=0,
                        N_tot=0, E=0.0, sigma_E=np.inf)
        E       = (N_HH - N_HV - N_VH + N_VV) / N_tot
        sigma_E = 1.0 / np.sqrt(N_tot)
        return dict(N_HH=N_HH, N_HV=N_HV, N_VH=N_VH, N_VV=N_VV,
                    N_tot=N_tot, E=E, sigma_E=sigma_E)

    # full CHSH run with Poisson sampling at each of the 4 settings ─────────────
    def run_chsh(self, P_pump: Optional[float] = None, T_acq: float = 15.0) -> Dict:
        """
        Simulate the full 4-setting CHSH experiment.

        CHSH operator (Entanglement notes, Eq. 25):
            S = E(a1,b1) + E(a1,b2) + E(a2,b1) - E(a2,b2)

        Uncertainty (Poisson, settings independent):
            sigma_S = sqrt(sum_i sigma_Ei^2)

        Parameters
        ----------
        P_pump : pump power [W] (defaults to sim.p.P_max)
        T_acq  : acquisition time per setting [s]

        Returns
        -------
        dict with 'settings', 'S', 'sigma_S', 'S_theory', 'violates'
        """
        if self.sim is None:
            raise RuntimeError("run_chsh requires a SPDCSimulator — pass one to __init__")
        if P_pump is None:
            P_pump = self.sim.p.P_max

        a1, a2 = self.angles['alpha']
        b1, b2 = self.angles['beta']
        chsh_configs = [(a1, b1, +1.0), (a1, b2, +1.0), (a2, b1, +1.0), (a2, b2, -1.0)]

        results, S, var_S = {}, 0.0, 0.0
        for alpha, beta, sign in chsh_configs:
            key = f'a={np.rad2deg(alpha):.1f}°, b={np.rad2deg(beta):.1f}°'
            r   = self.sample_counts(alpha, beta, P_pump, T_acq)
            results[key] = r
            S    += sign * r['E']
            var_S += r['sigma_E']**2

        return {'settings': results, 'S': S, 'sigma_S': np.sqrt(var_S),
                'S_theory': self.visibility * 2.0 * np.sqrt(2.0),
                'violates': abs(S) > 2.0}

    def print_summary(self, result: Dict):
        
        label = 's' if self.state == 'singlet' else 't'
        niceprint(
            f"**CHSH Result (simulation) — $|\\psi_{{{label}}}\\rangle$,"
            f"  $V = {self.visibility:.2f}$**", 4
        )
        
        label = "**Yes** <span style=\"color:green\">&#x2713;</span>" if result["violates"] else "**No** <span style=\"color:red\">&#x2717;</span>"
        niceprint(
            f"$\\langle S \\rangle = {result['S']:.4f} \\pm {result['sigma_S']:.4f}$  "
            f"QM prediction: $2\\sqrt{{2}}\\cdot V = {result['S_theory']:.4f}$  "
            f"Violates Bell: {label}"
        )

    def plot_correlation_sweep(
        self,
        fixed_alpha: float = 0.0,
        n_points: int = 73,
        P_pump: Optional[float] = None,
        T_acq: float = 15.0,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot E(alpha, beta) vs. beta with alpha fixed.

        Parameters
        ----------
        fixed_alpha : Alice's fixed angle [radians]
        n_points    : points for the smooth theory curve
        P_pump      : pump power [W] (ignored if sim=None)
        T_acq       : acquisition time per data point [s]
        """
        betas_theory = np.linspace(0, 2 * np.pi, n_points)
        E_theory = np.array([
            np.dot(self._joint_probs(fixed_alpha, b), [1, -1, -1, 1])
            for b in betas_theory
        ])

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.plot(np.rad2deg(betas_theory), E_theory,
                color='steelblue', linewidth=2, label='Theory', zorder=2)

        if self.sim is not None:
            if P_pump is None:
                P_pump = self.sim.p.P_max
            betas_data = np.deg2rad(np.arange(0, 361, 5))
            E_data, E_err = [], []
            for b in betas_data:
                r = self.sample_counts(fixed_alpha, float(b), P_pump, T_acq)
                E_data.append(r['E']); E_err.append(r['sigma_E'])
            ax.errorbar(np.rad2deg(betas_data), E_data, yerr=E_err,
                        fmt='o', color='crimson', markersize=4,
                        elinewidth=0.8, capsize=2, label='Simulated data', zorder=3)

        b1, b2 = self.angles['beta']
        for b, lbl in [(b1, r'$\beta_1$'), (b2, r'$\beta_2$')]:
            ax.axvline(np.rad2deg(b % (2 * np.pi)),
                       color='gray', linestyle='--', linewidth=1.2, label=lbl)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlabel(r'$\beta$ (degrees)')
        ax.set_ylabel(r'$E(\alpha,\,\beta)$')
        label = 's' if self.state == 'singlet' else 't'
        ax.set_title(
            f'Correlation sweep: $|\\psi_{{{label}}}\\rangle$,  '
            f'$\\alpha = {np.rad2deg(fixed_alpha):.1f}^\\circ$,  '
            f'$V = {self.visibility:.2f}$'
        )
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        return ax

    def plot_all_sweeps(
        self,
        P_pump: Optional[float] = None,
        T_acq: float = 15.0,
    ) -> plt.Figure:
        """
        Four-panel figure: one panel per canonical Alice setting {H, V, P, M},
        each showing the full correlation sweep vs. Bob's angle.
        """
        alice_settings = [0.0, np.pi / 2, np.pi / 4, -np.pi / 4]
        alice_labels   = ['H ($0^\\circ$)', 'V ($90^\\circ$)', 'P ($45^\\circ$)', 'M ($-45^\\circ$)']
        label = 's' if self.state == 'singlet' else 't'

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        fig.suptitle(
            f'Correlation Sweeps: $|\\psi_{{{label}}}\\rangle$,  $V = {self.visibility:.2f}$',
            fontsize=13, fontweight='bold'
        )
        for ax, alpha, lbl in zip(axes.flat, alice_settings, alice_labels):
            self.plot_correlation_sweep(alpha, P_pump=P_pump, T_acq=T_acq, ax=ax)
            ax.set_title(f'Alice fixed at {lbl}', fontsize=11)
        plt.tight_layout()
        return fig

    # ───────────────────────────────────────────────────────────────────────────
    # Photon Statistics Methods (g^(2) analysis)
    # ───────────────────────────────────────────────────────────────────────────

    @staticmethod
    def g2_unheralded(
        N1: float, N2: float, N12: float, tau: float, T: float,
    ) -> Tuple[float, float]:
        """
        Unheralded g^(2)(0)

        Parameters
        ----------
        N1, N2  : singles on detectors 1 and 2 (over run length T)
        N12     : coincidences at tau = 0
        tau     : coincidence window [s]
        T       : run length [s]
        """
        N_ac = N1 * N2 * tau / T
        if N_ac == 0:
            return np.inf, np.inf
        g2       = N12 / N_ac
        sigma_g2 = np.sqrt(N12) / N_ac
        return g2, sigma_g2

    @staticmethod
    def g2_heralded(
        N1: float, N12: float, N13: float, N123: float,
    ) -> Tuple[float, float]:
        """
        Heralded g^(2)(0) from a three-fold coincidence measurement.

        Parameters
        ----------
        N1       : herald singles
        N12, N13 : two-fold coincidences (herald-arm2, herald-arm3)
        N123     : three-fold coincidences
        """
        denom = N12 * N13
        if denom == 0:
            return np.inf, np.inf
        g2_H       = N123 * N1 / denom
        sigma_g2_H = g2_H * np.sqrt(1 / max(N123, 1) + 1 / N12 + 1 / N13 + 1 / N1)
        return g2_H, sigma_g2_H

    def print_g2(
        self,
        N1: float, N2: float, N12: float, tau: float, T: float,
        N13: Optional[float] = None, N123: Optional[float] = None,
    ):
        # unheralded
        g2, sg2 = self.g2_unheralded(N1, N2, N12, tau, T)
        niceprint("**Photon Statistics** — $g^{(2)}(0)$", 4)
        niceprint(f"Unheralded: $g^{{(2)}}(0) = {g2:.4f} \\pm {sg2:.4f}$"
                  + ("**non-classical:** ($< 0.5$)" if g2 < 0.5 else ""))
        
        # heralded (optional)
        if N13 is not None and N123 is not None:
            g2_H, sg2_H = self.g2_heralded(N1, N12, N13, N123)
            niceprint(f"Heralded: $g^{{(2)}}_H(0) = {g2_H:.4f} \\pm {sg2_H:.4f}$"
                      + ("**single photon confirmed:** ($< 0.5$)" if g2_H < 0.5 else ""))




