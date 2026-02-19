from .general import *

import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from scipy.signal import sawtooth
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Union, Literal, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from qutip import Qobj, basis, tensor, fock



c = 299792458  # Speed of light (m/s)


# =============================================================================
# SECTION 1: BASE CLASSES
# =============================================================================

class OpticalComponent(ABC):
    """
    Base class for all optical components.
    
    Each component must implement:
    - get_abcd_matrix() for classical beam propagation
    - get_jones_matrix() for polarization (if applicable)
    - apply_quantum() for quantum state evolution (if applicable)
    """
    
    def __init__(self, name: str = "Component"):
        self.name = name
    
    @abstractmethod
    def get_abcd_matrix(self) -> np.ndarray:
        """Return 2x2 ABCD matrix for ray transfer analysis."""
        pass
    
    def get_jones_matrix(self) -> Optional[np.ndarray]:
        """Return 2x2 Jones matrix for polarization (None if not applicable)."""
        return None
    
    def apply_quantum(self, state):
        """Apply quantum operation to state (default: identity)."""
        return state
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class GaussianBeamTool:
    """
    Tools for Gaussian beam calculations using ABCD matrices.
    
    The complex beam parameter q encodes both beam size and wavefront curvature:
        1/q = 1/R - i*λ/(π*w²)
    
    where:
        R = radius of curvature of wavefront
        w = beam width (1/e² intensity radius)
        λ = wavelength
    
    At a beam waist (R → ∞):
        q = i*z_R  where z_R = π*w₀²/λ (Rayleigh range)
    
    Source: ABCD Matrices lecture notes, Eq. 1-3 (pages 1-2)
    """
    
    def __init__(self, wavelength: float):
        """
        Args:
            wavelength: Wavelength in meters
        """
        self.lam = wavelength
        self.k = 2 * np.pi / wavelength  # Wave number
    
    def q_from_waist(self, w0: float, z: float = 0, z0: float = 0) -> complex:
        """
        Calculate q-parameter from beam waist and position.
        
        Math:
            z_R = π*w₀²/λ
            q(z) = (z - z₀) + i*z_R
        
        Args:
            w0: Beam waist size (meters)
            z: Position (meters)
            z0: Waist location (meters, default 0)
        
        Returns:
            Complex beam parameter q
        
        Source: ABCD Matrices notes, Eq. 2 (page 1)
        """
        z_R = np.pi * w0**2 / self.lam
        return (z - z0) + 1j * z_R
    
    def waist_from_q(self, q: complex) -> float:
        """
        Extract beam waist from q-parameter.
        
        Math:
            w = √(λ*Im(q)/π)
        
        Source: ABCD Matrices notes, Eq. 3 (page 2)
        """
        if q.imag <= 0:
            raise ValueError(f"Invalid q: Im(q) = {q.imag} must be positive")
        return np.sqrt(self.lam * abs(q.imag) / np.pi)
    
    def R_from_q(self, q: complex) -> float:
        """
        Extract wavefront radius of curvature from q.
        
        Math:
            R = Re(q) * [1 + (Im(q)/Re(q))²]
        
        Returns:
            Radius of curvature (infinite at waist)
        """
        if abs(q.real) < 1e-10:
            return np.inf
        return q.real * (1 + (q.imag / q.real)**2)
    
    def propagate_q(self, q: complex, M: np.ndarray) -> complex:
        """
        Propagate q through ABCD matrix.
        
        Math:
            q' = (A*q + B) / (C*q + D)
        
        Args:
            q: Input beam parameter
            M: 2x2 ABCD matrix
        
        Returns:
            Output beam parameter
        
        Source: ABCD Matrices notes, Eq. 4 (page 2)
        """
        A, B = M[0, 0], M[0, 1]
        C, D = M[1, 0], M[1, 1]
        return (A * q + B) / (C * q + D)
    
    def rayleigh_range(self, w0: float) -> float:
        """
        Calculate Rayleigh range.
        
        Math: z_R = π*w₀²/λ
        
        Source: ABCD Matrices notes, page 1
        """
        return np.pi * w0**2 / self.lam
    
    def divergence_angle(self, w0: float) -> float:
        """
        Far-field divergence half-angle.
        
        Math: θ = λ/(π*w₀)
        
        Source: ABCD Matrices notes, page 1
        """
        return self.lam / (np.pi * w0)

class QuantumState:
    """
    Represents a quantum state for interferometry and quantum optics.
    
    Math: A quantum state is represented as |ψ⟩ = Σᵢ cᵢ|i⟩
    where |i⟩ are basis states (e.g., different optical paths)
    and cᵢ are complex probability amplitudes with Σᵢ|cᵢ|² = 1.
    
    Example:
    --------
    # Photon in path 0
    state = QuantumState(0, num_paths=2)
    
    # Apply beam splitter
    bs = BeamSplitter()
    new_state = state.apply_operator(bs.get_quantum_operator())
    new_state.measure(0)  # Probability in path 0
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
    # Create superposition
    state = QuantumState(0, num_paths=2)
    bs = BeamSplitter()
    state = state.apply_operator(bs.get_quantum_operator())
    
    # Add π/2 phase shift to path 1
    ps = PhaseShifter(path_index=1, phase=np.pi/2)
    state = state.apply_operator(ps.get_quantum_operator())
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
    # Constructive interference in output 0
    mz = MachZehnderInterferometer(delta_L=0, wavelength=632.8e-9)
    P0, P1 = mz.get_output_probabilities()
    print(f"P0 = {P0:.3f}, P1 = {P1:.3f}")
    P0 = 1.000, P1 = 0.000
    
    # Destructive interference (π phase shift)
    mz = MachZehnderInterferometer(delta_L=316.4e-9, wavelength=632.8e-9)
    P0, P1 = mz.get_output_probabilities()
    print(f"P0 = {P0:.3f}, P1 = {P1:.3f}")
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


# =============================================================================
# SECTION 2: CLASSICAL OPTICAL COMPONENTS (ABCD Matrices)
# =============================================================================

class FreeSpace(OpticalComponent):
    """
    Free space propagation.
    
    ABCD Matrix:
        M = [[1, d],
             [0, 1]]
    
    Source: ABCD Matrices notes, Eq. 5 (page 2)
    """
    
    def __init__(self, distance: float, name: str = "Free Space"):
        super().__init__(name)
        self.d = distance
    
    def get_abcd_matrix(self) -> np.ndarray:
        return np.array([[1, self.d],
                        [0, 1]], dtype=float)
    
    def __repr__(self):
        return f"FreeSpace(d={self.d:.4f} m)"


class ThinLens(OpticalComponent):
    """
    Thin lens (converging if f > 0, diverging if f < 0).
    
    ABCD Matrix:
        M = [[1,    0],
             [-1/f, 1]]
    
    Source: ABCD Matrices notes, Eq. 6 (page 3)
    """
    
    def __init__(self, focal_length: float, name: str = "Thin Lens"):
        super().__init__(name)
        self.f = focal_length
    
    def get_abcd_matrix(self) -> np.ndarray:
        return np.array([[1, 0],
                        [-1/self.f, 1]], dtype=float)
    
    def __repr__(self):
        return f"ThinLens(f={self.f:.4f} m)"

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


class CurvedMirror(OpticalComponent):
    """
    Curved mirror with radius of curvature R.
    
    ABCD Matrix:
        M = [[1,    0],
             [-2/R, 1]]
    
    Note: For a mirror, the effective focal length is f = R/2,
    so the power is -2/R (negative for concave mirror).
    
    Source: ABCD Matrices notes, Eq. 7 (page 3)
    """
    
    def __init__(self, radius: float, name: str = "Curved Mirror"):
        super().__init__(name)
        self.R = radius
    
    def get_abcd_matrix(self) -> np.ndarray:
        return np.array([[1, 0],
                        [-2/self.R, 1]], dtype=float)
    
    def __repr__(self):
        return f"CurvedMirror(R={self.R:.4f} m)"


class FlatMirror(OpticalComponent):
    """
    Flat mirror (infinite radius of curvature).
    
    ABCD Matrix:
        M = [[1, 0],
             [0, 1]]
    
    Source: General optics formula
    """
    
    def __init__(self, name: str = "Flat Mirror"):
        super().__init__(name)
    
    def get_abcd_matrix(self) -> np.ndarray:
        return np.array([[1, 0],
                        [0, 1]], dtype=float)
    
    def __repr__(self):
        return f"FlatMirror()"



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
    state_in = QuantumState(0, num_paths=2)  # Photon in path 0
    bs = BeamSplitter()
    state_out = state_in.apply_operator(bs.get_quantum_operator())
    state_out.measure(0), state_out.measure(1)
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


class DielectricInterface(OpticalComponent):
    """
    Refraction at a planar dielectric interface.
    
    ABCD Matrix:
        M = [[1,   0],
             [0, n₁/n₂]]
    
    Source: General optics formula
    """
    
    def __init__(self, n1: float, n2: float, name: str = "Interface"):
        super().__init__(name)
        self.n1 = n1
        self.n2 = n2
    
    def get_abcd_matrix(self) -> np.ndarray:
        return np.array([[1, 0],
                        [0, self.n1/self.n2]], dtype=float)
    
    def __repr__(self):
        return f"Interface(n₁={self.n1:.3f}, n₂={self.n2:.3f})"


# =============================================================================
# SECTION 3: POLARIZATION COMPONENTS (Jones Matrices)
# =============================================================================

class PolarizationComponent(OpticalComponent):
    """Base class for components that affect polarization."""
    
    def __init__(self, name: str = "Polarization Component"):
        super().__init__(name)
    
    def get_abcd_matrix(self) -> np.ndarray:
        """Polarization optics don't affect beam propagation."""
        return np.eye(2)
    
    @staticmethod
    def rotation_matrix(angle: float) -> np.ndarray:
        """
        2D rotation matrix.
        
        Math:
            R(θ) = [[cos(θ), -sin(θ)],
                    [sin(θ),  cos(θ)]]
        
        Source: Polarization notes, Eq. 11 (page 5)
        """
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s],
                        [s,  c]])


class Polarizer(PolarizationComponent):
    """
    Linear polarizer at angle θ from horizontal.
    
    Jones Matrix (horizontal polarizer):
        T_H = [[1, 0],
               [0, 0]]
    
    Rotated by angle θ:
        T(θ) = R(θ) · T_H · R(-θ)
    
    where R(θ) is the rotation matrix.
    
    Source: Polarization notes, Eq. 8, 12 (pages 4-5)
    """
    
    def __init__(self, angle: float = 0.0, name: str = "Polarizer"):
        super().__init__(name)
        self.angle = angle
        
        # Base horizontal polarizer
        T_H = np.array([[1.0, 0.0],
                       [0.0, 0.0]], dtype=complex)
        
        # Rotate if needed
        if angle != 0:
            R = self.rotation_matrix(angle)
            self.jones_matrix = R @ T_H @ R.T
        else:
            self.jones_matrix = T_H
    
    def get_jones_matrix(self) -> np.ndarray:
        return self.jones_matrix
    
    def __repr__(self):
        angle_deg = np.rad2deg(self.angle)
        return f"Polarizer(θ={angle_deg:.1f}°)"

class RetroReflectiveMirror(PolarizationComponent):
    """
    Retroreflective mirror for polarization analysis.
    
    Reflects light back and performs complex conjugation on Jones vector.
    This flips the handedness of circular polarization.
    """
    
    def __init__(self, name: str = "Retroreflective Mirror"):
        super().__init__(name)
    
    def get_abcd_matrix(self) -> np.ndarray:
        return np.eye(2)
    
    def get_jones_matrix(self) -> Optional[np.ndarray]:
        return None
    
    def apply_reflection(self, jones_vector: np.ndarray) -> np.ndarray:
        """Apply mirror reflection = complex conjugate of Jones vector."""
        return np.conj(jones_vector)
    
    def __repr__(self):
        return "RetroReflectiveMirror()"

class HalfWavePlate(PolarizationComponent):
    """
    Half-wave plate (HWP) with fast axis at angle θ.
    
    A HWP introduces a phase shift of π between fast and slow axes.
    It rotates linear polarization by 2θ.
    
    Jones Matrix (fast axis horizontal):
        T_HWP = [[1,  0],
                 [0, -1]]
    
    Rotated by angle θ:
        T(θ) = R(θ) · T_HWP · R(-θ)
    
    Source: Polarization notes, Eq. 37 (page 15)
    """
    
    def __init__(self, fast_axis_angle: float = 0.0, name: str = "HWP", retardance_deviation: float = 0.0):
        super().__init__(name)
        self.angle = fast_axis_angle
        self.retardance_deviation = retardance_deviation
        self.retardance_phase = np.exp(1j * retardance_deviation)
        
        # Base matrix (fast axis horizontal)
        T_base = np.array([[1.0,  0.0],
                          [0.0, -1.0 * self.retardance_phase]], dtype=complex)
        
        # Rotate if needed
        if fast_axis_angle != 0:
            R = self.rotation_matrix(fast_axis_angle)
            self.jones_matrix = R @ T_base @ R.T
        else:
            self.jones_matrix = T_base
    
    def get_jones_matrix(self) -> np.ndarray:
        return self.jones_matrix
    
    def __repr__(self):
        angle_deg = np.rad2deg(self.angle)
        return f"HWP(θ={angle_deg:.1f}°, δ={self.retardance_deviation:.2f} rad)"


class QuarterWavePlate(PolarizationComponent):
    """
    Quarter-wave plate (QWP) with fast axis at angle θ.
    
    A QWP introduces a phase shift of π/2 between fast and slow axes.
    At θ = 45°, it converts linear → circular polarization.
    
    Jones Matrix (fast axis horizontal):
        T_QWP = [[1,  0],
                 [0, -i]]
    
    Rotated by angle θ:
        T(θ) = R(θ) · T_QWP · R(-θ)
    
    Source: Polarization notes, Eq. 42 (page 17)
    """
    
    def __init__(self, fast_axis_angle: float = 0.0, name: str = "QWP", retardance_deviation: float = 0.0):
        super().__init__(name)
        self.angle = fast_axis_angle
        self.retardance_deviation = retardance_deviation
        self.retardance_phase = np.exp(1j * retardance_deviation)
        
        # Base matrix (fast axis horizontal)
        T_base = np.array([[1.0,   0.0],
                          [0.0, -1.0j * self.retardance_phase]], dtype=complex)
        
        # Rotate if needed
        if fast_axis_angle != 0:
            R = self.rotation_matrix(fast_axis_angle)
            self.jones_matrix = R @ T_base @ R.T
        else:
            self.jones_matrix = T_base
    
    def get_jones_matrix(self) -> np.ndarray:
        return self.jones_matrix
    
    def __repr__(self):
        angle_deg = np.rad2deg(self.angle)
        return f"QWP(θ={angle_deg:.1f}°, δ={self.retardance_deviation:.2f} rad)"


class PolarizingBeamSplitter(PolarizationComponent):
    """
    Polarizing Beam Splitter compatible with your OpticalCircuit framework.
    
    Jones Matrix for transmitted (H) port: [[1, 0], [0, 0]]
    Jones Matrix for reflected (V) port: [[0, 0], [0, 1]]
    
    Source: Polarization Notes, Section 1.2 (pages 3-4)
    """
    
    def __init__(self, port: str = 'transmitted', name: str = "PBS"):
        super().__init__(name)
        self.port = port.lower()
        
        # Define Jones matrices for each port
        if self.port in ['transmitted', 'horizontal', 'h', 't']:
            self.jones_matrix = np.array([[1.0, 0.0],
                                          [0.0, 0.0]], dtype=complex)
        elif self.port in ['reflected', 'vertical', 'v', 'r']:
            self.jones_matrix = np.array([[0.0, 0.0],
                                          [0.0, 1.0]], dtype=complex)
        else:
            raise ValueError(f"Invalid port '{port}'. Use 'transmitted' or 'reflected'.")
    
    def get_jones_matrix(self) -> np.ndarray:
        return self.jones_matrix
    
    def __repr__(self):
        return f"PBS(port='{self.port}')"



# =============================================================================
# SECTION 4: OPTICAL CIRCUIT BUILDER
# =============================================================================

class OpticalCircuit:
    """
    Build and analyze custom optical circuits.
    
    This class allows you to chain optical components and perform
    calculations for Gaussian beam propagation and polarization.
    
    Example:
        circuit = OpticalCircuit(wavelength=632.8e-9)
        circuit.add_free_space(0.1)  # 10 cm
        circuit.add_thin_lens(0.05)  # 5 cm focal length
        circuit.add_free_space(0.15)
        
        # Propagate a beam
        q_in = circuit.beam.q_from_waist(w0=1e-3, z=0)
        q_out = circuit.propagate_gaussian(q_in)
        w_out = circuit.beam.waist_from_q(q_out)
    """
    
    def __init__(self, wavelength: float, name: str = "Optical Circuit"):
        """
        Args:
            wavelength: Wavelength in meters
            name: Circuit name for identification
        """
        self.wavelength = wavelength
        self.name = name
        self.components: List[OpticalComponent] = []
        self.beam = GaussianBeamTool(wavelength)
        self.intensity_in = 0.0
        self.intensity_out = 0.0
        self.transmission = 0.0
    
    # -------------------------------------------------------------------------
    # Component Addition Methods
    # -------------------------------------------------------------------------
    
    def add_component(self, component: OpticalComponent):
        """Add a custom component to the circuit."""
        self.components.append(component)
        return self
    
    def add_free_space(self, distance: float):
        """Add free space propagation (distance in meters)."""
        self.components.append(FreeSpace(distance))
        return self
    
    def add_thin_lens(self, focal_length: float):
        """Add a thin lens (focal length in meters)."""
        self.components.append(ThinLens(focal_length))
        return self
    
    def add_curved_mirror(self, radius: float):
        """Add a curved mirror (radius of curvature in meters)."""
        self.components.append(CurvedMirror(radius))
        return self
    
    def add_flat_mirror(self):
        """Add a flat mirror."""
        self.components.append(FlatMirror())
        return self
    
    def add_retroreflective_mirror(self):
        """Add a retroreflective mirror to the circuit."""
        self.components.append(RetroReflectiveMirror())
        return self
    
    def add_polarizer(self, angle: float = 0.0):
        """
        Add a polarizer (angle in radians from horizontal).
        
        Horizontal polarizer: angle = 0
        
        Vertical polarizer: angle = pi/2
        """
        self.components.append(Polarizer(angle))
        return self
    
    def add_hwp(self, fast_axis_angle: float = 0.0, retardance_deviation: float = 0.0):
        """Add a half-wave plate (fast axis angle in radians)."""
        self.components.append(HalfWavePlate(fast_axis_angle, retardance_deviation=retardance_deviation))
        return self
    
    def add_qwp(self, fast_axis_angle: float = 0.0, retardance_deviation: float = 0.0):
        """Add a quarter-wave plate (fast axis angle in radians)."""
        self.components.append(QuarterWavePlate(fast_axis_angle, retardance_deviation=retardance_deviation))
        return self
    
    def add_pbs(self, port: str = 'transmitted'):
        """Add a polarizing beam splitter."""
        self.components.append(PolarizingBeamSplitter(port=port))
        return self
    
    # -------------------------------------------------------------------------
    # Analysis Methods
    # -------------------------------------------------------------------------
    
    def get_total_abcd_matrix(self) -> np.ndarray:
        """
        Calculate total ABCD matrix for the circuit.
        
        Matrices are composed right-to-left (beam travels through
        components in the order they were added).
        
        Math:
            M_total = M_N · M_{N-1} · ... · M_2 · M_1
        
        Returns:
            2x2 ABCD matrix for the entire circuit
        """
        M_total = np.eye(2)
        for component in reversed(self.components):
            M_total = component.get_abcd_matrix() @ M_total
        return M_total
    
    def get_total_jones_matrix(self) -> np.ndarray:
        """
        Calculate total Jones matrix for polarization components.
        
        Returns:
            2x2 Jones matrix (or identity if no polarization components)
        """
        J_total = np.eye(2, dtype=complex)
        for component in self.components:
            J = component.get_jones_matrix()
            if J is not None:
                J_total = J @ J_total
        return J_total
    
    def propagate_gaussian(self, q_in: complex) -> complex:
        """
        Propagate a Gaussian beam through the circuit.
        
        Args:
            q_in: Input beam parameter
        
        Returns:
            Output beam parameter
        """
        M = self.get_total_abcd_matrix()
        return self.beam.propagate_q(q_in, M)
    
    def propagate_polarization(self, state_in: np.ndarray, 
                              return_all_states: bool = False):
        """
        Propagate a polarization state through the circuit.
        
        Automatically handles retroreflective mirrors:
        - Components before mirror: forward propagation
        - Mirror: complex conjugation
        - Components before mirror: backward propagation (SKIPPING POLARIZERS)
        - Components after mirror: forward propagation
        
        Args:
            state_in: Input Jones vector (2D complex array)
            return_all_states: If True, return dict with intermediate states
        
        Returns:
            Output Jones vector (or dict if return_all_states=True)
        """
        # Check for retroreflective mirror
        mirror_indices = [i for i, comp in enumerate(self.components) 
                         if isinstance(comp, RetroReflectiveMirror)]
        
        if not mirror_indices:
            # === NO MIRROR: Normal forward propagation ===
            J = self.get_total_jones_matrix()
            state_out = J @ state_in
            
            self.intensity_in = np.abs(state_in[0])**2 + np.abs(state_in[1])**2
            self.intensity_out = np.abs(state_out[0])**2 + np.abs(state_out[1])**2
            self.transmission = self.intensity_out / self.intensity_in if self.intensity_in > 0 else 0
            
            return state_out
        
        # === MIRROR PRESENT: Automatic forward→mirror→backward handling ===
        if len(mirror_indices) > 1:
            raise ValueError("Circuit can only contain one RetroReflectiveMirror")
        
        mirror_idx = mirror_indices[0]
        components_before = self.components[:mirror_idx]
        mirror = self.components[mirror_idx]
        components_after = self.components[mirror_idx + 1:]
        
        # Forward path (up to mirror)
        state = state_in.copy()
        for component in components_before:
            J = component.get_jones_matrix()
            if J is not None:
                state = J @ state
        state_before_mirror = state.copy()
        
        # Mirror reflection (complex conjugation)
        state = mirror.apply_reflection(state)
        state_after_mirror = state.copy()
        
        # Backward path (inverse matrices, reversed order)
        # CRITICAL: Skip polarizers (they're singular/absorptive)
        for component in reversed(components_before):
            J = component.get_jones_matrix()
            if J is not None:
                # Skip polarizers - they're absorptive (non-invertible)
                if isinstance(component, Polarizer):
                    continue
                
                # For waveplates and other unitary elements, use Hermitian conjugate
                is_unitary = np.allclose(J.conj().T @ J, np.eye(2))
                if is_unitary:
                    J_inv = J.conj().T
                else:
                    # Check if singular before inverting
                    det = np.linalg.det(J)
                    if abs(det) < 1e-10:
                        continue  # Skip singular matrices
                    J_inv = np.linalg.inv(J)
                
                state = J_inv @ state
        state_after_backward = state.copy()
        
        # Components after mirror (e.g., return polarizer)
        for component in components_after:
            J = component.get_jones_matrix()
            if J is not None:
                state = J @ state
        state_final = state.copy()
        
        # Calculate transmission
        self.intensity_in = np.abs(state_in[0])**2 + np.abs(state_in[1])**2
        self.intensity_out = np.abs(state_final[0])**2 + np.abs(state_final[1])**2
        self.transmission = self.intensity_out / self.intensity_in if self.intensity_in > 0 else 0
        
        return state_final
    
    def analyze_beam_propagation(self, q_in: complex, 
                                 num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate beam width evolution through the circuit.
        
        Args:
            q_in: Input beam parameter
            num_points: Number of points to sample
        
        Returns:
            (positions, beam_widths) arrays
        """
        positions = []
        widths = []
        
        z_current = 0
        q_current = q_in
        
        for component in self.components:
            if isinstance(component, FreeSpace):
                # Sample through free space
                z_local = np.linspace(0, component.d, num_points // len(self.components))
                for z in z_local:
                    M_local = FreeSpace(z).get_abcd_matrix()
                    q_local = self.beam.propagate_q(q_current, M_local)
                    positions.append(z_current + z)
                    widths.append(self.beam.waist_from_q(q_local))
                
                z_current += component.d
                q_current = self.beam.propagate_q(q_current, component.get_abcd_matrix())
            else:
                # Just record the position at the component
                q_current = self.beam.propagate_q(q_current, component.get_abcd_matrix())
                positions.append(z_current)
                widths.append(self.beam.waist_from_q(q_current))
        
        return np.array(positions), np.array(widths)
    
    def clear(self):
        """Remove all components from the circuit."""
        self.components = []
        return self
    
    def __repr__(self):
        return f"OpticalCircuit(λ={self.wavelength*1e9:.1f} nm, {len(self.components)} components)"
    
    def __str__(self):
        """Pretty print the circuit."""
        lines = [f"\n{self.name}"]
        lines.append("=" * 60)
        lines.append(f"Wavelength: {self.wavelength*1e9:.1f} nm")
        lines.append(f"Components ({len(self.components)}):")
        for i, component in enumerate(self.components, 1):
            lines.append(f"  {i}. {component}")
        return "\n".join(lines)


# =============================================================================
# SECTION 5: SPECIALIZED CALCULATORS
# =============================================================================

class TelescopeOptimizer:
    """
    Optimize telescope designs for mode matching and fiber coupling.
    
    A telescope transforms beam size and position using two lenses:
        Magnification: M = -f₂/f₁
        Effective focal length: f_eff = f₁*f₂/(f₁ + f₂ - d)
    
    Source: ABCD Matrices notes, Fiber Optics notes
    """
    
    def __init__(self, wavelength: float):
        self.wavelength = wavelength
        self.beam = GaussianBeamTool(wavelength)
    
    def telescope_abcd(self, f1: float, f2: float, separation: float) -> np.ndarray:
        """
        ABCD matrix for a telescope.
        
        Math:
            M = M_lens2 · M_space · M_lens1
            M = [[1, 0], [-1/f₂, 1]] · [[1, d], [0, 1]] · [[1, 0], [-1/f₁, 1]]
        
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


class FiberCoupler:
    """
    Calculate fiber coupling efficiency.
    
    Coupling efficiency depends on mode overlap between beam and fiber mode:
        η = |∫ψ_beam* ψ_fiber dA|² / (∫|ψ_beam|² dA · ∫|ψ_fiber|² dA)
    
    For Gaussian beams with matched parameters, this simplifies to:
        η = 4/[(w_beam/w_fiber + w_fiber/w_beam)² + (λ*R_beam/(π*w_beam*w_fiber))²]
    
    Source: Fiber Optics notes, pages 5-7
    """
    
    def __init__(self, wavelength: float):
        self.wavelength = wavelength
        self.beam = GaussianBeamTool(wavelength)
    
    def coupling_efficiency(self, w_beam: float, R_beam: float,
                          w_fiber: float) -> float:
        """
        Calculate coupling efficiency for mode matching.
        
        Math:
            η = 4 / [(w₁/w₂ + w₂/w₁)² + (λ*R/(π*w₁*w₂))²]
        
        Args:
            w_beam: Beam waist at fiber (meters)
            R_beam: Beam radius of curvature at fiber (meters, use inf for flat)
            w_fiber: Fiber mode field diameter / 2 (meters)
        
        Returns:
            Coupling efficiency (0 to 1)
        
        Source: Fiber Optics notes, Eq. 15 (page 7)
        """
        w_ratio = w_beam / w_fiber
        
        if np.isinf(R_beam):
            curvature_term = 0
        else:
            curvature_term = (self.wavelength * R_beam) / (np.pi * w_beam * w_fiber)
        
        denominator = (w_ratio + 1/w_ratio)**2 + curvature_term**2
        return 4 / denominator
    
    def numerical_aperture(self, n_core: float, n_cladding: float) -> float:
        """
        Calculate fiber numerical aperture.
        
        Math:
            NA = √(n_core² - n_cladding²)
        
        Source: Fiber Optics notes, page 3
        """
        return np.sqrt(n_core**2 - n_cladding**2)


# =============================================================================
# SECTION 6: CAVITY ANALYSIS
# =============================================================================

class CavityAnalyzer:
    """
    Analyze optical cavity stability and modes.
    
    A cavity is stable when the round-trip beam reproduces itself.
    This requires: 0 < g₁*g₂ < 1
    
    where g_i = 1 - L/R_i for each mirror.
    
    Source: Optical Cavities notes, pages 1-3
    """
    
    def __init__(self, wavelength: float):
        self.wavelength = wavelength
        self.beam = GaussianBeamTool(wavelength)
    
    def stability_parameter(self, L1: float, L2: float, 
                          R1: float, R2: float) -> float:
        """
        Calculate g₁*g₂ stability parameter.
        
        Math:
            g₁ = 1 - L₁/R₁
            g₂ = 1 - L₂/R₂
            Stable if: 0 < g₁*g₂ < 1
        
        Source: Optical Cavities notes, Eq. 5 (page 2)
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
        
        Source: Optical Cavities notes, pages 3-4
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
        
        Math:
            FSR = c / (2·L_total)
        
        Args:
            L_total: Total round-trip cavity length (m)
        
        Returns:
            FSR in Hz
        
        Source: Optical Cavities notes, page 8
        """
        return c / (2 * L_total)
    
    def finesse(self, reflectivity: float) -> float:
        """
        Calculate cavity finesse.
        
        Math:
            F = π·√R / (1-R)
        
        where R is the product of all mirror reflectivities.
        
        Args:
            reflectivity: Total power reflectivity (product of all mirrors)
        
        Returns:
            Finesse (dimensionless)
        
        Source: Optical Cavities notes, page 9
        """
        R = reflectivity
        return np.pi * np.sqrt(R) / (1 - R)
    
    def linewidth(self, FSR: float, finesse: float) -> float:
        """
        Calculate cavity linewidth (FWHM).
        
        Math:
            Δν = FSR / F
        
        Args:
            FSR: Free spectral range (Hz)
            finesse: Cavity finesse
        
        Returns:
            Linewidth in Hz
        
        Source: Optical Cavities notes, page 9
        """
        return FSR / finesse
    
    def quality_factor(self, frequency: float, linewidth: float) -> float:
        """
        Calculate cavity quality factor.
        
        Math:
            Q = ν₀ / Δν
        
        Args:
            frequency: Resonance frequency (Hz)
            linewidth: Cavity linewidth (Hz)
        
        Returns:
            Quality factor Q (dimensionless)
        
        Source: Optical Cavities notes, page 10
        """
        return frequency / linewidth
    
    def bounce_number(self, finesse: float) -> float:
        """
        Average number of round trips photon makes in cavity.
        
        Math:
            N_bounce = F / π
        
        Args:
            finesse: Cavity finesse
        
        Returns:
            Average number of bounces
        
        Source: Optical Cavities notes, page 10
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
        q_check = self.beam.propagate_q(q, M)
        error = abs(q_check - q) / abs(q)
        if error > 0.01:  # 1% relative error
            warnings.warn(f"Eigenmode verification: relative error = {error:.6f}")
        
        return q
    
    def full_analysis(self, L1: float, L2: float, R1: float, R2: float,
                     reflectivity: float) -> Dict:
        """
        Perform complete cavity analysis.
        
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


# =============================================================================
# SECTION 7: HELPER FUNCTIONS
# =============================================================================

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





@dataclass
class CavityGeometry:
    """Bow-tie cavity geometry parameters (all in cm for compatibility)"""
    W: float  # Width of bow-tie cavity in cm
    H: float  # Height of bow-tie cavity in cm
    f_concave: float  # Focal length of concave mirror in cm
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


# =============================================================================
# BOW-TIE CAVITY CLASS
# =============================================================================

class BowTieCavity:
    """
    Bow-tie cavity with automatic geometry calculations.
    
    This class handles the geometry of a bow-tie cavity where:
    - Two flat mirrors are separated by W (width)
    - Two curved mirrors are separated by diagonal distance
    - Height H determines the bow-tie shape
    """
    
    def __init__(self, geometry: CavityGeometry):
        """
        Args:
            geometry: CavityGeometry object with cavity parameters
        """
        self.geometry = geometry
        self.W = geometry.W * 1e-2  # Convert cm to m
        self.H = geometry.H * 1e-2  # Convert cm to m
        self.f_concave = geometry.f_concave * 1e-2  # Convert cm to m
        self.R_concave = 2 * self.f_concave  # R = 2f for mirror
        self.wavelength = geometry.wavelength * 1e-2  # Convert cm to m
        
        # Calculate product of reflectivities
        self.reflectivity = np.prod(geometry.R_mirrors)
        
        # Calculate effective cavity lengths
        self.d_diagonal = np.sqrt(self.W**2 + (self.H/2)**2)
        self.L1 = self.d_diagonal + self.W
        self.L2 = self.d_diagonal + self.W
        
        # Create analyzer
        self.analyzer = CavityAnalyzer(self.wavelength)
    
    def _build_roundtrip_cm(self):
        """
        Build round-trip ABCD matrix in cm and find the cavity eigenmode q at the input mirror.
        """
        W   = self.geometry.W             # cm
        H   = self.geometry.H             # cm
        f   = self.geometry.f_concave     # cm
        lam = self.geometry.wavelength    # cm

        R_concave = 2 * f                 # radius of curvature in cm
        d_diag  = np.sqrt(W**2 + (H/2)**2)
        d_horiz = W

        M_d      = np.array([[1, d_diag],  [0, 1]])
        M_h      = np.array([[1, d_horiz], [0, 1]])
        M_curved = np.array([[1, 0], [-2/R_concave, 1]])
        M_flat   = np.eye(2)

        # Round-trip starting from M1, composing in path order (leftmost = first element)
        M_rt = (M_d @ M_flat @ M_h @ M_curved @
                M_d @ M_flat @ M_h @ M_flat)

        # Stability: |( A + D ) / 2| < 1  (ring-cavity criterion)
        A, D = M_rt[0, 0], M_rt[1, 1]
        self.stability_param_rt = abs((A + D) / 2)
        self.is_stable_rt = self.stability_param_rt < 1.0
        self.cavity_length_cm = 2 * (d_diag + d_horiz)

        # Eigenmode q at input mirror (in cm)
        beam = GaussianBeamTool(lam)
        q_at_input = self.analyzer.find_eigenmode(M_rt)
        self.q_at_input = q_at_input

        # Waist: propagate from M1 to where Re(q)=0
        z_to_waist = -q_at_input.real   # cm
        M_to_waist = np.array([[1, z_to_waist], [0, 1]])
        q_at_waist = beam.propagate_q(q_at_input, M_to_waist)
        self.cavity_waist   = beam.waist_from_q(q_at_waist)   # cm
        self.waist_location = z_to_waist                       # cm
    
    def analyze_cavity(self) -> Dict:
        """
        Perform complete cavity analysis.
        
        Returns:
            Dictionary with all cavity properties
        """
        results = self.analyzer.full_analysis(
            L1=self.L1,
            L2=self.L2,
            R1=self.R_concave,
            R2=self.R_concave,
            reflectivity=self.reflectivity
        )
        
        # Add geometry info
        results['W'] = self.W
        results['H'] = self.H
        results['d_diagonal'] = self.d_diagonal
        results['R_concave'] = self.R_concave
        self.results = results
        
        # Compute eigenmode q in cm
        self._build_roundtrip_cm()
        
        return results
    
    def print_summary(self, results: Optional[Dict] = None):
        """
        Print cavity analysis summary using niceprint.
        
        Args:
            results: Results dict from analyze_cavity() (optional)
        """
        if results is None:
            results = self.analyze_cavity()
        
        niceprint('---')
        niceprint(f"**Cavity Analysis**", 3)
        
        if results['stable']:
            niceprint(f"<u> Cavity Geometry </u>",5)
            niceprint(f"width: {self.geometry.W:.2f} cm, height: {self.geometry.H:.2f} cm <br>" +
                    f"concave mirror focal length: {self.geometry.f_concave:.2f} cm <br>" +
                    f"round-trip length: {results['L_total']*100:.2f} cm"
                    )
            
            niceprint(f"<u> Cavity Stability </u>",5)
            niceprint("Goal: |g| < 1 for stable cavity <br>" +
                    f"Stability parameter |g|: {results['g_parameter']:.4f} <br>"
                    )
            
            niceprint(fr"<u> Cavity Mode ($\text{{TEM}}_{{00}}$) </u>", 5)
            niceprint(fr"Waist size $w_0$: {results['w0']*1e6:.1f} $\mu m$ <br>" +
                    fr"Waist location from input mirror M1: {results['z0']*100:.2f} cm"
                    )
            
            niceprint(f"<u> Cavity Properties </u>", 5)
            niceprint(f"Free Spectral Range (FSR): {results['FSR']/1e9:.2f} GHz <br>" +
                    f"Finesse: {results['finesse']:7.1f} <br>" if results['finesse'] is not None else "N/A <br>" +
                    f"Linewidth (FWHM): {results['linewidth']/1e6:7.2f} MHz <br>" if results['linewidth'] is not None else "Linewidth (FWHM): N/A <br>" +
                    f"Quality factor Q: {results['Q']:.2e} <br>" if results['Q'] is not None else "Quality factor Q: N/A <br>" +
                    f"Bounce number: {results['n_bounce']:.1f}" if results['n_bounce'] is not None else "Bounce number: N/A"
                     )
        else:
            niceprint(f"**CAVITY UNSTABLE**: {results['g_parameter']:.3f} (must be < 1)")


# =============================================================================
# MODE MATCH OPTIMIZER
# =============================================================================

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


# =============================================================================
# CAVITY TRANSMISSION SIMULATOR
# =============================================================================

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
        self.wavelength =  self.cavity.wavelength
        self.input_power = input_power
        self.piezo = piezo if piezo else PiezoActuator()
        self.detector = detector if detector else Photodetector()
        self.mode_match_efficiency = mode_match_efficiency
        
        # Calculate cavity parameters
        self._calculate_transmission_parameters()
    
    def _calculate_transmission_parameters(self):
        """Calculate cavity transmission parameters"""
        # Get mirror reflectivities
        R = self.cavity.geometry.R_mirrors
        
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
        if self.cavity.results['finesse'] and self.cavity.results['FSR']:
            self.linewidth_hz = self.cavity.results['FSR'] / self.cavity.results['finesse']
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
        wavelength = self.cavity.geometry.wavelength
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
                  f"Finesse: {self.cavity.results['finesse']:7.1f} <br>" +
                  (f"Linewidth (FWHM): {self.linewidth_hz/1e6:7.2f} MHz <br>" if self.linewidth_hz else "") +
                  f"Free Spectral Range: {self.cavity.results['FSR']/1e9:7.2f} GHz"
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
        wavelength_nm = self.cavity.geometry.wavelength * 1e7  # cm to nm
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
        circuit = OpticalCircuit(wavelength=self.wavelength)
        circuit.add_free_space(total_distance)
        M_free_space = circuit.get_total_abcd_matrix()
        q_no_tele = optimizer.beam.propagate_q(q_start, M_free_space)
        eff_no_tele = optimizer.coupling_efficiency(q_no_tele, optimizer.cavity.q_at_input)
        
        w_cavity = optimizer.beam.waist_from_q(optimizer.cavity.q_at_input)
        R_cavity = optimizer.beam.R_from_q(optimizer.cavity.q_at_input)
        
        improvement = eff_with_tele / eff_no_tele if eff_no_tele > 0 else np.inf
        
        niceprint('<u> Coupling Efficiency </u>', 5)
        niceprint(f"Without telescope: {eff_no_tele*100:7.2f} % <br>" +
                f"With telescope:    {eff_with_tele*100:7.2f} % <br>" +
                f"Improvement:      {improvement:7.2f}x")
        
        
        # transmission parameters
        R_mirrors = self.cavity.geometry.R_mirrors
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

# =============================================================================
# CAVITY STABILITY MAP PLOTTING
# =============================================================================

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