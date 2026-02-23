import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
from qutip import Qobj



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

class FreeSpace(OpticalComponent):
    """
    Free space propagation.
    
    ABCD Matrix:
        M = [[1, d],
             [0, 1]]
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
    Mirror for quantum optics (introduces pi phase shift).
    
    Math: Reflection introduces pi phase shift
        U = -I = [[-1, 0 ],
                  [0, -1]]
    
    Physical meaning:
    - Reflection causes pi phase shift (flips sign)
    - Important for interferometer analysis
    
    Note: Can specify which path gets reflected (for asymmetric setups)
    """
    
    def __init__(self, path_index: int = None):
        """
        Parameters:
        -----------
        path_index : int or None
            If specified, only that path gets pi shift
            If None, all paths get pi shift
        """
        self.path_index = path_index
    
    def get_quantum_operator(self, num_paths: int = 2) -> Qobj:
        """Get quantum operator for mirror"""
        data = np.eye(num_paths, dtype=complex)
        if self.path_index is None:
            # All paths get pi shift
            data = -data
        else:
            # Only specified path gets pi shift
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
        U = (1/√2) [[1,  1],
                    [1, -1]]
    
    Physical meaning:
    - Input photon in path 0 → equal superposition (|0⟩ + |1⟩)/√2
    - Input photon in path 1 → equal superposition (|0⟩ - |1⟩)/√2
    - Relative pi phase shift on one path (sign difference)
    
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
            2x2 unitary matrix representing beam splitter transformation
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
             [0, eta1/eta2]]
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
            R(theta) = [[cos, -sin],
                        [sin,  cos]]
        """
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s],
                        [s,  c]])

class Polarizer(PolarizationComponent):
    """
    Linear polarizer at angle theta from horizontal.
    
    Jones Matrix (horizontal polarizer):
        Th = [[1, 0],
              [0, 0]]
    
    Rotated by angle theta:
        T(theta) = R(theta) · Th · R(-theta)
    
    where R(theta) is the rotation matrix.
    """
    
    def __init__(self, angle: float = 0.0, name: str = "Polarizer"):
        super().__init__(name)
        self.angle = angle
        
        # Base horizontal polarizer
        Th = np.array([[1.0, 0.0],
                       [0.0, 0.0]], dtype=complex)
        
        # Rotate if needed
        if angle != 0:
            R = self.rotation_matrix(angle)
            self.jones_matrix = R @ Th @ R.T
        else:
            self.jones_matrix = Th
    
    def get_jones_matrix(self) -> np.ndarray:
        return self.jones_matrix
    
    def __repr__(self):
        angle_deg = np.rad2deg(self.angle)
        return f"Polarizer($\\theta={angle_deg:.1f}^\\circ$)"

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
    Half-wave plate (HWP) with fast axis at angle theta.
    
    A HWP introduces a phase shift of pi between fast and slow axes.
    It rotates linear polarization by 2theta.
    
    Jones Matrix (fast axis horizontal):
        ThWP = [[1,  0],
                 [0, -1]]
    
    Rotated by angle theta:
        T(theta) = R(theta) · ThWP · R(-theta)
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
        return f"HWP($\\theta={angle_deg:.1f}^\\circ, \\delta={self.retardance_deviation:.2f}$ rad)"

class QuarterWavePlate(PolarizationComponent):
    """
    Quarter-wave plate (QWP) with fast axis at angle theta.
    
    A QWP introduces a phase shift of pi/2 between fast and slow axes.
    At theta = 45 deg, it converts linear to circular polarization.
    
    Jones Matrix (fast axis horizontal):
        T_QWP = [[1,  0],
                 [0, -i]]
    
    Rotated by angle theta:
        T(theta) = R(theta) · T_QWP · R(-theta)
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
        return f"QWP($\\theta={angle_deg:.1f}^\\circ, \\delta={self.retardance_deviation:.2f}$ rad)"

class PolarizingBeamSplitter(PolarizationComponent):
    """
    Polarizing Beam Splitter.
    
    Jones Matrix for transmitted (H) port: [[1, 0], [0, 0]]
    
    Jones Matrix for reflected (V) port: [[0, 0], [0, 1]]
    """
    
    def __init__(self, port: str = 'transmitted', name: str = "PBS"):
        super().__init__(name)
        self.port = port.lower()
        
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





