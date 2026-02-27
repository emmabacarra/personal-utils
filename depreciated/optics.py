import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar, differential_evolution
from scipy.signal import sawtooth
from qutip import Qobj, basis
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict, Optional
from dataclasses import dataclass
from typing import Tuple, Optional


c = 299792458

class OpticalComponent(ABC):
    
    def __init__(self, name: str = "Component"):
        self.name = name
    
    @abstractmethod
    def get_abcd_matrix(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_quantum_operator(self, num_paths: int = 2) -> Qobj:
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class GaussianBeam:
    """
    gaussian laser beam with complex beam parameter q
    
    
    laser beam waist: narrowest point of the beam
    Re(q): distance from waist
    Im(q): Rayleigh range (related to waist width)
    """
    
    def __init__(self, w0: float, wavelength: float, z: float = 0):
        """
        Args:
            w0: beam waist radius (m)
            wavelength: wavelength (m)
            z: distance from waist (m)
        """
        self.w0 = w0
        self.wavelength = wavelength
        self.zR = np.pi * w0**2 / wavelength  # Rayleigh range
        self.z = z
        self.q = self._calculate_q()
    
    def _calculate_q(self) -> complex:
        return self.z + 1j * self.zR
    
    def get_width(self) -> float:
        '''
        beam width w(z)
        '''
        return self.w0 * np.sqrt(1 + (self.z / self.zR)**2)
    
    def get_radius_of_curvature(self) -> float:
        """
        radius of curvature R(z)
        """
        if np.abs(self.z) < 1e-12:
            return np.inf
        return self.z * (1 + (self.zR / self.z)**2)
    
    def propagate_through_abcd(self, M: np.ndarray) -> 'GaussianBeam':
        A, B = M[0, 0], M[0, 1]
        C, D = M[1, 0], M[1, 1]
        
        q_out = (A * self.q + B) / (C * self.q + D)
        
        # Extract new parameters from q_out
        inv_q = 1 / q_out
        z_out = np.real(q_out)
        zR_out = np.imag(q_out)
        w0_out = np.sqrt(self.wavelength * zR_out / np.pi)
        
        new_beam = GaussianBeam(w0_out, self.wavelength, z_out)
        new_beam.q = q_out
        new_beam.zR = zR_out
        return new_beam
    
    def __repr__(self):
        return f"GaussianBeam(w0={self.w0*1e6:.1f}um, z={self.z*100:.2f}cm, w(z)={self.get_width()*1e6:.1f}um)"


class QuantumState:
    
    def __init__(self, state: Union[Qobj, int], num_paths: int = 2):
        if isinstance(state, int):
            # basis state |i⟩
            self.state = basis(num_paths, state)
        else:
            self.state = state
        self.num_paths = num_paths
    
    def measure(self, path_index: int) -> float:
        """
        Measure probability of photon in specified path.
        
        Math: P(path i) = |⟨i|ψ⟩|² = |cᵢ|²
        """
        amplitude = self.state[path_index, 0]
        return np.abs(amplitude)**2
    
    def apply_operator(self, operator: Qobj) -> 'QuantumState':
        """Apply unitary operator to state: |ψ_out⟩ = Û|ψ_in⟩"""
        new_state = operator * self.state
        return QuantumState(new_state, self.num_paths)
    
    def __repr__(self):
        probs = [self.measure(i) for i in range(self.num_paths)]
        return f"QuantumState(probs={[f'{p:.3f}' for p in probs]})"


class FreeSpace(OpticalComponent):
    
    def __init__(self, distance: float, name: str = "FreeSpace"):
        super().__init__(name)
        self.distance = distance
    
    def get_abcd_matrix(self) -> np.ndarray:
        return np.array([[1, self.distance],
                        [0, 1]])
    
    def get_quantum_operator(self, num_paths: int = 2) -> Qobj:
        """
        leaves quantum state invariant
        """
        return Qobj(np.eye(num_paths, dtype=complex))
    
    def __repr__(self):
        return f"FreeSpace(d={self.distance*100:.2f}cm)"


class ThinLens(OpticalComponent):
    
    def __init__(self, focal_length: float, name: str = "Lens"):
        super().__init__(name)
        self.f = focal_length
    
    def get_abcd_matrix(self) -> np.ndarray:
        return np.array([[1, 0],
                        [-1/self.f, 1]])
    
    def get_quantum_operator(self, num_paths: int = 2) -> Qobj:
        """Lens doesn't affect quantum state (classical optical element)"""
        return Qobj(np.eye(num_paths, dtype=complex))
    
    def __repr__(self):
        return f"ThinLens(f={self.f*100:.2f}cm)"


class Mirror(OpticalComponent):
    """
    flat mirror
    """
    
    def __init__(self, path_index: int = 0, name: str = "Mirror"):
        super().__init__(name)
        self.path_index = path_index
    
    def get_abcd_matrix(self) -> np.ndarray:
        return np.eye(2)
    
    def get_quantum_operator(self, num_paths: int = 2) -> Qobj:
        """
        phase shift of pi
        """
        data = np.eye(num_paths, dtype=complex)
        data[self.path_index, self.path_index] = -1
        return Qobj(data)
    
    def __repr__(self):
        return f"Mirror(path={self.path_index})"

class CurvedMirror(OpticalComponent):
    """
    curved mirror with radius of curvature R
    """
    
    def __init__(self, radius_of_curvature: float, name: str = "CurvedMirror"):
        super().__init__(name)
        self.R = radius_of_curvature
    
    def get_abcd_matrix(self) -> np.ndarray:
        return np.array([[1, 0],
                        [-2/self.R, 1]])
    
    def get_quantum_operator(self, num_paths: int = 2) -> Qobj:
        """
        phase shift of pi
        """
        return Qobj(-np.eye(num_paths, dtype=complex))
    
    def __repr__(self):
        return f"CurvedMirror(R={self.R*100:.2f}cm)"


class BeamSplitter(OpticalComponent):
    
    def __init__(self, input_a: int = 0, input_b: int = 1, name: str = "BS"):
        super().__init__(name)
        self.input_a = input_a
        self.input_b = input_b
    
    def get_abcd_matrix(self) -> np.ndarray:
        return np.eye(2)
    
    def get_quantum_operator(self, num_paths: int = 2) -> Qobj:
        data = np.eye(num_paths, dtype=complex)
        data[self.input_a, self.input_a] = 1/np.sqrt(2)
        data[self.input_a, self.input_b] = 1/np.sqrt(2)
        data[self.input_b, self.input_a] = 1/np.sqrt(2)
        data[self.input_b, self.input_b] = -1/np.sqrt(2)
        return Qobj(data)
    
    def __repr__(self):
        return f"BeamSplitter(paths={self.input_a},{self.input_b})"


class PhaseShifter(OpticalComponent):
    """
    phase shift e^(i phi): adds optical path length
    """
    
    def __init__(self, path_index: int, phase: float, name: str = "Phase"):
        super().__init__(name)
        self.path_index = path_index
        self.phase = phase
    
    def get_abcd_matrix(self) -> np.ndarray:
        """just like free space propagation"""
        return np.eye(2)
    
    def get_quantum_operator(self, num_paths: int = 2) -> Qobj:
        data = np.eye(num_paths, dtype=complex)
        data[self.path_index, self.path_index] = np.exp(1j * self.phase)
        return Qobj(data)
    
    def __repr__(self):
        return f"PhaseShifter(path={self.path_index}, φ={self.phase:.3f}rad)"



class OpticalCircuit:
    """
    container for optical components
    """
    
    def __init__(self, name: str = "Circuit"):
        self.name = name
        self.components: List[OpticalComponent] = []
    
    def add_component(self, component: OpticalComponent):
        self.components.append(component)
    
    def get_total_abcd_matrix(self) -> np.ndarray:
        M_total = np.eye(2)
        for component in self.components:
            M = component.get_abcd_matrix()
            M_total = M @ M_total  # Matrix multiplication: right to left
        return M_total
    
    def propagate_gaussian_beam(self, input_beam: GaussianBeam) -> GaussianBeam:
        M_total = self.get_total_abcd_matrix()
        return input_beam.propagate_through_abcd(M_total)
    
    def propagate_quantum_state(self, input_state: QuantumState) -> QuantumState:
        state = input_state
        for component in self.components:
            operator = component.get_quantum_operator(state.num_paths)
            state = state.apply_operator(operator)
        return state
    
    def __repr__(self):
        comp_list = "\n  ".join([str(c) for c in self.components])
        return f"OpticalCircuit '{self.name}':\n  {comp_list}"


class MachZehnderInterferometer(OpticalCircuit):
    
    def __init__(self, wavelength: float, delta_L: float = 0, 
                 n: float = 1.0, include_mirrors: bool = True):
        super().__init__("Mach-Zehnder")
        
        self.delta_L = delta_L
        self.wavelength = wavelength
        self.n = n
        
        self.add_component(BeamSplitter(0, 1, "BS1"))
        
        if include_mirrors:
            self.add_component(Mirror(0, "M0"))
            self.add_component(Mirror(1, "M1"))
        
        # phase shift from path length difference
        delta_phi = (2 * np.pi / wavelength) * n * delta_L
        self.add_component(PhaseShifter(1, delta_phi, "Phase"))
        
        self.add_component(BeamSplitter(0, 1, "BS2"))
    
    def get_output_probabilities(self) -> Tuple[float, float]:
        input_state = QuantumState(0, num_paths=2)  # start in path 0
        output_state = self.propagate_quantum_state(input_state)
        
        P0 = output_state.measure(0)
        P1 = output_state.measure(1)
        
        return P0, P1



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



class ModeMatchingOptimizer:
    """
    optimize telescope system to match laser beam to cavity mode
    """
    
    def __init__(self, laser_beam: GaussianBeam, cavity: OpticalCavity,
                 distance_to_cavity: float):
        
        self.laser_beam = laser_beam
        self.cavity = cavity
        self.distance_to_cavity = distance_to_cavity
        
        # Target q-parameter at cavity
        self.q_target = self.cavity.q_res
    
    def merit_function(self, params: np.ndarray) -> float:
        """
        merit function to minimize, returns weighted error between laser and cavity modes
        """
        d1, f1, d_lens, f2 = params
        d2 = self.distance_to_cavity - d1 - d_lens
        
        # Physical constraints
        if d2 < 0.05 or d1 < 0.05 or d_lens < 0.05:
            return 1e10
        if f1 < 0.03 or f2 < 0.03:
            return 1e10
        if f1 > 1.0 or f2 > 1.0:
            return 1e10
        
        # Build telescope circuit
        circuit = OpticalCircuit("Telescope")
        circuit.add_component(FreeSpace(d1))
        circuit.add_component(ThinLens(f1))
        circuit.add_component(FreeSpace(d_lens))
        circuit.add_component(ThinLens(f2))
        circuit.add_component(FreeSpace(d2))
        
        # Propagate laser through telescope
        output_beam = circuit.propagate_gaussian_beam(self.laser_beam)
        q_at_cavity = output_beam.q
        
        # Calculate error
        error_real = np.abs(np.real(q_at_cavity) - np.real(self.q_target))
        error_imag = np.abs(np.imag(q_at_cavity) - np.imag(self.q_target))
        
        # Weight imaginary part more (waist size more critical than location)
        total_error = error_real + 10 * error_imag
        
        return total_error
    
    def optimize(self, initial_guess: Optional[np.ndarray] = None) -> dict:
        
        if initial_guess is None:
            # Reasonable starting guess
            d_avg = self.distance_to_cavity / 3
            initial_guess = [d_avg, 0.15, d_avg, 0.15]
        
        result = minimize(self.merit_function, initial_guess,
                         method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        d1_opt, f1_opt, d_lens_opt, f2_opt = result.x
        d2_opt = self.distance_to_cavity - d1_opt - d_lens_opt
        
        # Calculate final beam
        circuit = OpticalCircuit("Optimized Telescope")
        circuit.add_component(FreeSpace(d1_opt))
        circuit.add_component(ThinLens(f1_opt))
        circuit.add_component(FreeSpace(d_lens_opt))
        circuit.add_component(ThinLens(f2_opt))
        circuit.add_component(FreeSpace(d2_opt))
        
        final_beam = circuit.propagate_gaussian_beam(self.laser_beam)
        
        return {
            'd1': d1_opt,
            'f1': f1_opt,
            'd_lens': d_lens_opt,
            'f2': f2_opt,
            'd2': d2_opt,
            'final_beam': final_beam,
            'error': result.fun,
            'success': result.success,
            'circuit': circuit
        }



def plot_beam_propagation(circuit: OpticalCircuit, input_beam: GaussianBeam,
                         z_range: Tuple[float, float], num_points: int = 500):
    z_array = np.linspace(z_range[0], z_range[1], num_points)
    w_array = []
    
    # Track position through circuit
    z_current = 0
    component_positions = [0]
    beam_current = GaussianBeam(input_beam.w0, input_beam.wavelength, 0)
    
    for i, z in enumerate(z_array):
        # Find which component we're at
        z_in_circuit = z - z_range[0]
        
        # Propagate from current position
        dz = z_in_circuit - z_current
        if dz > 0:
            temp_beam = GaussianBeam(beam_current.w0, beam_current.wavelength,
                                    beam_current.z + dz)
            w_array.append(temp_beam.get_width() * 1e3)  # to mm
        else:
            w_array.append(beam_current.get_width() * 1e3)
    
    plt.figure(figsize=(12, 6))
    plt.plot(z_array * 100, w_array, 'b-', linewidth=2)
    plt.xlabel('Position (cm)')
    plt.ylabel('Beam Width w(z) (mm)')
    plt.title('Gaussian Beam Propagation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cavity_stability_map(R: float, rho: float, W_range: Tuple[float, float],
                              H_range: Tuple[float, float], W_design: float,
                              H_design: float, wavelength, num_points: int = 250):
    """
    contour plots of cavity stability and FSR as function of width and height
    """
    W_arr = np.linspace(W_range[0], W_range[1], num_points)
    H_arr = np.linspace(H_range[0], H_range[1], num_points)
    W_grid, H_grid = np.meshgrid(W_arr, H_arr)
    
    FSR_grid = np.zeros_like(W_grid)
    stability_grid = np.zeros_like(W_grid)
    
    # Calculate properties over parameter space
    for i in range(num_points):
        for j in range(num_points):
            W, H = W_grid[i,j], H_grid[i,j]
            d_diag = np.sqrt(W**2 + H**2)
            L1 = d_diag + W
            L2 = d_diag + W
            
            cavity = OpticalCavity(L1, L2, R, R, rho, wavelength)
            cavity._calculate_resonance_properties()
            FSR_grid[i,j] = cavity.FSR
            stability_grid[i,j] = cavity.stability_param
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ---------------------------------------------------------
    
    FSR_GHz = FSR_grid / 1e9 # convert to GHz
    
    levels_fsr = np.linspace(np.min(FSR_GHz), np.max(FSR_GHz), 25)
    contour_fsr = ax1.contourf(W_grid*100, H_grid*100, FSR_GHz, 
                               levels=levels_fsr, cmap='plasma')
    contour_lines = ax1.contour(W_grid*100, H_grid*100, FSR_GHz, 
                levels=15, colors='white', linewidths=0.5, alpha=0.4)
    ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    
    # annotate
    stability_boundary = ax1.contour(W_grid*100, H_grid*100, stability_grid, 
                                     levels=[1.0], colors='cyan', linewidths=3)
    ax1.clabel(stability_boundary, inline=True, fontsize=10, fmt='Stability Limit')
    unstable_mask = stability_grid > 1
    ax1.contourf(W_grid*100, H_grid*100, unstable_mask.astype(float),
                levels=[0.5, 1.5], colors='black', alpha=0.2)
    
    # Mark current design
    ax1.plot(W_design*100, H_design*100, 'r*', markersize=20, 
            markeredgecolor='white', markeredgewidth=2, 
            label=f'W: {W_design*100:.1f} cm, H: {H_design*100:.1f} cm')
    
    plt.colorbar(contour_fsr, ax=ax1, label='FSR (GHz)', pad=0.02)
    ax1.set_xlabel('Width W (cm)'), ax1.set_ylabel('Height H (cm)')
    ax1.set_title('FSR - Full Parameter Space', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ---------------------------------------------------------
    
    levels_stab = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    colors_stab = plt.cm.RdYlGn_r(np.linspace(0, 1, len(levels_stab)-1))
    
    contour_stab = ax2.contourf(W_grid*100, H_grid*100, stability_grid,
                               levels=levels_stab, colors=colors_stab, alpha=0.8)
    contour_lines_stab = ax2.contour(W_grid*100, H_grid*100, stability_grid,
                                     levels=levels_stab, colors='black', 
                                     linewidths=1, alpha=0.5)
    ax2.clabel(contour_lines_stab, inline=True, fontsize=9, fmt='%.1f')
    
    # annotate
    boundary = ax2.contour(W_grid*100, H_grid*100, stability_grid,
                          levels=[1.0], colors='red', linewidths=4)
    ax2.clabel(boundary, inline=True, fontsize=12, fmt='Stability Limit')
    ax2.plot(W_design*100, H_design*100, 'r*', markersize=20,
            markeredgecolor='white', markeredgewidth=2, 
            label=f'W: {W_design*100:.1f} cm, H: {H_design*100:.1f} cm')
    
    cbar2 = plt.colorbar(contour_stab, ax=ax2, label='Stability Parameter (Stable btwn 0 and 1)', pad=0.02)
    ax2.set_xlabel('Width W (cm)'), ax2.set_ylabel('Height H (cm)')
    ax2.set_title(r'Stability Parameter $\frac{(A+D)^2}{4}$', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def build_optical_system(d0, d1, d2, d3):
    circuit = OpticalCircuit("Fiber Coupling System")
    
    circuit.add_component(FreeSpace(d0, "d0: laser to lens1"))
    circuit.add_component(ThinLens(f_lens1, "Telescope Lens 1"))
    circuit.add_component(FreeSpace(d1, "d1: lens1 to lens2"))
    circuit.add_component(ThinLens(f_lens2, "Telescope Lens 2"))
    circuit.add_component(FreeSpace(d2, "d2: lens2 to mirror1"))
    circuit.add_component(FreeSpace(d_mirror_sep, "mirror separation"))
    circuit.add_component(FreeSpace(d3, "d3: mirror2 to collimator"))
    
    return circuit


def merit_function(params, d0, laser_beam):
    """
    Merit function to minimize
    Goal: collimated beam at collimator with width w = w0_collimator_target
    Collimated means far from waist (large z/zR ratio)
    """
    d1, d2, d3 = params
    
    # Physical constraints (relaxed)
    if d1 < 0.005 or d2 < 0.005 or d3 < 0.005:
        return 1e10
    if d1 > 1.0 or d2 > 1.0 or d3 > 1.0:
        return 1e10
    
    # Build optical system and propagate beam
    try:
        circuit = build_optical_system(d0, d1, d2, d3)
        beam_at_collimator = circuit.propagate_gaussian_beam(laser_beam)
    except:
        return 1e10
    
    # We want:
    # 1. Beam WIDTH at collimator = w0_collimator_target
    # 2. Beam is collimated (far from waist, so |z| >> zR)
    
    beam_width = beam_at_collimator.get_width()
    error_width = np.abs(beam_width - w0_collimator_target) / w0_collimator_target
    
    # For collimation: want |z/zR| to be large (beam far from waist)
    # Radius of curvature R = z(1 + (zR/z)²) → ∞ as |z/zR| → ∞
    z_over_zR = np.abs(beam_at_collimator.z / beam_at_collimator.zR)
    # Penalize if beam is near its waist (want z_over_zR > 3 for good collimation)
    if z_over_zR < 3:
        error_collimation = (3 - z_over_zR) / 3
    else:
        error_collimation = 0
    
    total_error = error_width + 10 * error_collimation
    
    return total_error