from ..general import *
from .params import *
from .components import *

import numpy as np
from scipy.signal import sawtooth
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List, Tuple, Dict
from qutip import Qobj, basis
from scipy.ndimage import gaussian_filter




class GaussianBeamTool:
    """
    Tools for Gaussian beam calculations using ABCD matrices.
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
        """
        z_R = np.pi * w0**2 / self.lam
        return (z - z0) + 1j * z_R
    
    def waist_from_q(self, q: complex) -> float:
        """
        Extract beam waist from q-parameter.
        """
        if q.imag <= 0:
            raise ValueError(f"Invalid q: Im(q) = {q.imag} must be positive")
        return np.sqrt(self.lam * abs(q.imag) / np.pi)
    
    def R_from_q(self, q: complex) -> float:
        """
        Extract wavefront radius of curvature from q.
        """
        if abs(q.real) < 1e-10:
            return np.inf
        return q.real * (1 + (q.imag / q.real)**2)
    
    def propagate_q(self, q: complex, M: np.ndarray) -> complex:
        """
        Propagate q through ABCD matrix.
        """
        A, B = M[0, 0], M[0, 1]
        C, D = M[1, 0], M[1, 1]
        return (A * q + B) / (C * q + D)
    
    def rayleigh_range(self, w0: float) -> float:
        """
        Calculate Rayleigh range.
        """
        return np.pi * w0**2 / self.lam
    
    def divergence_angle(self, w0: float) -> float:
        """
        Far-field divergence half-angle.
        """
        return self.lam / (np.pi * w0)

class QuantumState:
    """
    Represents a quantum state for interferometry and quantum optics.
    
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
    
    Example:
    --------
    # Create superposition
    state = QuantumState(0, num_paths=2)
    bs = BeamSplitter()
    state = state.apply_operator(bs.get_quantum_operator())
    
    # Add pi/2 phase shift to path 1
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
            Diagonal matrix with e^(i  phi) on specified path
        """
        data = np.eye(num_paths, dtype=complex)
        data[self.path_index, self.path_index] = np.exp(1j * self.phase)
        return Qobj(data)
    
    def __repr__(self):
        return f"PhaseShifter(path={self.path_index}, $\\phi={self.phase:.3f}$ rad)"

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
    
    # Components ────────────────────────────────────────────────────────────
    
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
    
    # Analysis Methods ───────────────────────────────────────────────────────
    
    def get_total_abcd_matrix(self) -> np.ndarray:
        """
        Calculate total ABCD matrix for the circuit.
        
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
        Propagate a polarization state through the circuit. Automatically handles retroreflective mirrors.
        
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
            J = self.get_total_jones_matrix()
            state_out = J @ state_in
            
            self.intensity_in = np.abs(state_in[0])**2 + np.abs(state_in[1])**2
            self.intensity_out = np.abs(state_out[0])**2 + np.abs(state_out[1])**2
            self.transmission = self.intensity_out / self.intensity_in if self.intensity_in > 0 else 0
            
            return state_out
        
        if len(mirror_indices) > 1:
            raise ValueError("Circuit can only contain one RetroReflectiveMirror")
        
        mirror_idx = mirror_indices[0]
        components_before = self.components[:mirror_idx]
        mirror = self.components[mirror_idx]
        components_after = self.components[mirror_idx + 1:]
        
        # forward path (up to mirror)
        state = state_in.copy()
        for component in components_before:
            J = component.get_jones_matrix()
            if J is not None:
                state = J @ state
        state_before_mirror = state.copy()
        
        # mirror reflection (complex conjugation)
        state = mirror.apply_reflection(state)
        state_after_mirror = state.copy()
        
        # backward path (inverse matrices, reversed order)
        # skip polarizers (they're singular/absorptive)
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
        
        zcurrent = 0
        qcurrent = q_in
        
        for component in self.components:
            if isinstance(component, FreeSpace):
                # Sample through free space
                z_local = np.linspace(0, component.d, num_points // len(self.components))
                for z in z_local:
                    M_local = FreeSpace(z).get_abcd_matrix()
                    q_local = self.beam.propagate_q(qcurrent, M_local)
                    positions.append(zcurrent + z)
                    widths.append(self.beam.waist_from_q(q_local))
                
                zcurrent += component.d
                qcurrent = self.beam.propagate_q(qcurrent, component.get_abcd_matrix())
            else:
                # Just record the position at the component
                qcurrent = self.beam.propagate_q(qcurrent, component.get_abcd_matrix())
                positions.append(zcurrent)
                widths.append(self.beam.waist_from_q(qcurrent))
        
        return np.array(positions), np.array(widths)
    
    def clear(self):
        """Remove all components from the circuit."""
        self.components = []
        return self
    
    def __repr__(self):
        return f"OpticalCircuit($\\lambda$ = {self.wavelength*1e9:.1f} nm, {len(self.components)} components)"
    
    def __str__(self):
        """Pretty print the circuit."""
        lines = [f"\n{self.name}"]
        lines.append("=" * 60)
        lines.append(f"Wavelength: {self.wavelength*1e9:.1f} nm")
        lines.append(f"Components ({len(self.components)}):")
        for i, component in enumerate(self.components, 1):
            lines.append(f"  {i}. {component}")
        return "\n".join(lines)



class MachZehnderInterferometer:
    """
    Mach-Zehnder interferometer with quantum state evolution.
    
    Example:
    --------
    # Constructive interference in output 0
    mz = MachZehnderInterferometer(delta_L=0, wavelength=632.8e-9)
    P0, P1 = mz.get_output_probabilities()
    print(f"P0 = {P0:.3f}, P1 = {P1:.3f}")
    P0 = 1.000, P1 = 0.000
    
    # Destructive interference (pi phase shift)
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
        
        # phase shift from path difference
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
                f"$\\lambda = {self.wavelength*1e9:.1f}$nm, $\\phi = {self.delta_phi:.3f}$ rad)")



class FiberCoupler:
    """
    Calculate fiber coupling efficiency.
    """
    
    def __init__(self, wavelength: float):
        self.wavelength = wavelength
        self.beam = GaussianBeamTool(wavelength)
    
    def coupling_efficiency(self, w_beam: float, R_beam: float,
                          w_fiber: float) -> float:
        """
        Calculate coupling efficiency for mode matching.
        
        Args:
            w_beam: Beam waist at fiber (meters)
            R_beam: Beam radius of curvature at fiber (meters, use inf for flat)
            w_fiber: Fiber mode field diameter / 2 (meters)
        
        Returns:
            Coupling efficiency (0 to 1)
        """
        w_ratio = w_beam / w_fiber
        
        if np.isinf(R_beam):
            curvature_term = 0
        else:
            curvature_term = (self.wavelength * R_beam) / (np.pi * w_beam * w_fiber)
        
        denominator = (w_ratio + 1/w_ratio)**2 + curvature_term**2
        return 4 / denominator
    
    def numerical_aperture(self, ncore: float, ncladding: float) -> float:
        """
        Calculate fiber numerical aperture.
        """
        return np.sqrt(ncore**2 - ncladding**2)

class FiberOutcoupledBeamSimulator:
    """
    Simulate outcoupled-beam measurements.

    Parameters
    ----------
    wavelength      : float - laser wavelength [m]
    w_laser         : float - laser beam waist radius before fiber [m]
    w_fiber_sm      : float - SM fiber mode-field radius (MFD/2) [m]
    w_fiber_mm      : float - MM fiber coupling target radius (~0.75·r_core) [m]
    eta_theoretical : float - mode-overlap eta from FiberModeMatchOptimizer (0→1)
    P_in            : float - input laser power [W]
    fiber_type      : str - 'SM' or 'MM' (controls Q3 polarization model)
    """

    def __init__(
        self,
        wavelength:       float = 633e-9,
        w_laser:          float = 0.5e-3,
        w_fiber_sm:       float = 2.0e-6,
        w_fiber_mm:       float = 18.75e-6,
        eta_theoretical:  float = 0.95,
        P_in:             float = 1e-3,
        fiber_type:       str   = 'SM',
    ):
        self.lam            = wavelength
        self.w_laser        = w_laser
        self.w_fiber_sm     = w_fiber_sm
        self.w_fiber_mm     = w_fiber_mm
        self.eta_theory     = eta_theoretical
        self.P_in           = P_in
        self.fiber_type     = fiber_type.upper()
        self.beam           = GaussianBeamTool(wavelength)

    def coupling_efficiency(
        self,
        fresnel_R:     float = 0.04, # air/glass reflection at fiber face
        fiber_loss_db: float = 3.0,    # fiber transmission loss dB
        alignment_eff: float = 0.90,   # residual misalignment factor
        ax: Optional[plt.Axes] = None,
    ) -> dict:
        """
        Parameters
        ----------
        fresnel_R     : Fresnel power reflection coefficient at fiber face
        fiber_loss_db : Total fiber propagation loss in dB
        alignment_eff : Fractional efficiency due to alignment imperfection
        """
        eta_fresnel = 1.0 - fresnel_R
        eta_fiber   = 10 ** (-fiber_loss_db / 10.0)
        eta_real    = self.eta_theory * eta_fresnel * eta_fiber * alignment_eff
        P_out       = self.P_in * eta_real

        labels = [
            'Mode\noverlap $\\eta_m$',
            'Fresnel\n$\\eta_F$',
            'Fiber\nloss $\\eta_f$',
            'Alignment\n$\\eta_a$',
        ]
        values = [self.eta_theory, eta_fresnel, eta_fiber, alignment_eff]
        colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']

        # cumul has length n+1 = 5  (starting from 1 before any loss)
        cumul = np.cumprod([1] + values) * 100 # shape (5,)
        n = len(labels) # 4

        own_fig = ax is None
        if own_fig:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.bar(np.arange(n), [v * 100 for v in values],
               color=colors, edgecolor='white', linewidth=1.2, tick_label=labels)

        x_step = np.arange(n + 1) - 0.5 # fixing shape
        ax.step(x_step, cumul, where='post', color='k', linewidth=2,
                linestyle='--', label=f'Cumulative → {cumul[-1]:.1f}%')

        ax.axhline(self.eta_theory * 100, color='steelblue', linestyle=':',
                   linewidth=1.5,
                   label=f'Mode-overlap only: {self.eta_theory * 100:.1f}%')
        ax.set_ylabel('Efficiency contribution (%)', fontsize=11)
        ax.set_ylim(0, 115)
        ax.set_title(f'Coupling budget ({self.fiber_type})  '
            f'$P_{{out}}$ ≈ {P_out * 1e3:.2f} mW  '
            f'($\\eta$ = {eta_real * 100:.1f}%)',
            fontsize=10, fontweight='bold'
        )
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

        if own_fig:
            plt.tight_layout()

        return {'eta_mode': self.eta_theory, 'eta_fresnel': eta_fresnel,
                'eta_fiber': eta_fiber, 'eta_align': alignment_eff,
                'eta_real': eta_real, 'P_out': P_out}

    def mode_profiles(
        self,
        grid_size: int = 300,
    ) -> plt.Figure:
        """
        Simulate 2D intensity profiles before and after the fiber.
        """
        x = np.linspace(-3, 3, grid_size)
        y = np.linspace(-3, 3, grid_size)
        X, Y = np.meshgrid(x, y)
        R2 = X**2 + Y**2

        # before fiber: TEM_00 + 5% TEM_10 admixture
        I_00 = np.exp(-2 * R2)
        I_10 = (2 * X)**2 * np.exp(-2 * R2)
        I_10 /= I_10.max()
        I_before = 0.93 * I_00 + 0.07 * I_10
        I_before /= I_before.max()

        # after SM fiber: pure TEM_00
        I_sm = np.exp(-2 * R2)
        I_sm /= I_sm.max()

        # after MM fiber: speckle
        a_mm  = 25e-6
        NA_mm = 0.22
        N_modes = int(2 * (np.pi * a_mm / self.lam)**2 * NA_mm**2)
        N_modes = min(N_modes, 300)   # cap for speed

        rng = np.random.default_rng(seed=42)
        E_speckle = np.zeros((grid_size, grid_size), dtype=complex)
        for _ in range(N_modes):
            kx = rng.uniform(-1, 1)
            ky = rng.uniform(-1, 1)
            phi = rng.uniform(0, 2 * np.pi)
            amp = rng.rayleigh(1.0)
            E_speckle += amp * np.exp(1j * (kx * X + ky * Y + phi))

        I_mm = np.abs(E_speckle)**2
        mask = R2 > 2.5**2 # clip to a circular aperture (fiber core boundary)
        I_mm[mask] = 0
        I_mm = gaussian_filter(I_mm, sigma=0.5)   # mild blur to be realistic
        I_mm /= I_mm.max()

        fig = plt.figure(figsize=(13, 4))
        gs  = gridspec.GridSpec(1, 3, wspace=0.35)

        panels = [
            (I_before, 'Before fiber\n(raw HeNe — TEM$_{00}$ + TEM$_{10}$)', 'inferno'),
            (I_sm,     'After SM fiber\n(pure TEM$_{00}$ spatial filter)',     'inferno'),
            (I_mm,     'After MM fiber\n(speckle — many modes interfering)',   'inferno'),
        ]

        for i, (I, title, cmap) in enumerate(panels):
            ax = fig.add_subplot(gs[i])
            im = ax.imshow(I, extent=[-3, 3, -3, 3], origin='lower',
                           cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('x (beam widths)', fontsize=9)
            ax.set_ylabel('y (beam widths)', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label='Norm. intensity')

        fig.suptitle('Laser Mode Profiles: Before vs. After Fiber',
                     fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()

        return fig

    def pbs_polarization(
        self,
        t_total:     float = 60.0,   # seconds
        n_points:    int   = 600,
        air_event:   float = 15.0,   # seconds when air blast starts
        shake_event: float = 40.0,   # seconds when shaking starts
        event_duration: float = 8.0, # duration of each perturbation [s]
        seed:        int   = 7,
    ) -> plt.Figure:
        """
        Simulate PBS power vs. time with air and shake perturbations.
        """
        rng  = np.random.default_rng(seed)
        t    = np.linspace(0, t_total, n_points)
        dt   = t[1] - t[0]

        psi_in = np.array([1.0 + 0j, 0.0 + 0j]) # linearly H-polarized

        # noise amplitudes (radians per sqrt(step))
        if self.fiber_type == 'SM':
            sigma_quiet = 0.02
            sigma_air   = 0.20
            sigma_shake = 0.60
            n_modes_pol = 1    # one spatial mode, cleaner baseline
        else:  # MM
            sigma_quiet = 0.05
            sigma_air   = 0.40
            sigma_shake = 1.20
            n_modes_pol = 5    # average over multiple modes, more scrambled

        # simulate theta and delta as Ornstein-Uhlenbeck processes
        theta = np.zeros(n_points)
        delta = np.zeros(n_points)
        theta[0] = rng.uniform(0, np.pi)
        delta[0] = rng.uniform(0, 2 * np.pi)

        tau_relax = 10.0  # relaxation time constant [s]

        for i in range(1, n_points):
            # determine perturbation level at this time step
            in_air   = air_event   <= t[i] <= air_event   + event_duration
            in_shake = shake_event <= t[i] <= shake_event + event_duration

            if in_shake:
                sigma = sigma_shake
            elif in_air:
                sigma = sigma_air
            else:
                sigma = sigma_quiet

            # Ornstein-Uhlenbeck: mean-reverting random walk
            dtheta = -(theta[i-1] / tau_relax) * dt + sigma * rng.normal() * np.sqrt(dt)
            ddelta = -(delta[i-1] / tau_relax) * dt + sigma * rng.normal() * np.sqrt(dt)
            theta[i] = theta[i-1] + dtheta
            delta[i] = delta[i-1] + ddelta

        # compute P_H(t)
        from .helpers import _fiber_jones, _pbs_transmitted_power
        P_H = np.zeros(n_points)
        for i in range(n_points):
            if self.fiber_type == 'SM':
                T = _fiber_jones(theta[i], delta[i])
                psi_out = T @ psi_in
                P_H[i] = _pbs_transmitted_power(psi_out)
            else:
                # multi mode: average over n_modes_pol independent polarization modes
                vals = []
                for m in range(n_modes_pol):
                    T_m = _fiber_jones(
                        theta[i] + m * np.pi / n_modes_pol,
                        delta[i] + m * 0.8
                    )
                    psi_m = T_m @ psi_in
                    vals.append(_pbs_transmitted_power(psi_m))
                P_H[i] = float(np.mean(vals))

        P_V = 1.0 - P_H   # power conservation

        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

        ax1, ax2 = axes
        ax1.plot(t, P_H, color='#4c72b0', linewidth=1.2, label='$P_H$ (PBS transmitted)')
        ax1.plot(t, P_V, color='#dd8452', linewidth=1.2, linestyle='--',
                 label='$P_V$ (PBS reflected)')
        ax1.set_ylabel('Fraction of total power', fontsize=11)
        ax1.set_ylim(-0.05, 1.10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Shade perturbation windows
        for ax in [ax1, ax2]:
            ax.axvspan(air_event, air_event + event_duration,
                       color='skyblue', alpha=0.25, label='Air blast')
            ax.axvspan(shake_event, shake_event + event_duration,
                       color='salmon', alpha=0.25, label='Shaking')

        ax2.plot(t, theta % np.pi, color='#55a868', linewidth=1.0,
                 label='Fiber fast-axis θ (mod pi)')
        ax2.plot(t, delta % (2*np.pi) / (2*np.pi), color='#8172b2', linewidth=1.0,
                 linestyle='-.', label='Fiber retardance $\\delta / 2\\pi$')
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Fiber birefringence params.', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Annotation arrows
        ax1.annotate('Air blast\n(thermal $\\Delta n$)', xy=(air_event + event_duration/2, 0.05),
                     fontsize=8, ha='center', color='steelblue', fontweight='bold')
        ax1.annotate('Shaking\n(stress $\\Delta n$)', xy=(shake_event + event_duration/2, 0.05),
                     fontsize=8, ha='center', color='crimson', fontweight='bold')

        fig.suptitle(f'PBS Power vs. Time with Perturbations ({self.fiber_type} fiber)',fontsize=11, fontweight='bold')
        plt.tight_layout()
        return fig

class BowTieCavity:
    """
    Bow-tie cavity with automatic geometry calculations.
    """
    
    def __init__(self, geometry: CavityGeometry):
        """
        Args:
            geometry: CavityGeometry object with cavity parameters
        """
        self.geometry = geometry
        self.W = geometry.W
        self.H = geometry.H
        self.fconcave = geometry.fconcave
        self.Rconcave = 2 * self.fconcave  # R = 2f for mirror
        self.wavelength = geometry.wavelength
        
        # Calculate product of reflectivities
        self.reflectivity = np.prod(geometry.R_mirrors)
        
        # Calculate effective cavity lengths
        self.d_diagonal = np.sqrt(self.W**2 + (self.H/2)**2)
        self.L1 = self.d_diagonal + self.W
        self.L2 = self.d_diagonal + self.W
        
        # Create analyzer
        from .analyzers import CavityAnalyzer
        self.analyzer = CavityAnalyzer(self.wavelength)
    
    def _build_roundtrip(self):
        """
        Internal.
        Build round-trip ABCD matrix and find the cavity eigenmode q at the input mirror.
        All quantities in SI meters.
        """
        W   = self.geometry.W             # m
        H   = self.geometry.H             # m
        f   = self.geometry.fconcave      # m
        lam = self.geometry.wavelength    # m

        Rconcave = 2 * f                 # radius of curvature in m
        d_diag  = np.sqrt(W**2 + (H/2)**2)
        dhoriz = W

        M_d      = np.array([[1, d_diag],  [0, 1]])
        Mh      = np.array([[1, dhoriz], [0, 1]])
        Mcurved = np.array([[1, 0], [-2/Rconcave, 1]])
        M_flat   = np.eye(2)

        # Round-trip starting from M1, composing in path order (leftmost = first element)
        M_rt = (M_d @ M_flat @ Mh @ Mcurved @
                M_d @ M_flat @ Mh @ M_flat)

        # Stability: |( A + D ) / 2| < 1  (ring-cavity criterion)
        A, D = M_rt[0, 0], M_rt[1, 1]
        self.stability_param_rt = abs((A + D) / 2)
        self.is_stable_rt = self.stability_param_rt < 1.0
        self.cavity_length = 2 * (d_diag + dhoriz)   # m

        # Eigenmode q at input mirror (in m)
        beam = GaussianBeamTool(lam)
        q_at_input = self.analyzer.find_eigenmode(M_rt)
        self.q_at_input = q_at_input

        # Waist: propagate from M1 to where Re(q) = 0
        z_to_waist = -q_at_input.real   # m
        M_to_waist = np.array([[1, z_to_waist], [0, 1]])
        q_at_waist = beam.propagate_q(q_at_input, M_to_waist)
        self.cavity_waist   = beam.waist_from_q(q_at_waist)   # m
        self.waist_location = z_to_waist                       # m
    
    def analyze(self) -> Dict:
        """
        Perform complete cavity analysis.
        
        Returns:
            Dictionary with all cavity properties
        """
        results = self.analyzer.full_analysis(
            L1=self.L1,
            L2=self.L2,
            R1=self.Rconcave,
            R2=self.Rconcave,
            reflectivity=self.reflectivity
        )
        
        # Add geometry info
        results['W'] = self.W
        results['H'] = self.H
        results['d_diagonal'] = self.d_diagonal
        results['Rconcave'] = self.Rconcave
        self.results = results
        
        # Compute eigenmode q
        self._build_roundtrip()
        
        return results
    
    def print_summary(self, results: Optional[Dict] = None):
        """
        Print cavity analysis summary using niceprint.
        
        Args:
            results: Results dict from analyze() (optional)
        """
        if results is None:
            results = self.analyze()
        
        niceprint('---')
        niceprint(f"**Cavity Analysis**", 3)
        
        if results['stable']:
            niceprint(f"<u> Cavity Geometry </u>",5)
            niceprint(f"width: {self.geometry.W * 100:.2f} cm, height: {self.geometry.H * 100:.2f} cm <br>" +
                    f"concave mirror focal length: {self.geometry.fconcave * 100:.2f} cm <br>" +
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

class CavityTransmissionSimulator:
    """
    Simulates cavity transmission signal as piezo scans cavity length.
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
        """
        Internal. Calculate cavity transmission parameters.
        """
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
        
        Parameters:
        -----------
        delta_L : float
            Change in cavity length in m
        
        Returns:
        --------
        phase_shift : float
            Round-trip phase shift in radians
        """
        wavelength = self.cavity.geometry.wavelength  # m
        return 4 * np.pi * delta_L / wavelength
    
    def piezo_displacement(self, time: np.ndarray) -> np.ndarray:
        """
        Calculate piezo displacement as function of time. Uses triangle
        wave for scanning.
        
        Parameters:
        -----------
        time : np.ndarray
            Time array in seconds
        
        Returns:
        --------
        displacement : np.ndarray
            Piezo displacement in m
        """
        # Triangle wave voltage
        voltage = self.piezo.voltage_amplitude * sawtooth(
            2 * np.pi * self.piezo.frequency * time, 
            width=0.5
        ) + self.piezo.offset_voltage
        
        # Convert to displacement: nm/V * V = nm, then nm to m
        displacement_nm = self.piezo.displacement_per_volt * voltage
        displacement_m = displacement_nm * 1e-9
        
        return displacement_m
    
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
        displacement_nm = displacement / 1e-9
        
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
        wavelength_nm = self.cavity.geometry.wavelength * 1e9  # m to nm
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



class SPDCSimulator:

    def __init__(self, params: Optional[SPDC] = None):
        self.p = params or SPDC()
        self.lambda_spdc = 2.0 * self.p.lambda_pump

    def pump_photon_rate(self, P_pump: float) -> float:
        """Pump photon rate [photons/s] for pump power P_pump [W]."""
        from .helpers import photon_energy
        return P_pump / photon_energy(self.p.lambda_pump)

    def pair_rate(self, P_pump: float) -> float:
        """SPDC pair rate [pairs/s].  Linear in P_pump (undepleted pump)."""
        return self.p.spdc_efficiency * self.pump_photon_rate(P_pump)

    def spdc_W(self, P_pump: float) -> float:
        """
        Total power [W] in all down-converted photons.
        """
        return self.p.spdc_efficiency * P_pump

    def mA_to_W(self, I_mA: np.ndarray) -> np.ndarray:
        """
        Laser diode current [mA] --> estimated pump power [W].
        Linear above threshold, zero below.
        """
        I = np.asarray(I_mA, dtype=float)
        slope = self.p.P_max / (self.p.I_operating - self.p.I_threshold)
        return np.where(I > self.p.I_threshold,
                        slope * (I - self.p.I_threshold), 0.0)

    def Rcoin_W(self, P_pump: np.ndarray) -> np.ndarray:
        """
        Coincidence rate [counts/s] vs. pump power [W].
        """
        P = np.asarray(P_pump, dtype=float)
        return self.p.eta_1 * self.p.eta_2 * self.pair_rate(P)

    def Rcoin_HWP(
        self,
        fast_axis: np.ndarray,
        R_max: float,
        V: float = 1.0,
        phi: float = 0.0,
    ) -> np.ndarray:
        """
        Coincidence rate [counts/s] vs. HWP fast-axis angle [degrees] in
        arm 2, with arm 1 fixed measuring H.

        Parameters
        ----------
        fast_axis  : HWP fast-axis angles in arm 2 [degrees]
        R_max      : peak coincidence rate [counts/s]
        V : fringe visibility V in [0, 1]
        phi        : Bell state relative phase [radians]

        Returns
        -------
        R : ndarray of coincidence rates [counts/s]
        """
        fast_axis = np.asarray(fast_axis, dtype=float)

        H_in = np.array([1.0, 0.0], dtype=complex)
        V_in = np.array([0.0, 1.0], dtype=complex)

        # conditional probability of arm 2 passing PBS for each input polarization
        P_H = np.zeros(len(fast_axis))
        P_V = np.zeros(len(fast_axis))
        for i, theta in enumerate(fast_axis):
            
            circ = OpticalCircuit(wavelength=self.lambda_spdc)
            circ.add_hwp(fast_axis_angle=np.deg2rad(theta))

            outh = circ.propagate_polarization(H_in)
            out_V = circ.propagate_polarization(V_in)

            # PBS selects the H-component (element [0]) of the input state
            P_H[i] = np.abs(outh[0])**2
            P_V[i] = np.abs(out_V[0])**2

        # theoretical pure Jones fringe model:
        P_coincidence = 0.5 * P_H # coincidence probability = (1/2)P_H + (0)P_V
        R_jones  = R_max * P_coincidence
        self._last_R_jones  = R_jones

        
        # observed fringe model (when both HWPs scanned):
        theta_rad = np.deg2rad(fast_axis)
        R_fringe = (R_max / 2.0) * (1.0 + V * np.cos(4.0 * theta_rad + phi))
        self._last_R_fringe = R_fringe
        return R_fringe

    def print_summary(self):
        from .helpers import photon_energy
        
        niceprint('**SPDC Results**', 3)
        
        niceprint(f"<u> SPDC Source Parameters </u>", 5)
        niceprint(f"Pump wavelength: {self.p.lambda_pump*1e9:.0f} nm <br>" +
                  f"SPDC efficiency: {self.p.spdc_efficiency:.0e} <br>" +
                  f"Detection efficiencies: $\\eta_1 = {self.p.eta_1:.2f}, \\eta_2 = {self.p.eta_2:.2f}$ <br>" +
                  f"Operating current: {self.p.I_operating:.1f} mA <br>"
                  )
        
        E_p    = photon_energy(self.p.lambda_pump)
        E_spdc   = photon_energy(self.lambda_spdc)
        N_pump = self.pump_photon_rate(self.p.P_max)
        N_pair = self.pair_rate(self.p.P_max)
        N_spdc   = 2.0 * N_pair
        P_spdc   = self.spdc_W(self.p.P_max)
        Rc    = self.Rcoin_W(np.array([self.p.P_max]))[0]
        
        niceprint(f"<u> SPDC Calculations </u>", 5)
        niceprint(f"Power: <br>" +
                    f"$\\quad P_{{pump}}$ = {self.p.P_max*1e3:.1f} mW <br>" +
                    f"$\\quad P_{{SPDC}}$ = {P_spdc*1e3:.1f} mW <br>" +
                  f"Energies: <br>" +
                    f"$\\quad E_{{pump}}$ = {E_p:.3e} J <br>" +
                    f"$\\quad E_{{SPDC}}$ = {E_spdc:.3e} J <br>" +
                  f"Photon rates: <br>" +
                    f"$\\quad \\dot{{N}}_{{pump}}$ = {N_pump:.3e} photons/s <br> " +
                    f"$\\quad \\dot{{N}}_{{SPDC}}$ = {N_spdc:.3e} photons/s <br> " +
                    f"$\\quad \\dot{{N}}_{{pairs}}$ = {N_pair:.3e} pairs/s <br>" +
                  f"Expected coincidence rate: {Rc:.0f} counts/s"
                  )

    def plot_Rcoin_W(
        self,
        ax: Optional[plt.Axes] = None,
        n_points: int = 200,
    ) -> plt.Axes:
        """
        Plot coincidence rate vs. pump power.
        Shows the linear SPDC scaling predicted by the undepleted-pump model.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        I_mA = np.linspace(self.p.I_threshold, self.p.I_operating, n_points)
        P = self.mA_to_W(I_mA)
        R = self.Rcoin_W(P)

        ax.plot(P * 1e3, R, color='steelblue', linewidth=2)
        ax.set_xlabel("Pump power (mW)")
        ax.set_ylabel("Coincidence rate (counts/s)")
        ax.set_title("$R_\\mathrm{coinc}$ vs. pump power\n"
                     "(linear — undepleted-pump SPDC)")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_Rcoin_HWP(
        self,
        R_max: float = 10_000,
        V: float = 0.97,
        phi: float = 0.0,
        n_points: int = 361,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot coincidence rate vs. HWP angle in arm 2.

        Parameters
        ----------
        R_max      : peak coincidence rate [counts/s]
        V : fringe visibility (lab goal: >0.97 in HV basis)
        phi        : Bell state relative phase [radians]
        """
        from .helpers import visibility
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        theta = np.linspace(0, 90, n_points)
        R     = self.Rcoin_HWP(theta, R_max, V, phi)

        R_min_val = R.min()
        R_max_val = R.max()
        V_meas = visibility(R_max_val, R_min_val)

        ax.plot(theta, R, color='crimson', linewidth=2)
        ax.axhline(R_max_val, color='gray', linestyle='--', linewidth=0.8,
                   label=f"$R_{{\\max}}$ = {R_max_val:.0f}")
        ax.axhline(R_min_val, color='gray', linestyle=':', linewidth=0.8,
                   label=f"$R_{{\\min}}$ = {R_min_val:.0f}")
        ax.set_xlabel("HWP angle θ (degrees)")
        ax.set_ylabel("Coincidence rate (counts/s)")
        ax.set_title(f"Coincidence fringes vs. HWP angle\n"
                     f"Visibility $V$ = {V_meas:.3f}")
        ax.set_xticks(np.arange(0, 91, 22.5))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        return ax



class PhotonBeamSimulator:
    """
    Photon beam simulated as a Poisson-sampled time-binned stream.
    """

    def __init__(
        self,
        flux: float,
        dt_raw: float = None,
        mu_target: float = 0.01,
    ):
        """
        Parameters
        ----------
        flux      : average photon flux [photons/s]
        dt_raw    : raw time-bin width [s].  If None (default), chosen
                    automatically so that mu = flux * dt_raw = mu_target.
        mu_target : target mean photons per raw bin when dt_raw is auto-set.
                    Keep << 1 to stay in the Poisson regime.  Default 0.01.
        """
        self.flux      = flux
        self.mu_target = mu_target
        if dt_raw is None:
            # choose dt_raw so mu = mu_target, guaranteeing mu << 1
            self.dt_raw = mu_target / flux if flux > 0 else 1e-6
        else:
            self.dt_raw = dt_raw
        self.mu_raw = flux * self.dt_raw

    @classmethod
    def from_spdc(
        cls,
        simulator: 'SPDCSimulator',
        P_pump: float = None,
        dt_raw: float = None,
        mu_target: float = 0.01,
    ) -> 'PhotonBeamSimulator':
        """
        Initialize simulator from an arm of SPDC simulator.

        Parameters
        ----------
        simulator : SPDCSimulator instance
        P_pump    : pump power [W]  (defaults to simulator.p.P_max)
        dt_raw    : raw bin width [s].  If None, auto-selected from flux.
        mu_target : target mean photons per raw bin (default 0.01)
        """
        if P_pump is None:
            P_pump = simulator.p.P_max
        flux = simulator.pair_rate(P_pump) * simulator.p.eta_1
        return cls(flux=flux, dt_raw=dt_raw, mu_target=mu_target)

    def generate(self, T_run: float) -> np.ndarray:
        """
        Generate one run of Poisson-sampled photon counts.

        Parameters
        ----------
        T_run : total run duration [s]

        Returns
        -------
        counts : 1-D int array of length n_bins = round(T_run / dt_raw)
        """
        n_bins = int(T_run / self.dt_raw)
        return np.random.poisson(self.mu_raw, size=n_bins)

    def boxcar(self, raw_counts: np.ndarray, n_avg: int) -> np.ndarray:
        """
        Coarsen raw bins by summing n_avg consecutive bins.

        Parameters
        ----------
        raw_counts : 1-D int array from generate()
        n_avg      : number of raw bins to merge per output bin
        """
        n = (len(raw_counts) // n_avg) * n_avg
        return raw_counts[:n].reshape(-1, n_avg).sum(axis=1)

    def boxcar_to_window(self, raw_counts: np.ndarray, t_window: float) -> np.ndarray:
        """
        Coarsen raw bins to match a given time window.

        Parameters
        ----------
        raw_counts : 1-D int array from generate()
        t_window   : target bin width [s]  (e.g. 30e-9 for quED coincidence window)
        """
        n_avg = max(1, round(t_window / self.dt_raw))
        return self.boxcar(raw_counts, n_avg)

    def hbt_split(self, raw_counts: np.ndarray) -> tuple:
        """
        Simulate a 50:50 beamsplitter acting on the photon stream.

        Returns
        -------
        arm1, arm2 : two 1-D int arrays of the same length as raw_counts
        """
        arm1 = np.random.binomial(raw_counts, 0.5)
        arm2 = raw_counts - arm1
        return arm1, arm2

    def _g2_batch(self, n_bins: int, n_trials: int) -> np.ndarray:
        """
        Internal. Engine for simulating g^(2)(0) over many trials.

        Parameters
        ----------
        n_bins   : number of time bins per trial
        n_trials : number of independent trials

        Returns
        -------
        g2 : float array of shape (n_trials,), NaN where N_1 or N_2 == 0
        """
        mu = self.mu_raw

        N_total = np.random.poisson(n_bins * mu,       size=n_trials)
        N1      = np.random.binomial(N_total, 0.5)
        N2      = N_total - N1

        K   = np.random.poisson(n_bins * mu**2 / 2,   size=n_trials)
        N12 = np.random.binomial(K, 0.5)

        valid  = (N1 > 0) & (N2 > 0)
        g2     = np.full(n_trials, np.nan)
        g2[valid] = (N12[valid] * n_bins) / (N1[valid] * N2[valid])
        return g2

    def g2_zero(self, T_run: float) -> float:
        """
        Simulate one measurement of g^(2)(0).

        Parameters
        ----------
        T_run : run duration [s]
        """
        return float(self._g2_batch(int(T_run / self.dt_raw), 1)[0])

    def g2_distribution(self, T_run: float, n_trials: int = 500) -> dict:
        """
        Draw g^(2)(0) for n_trials independent runs.

        Parameters
        ----------
        T_run    : run duration per trial [s]
        n_trials : number of independent trials

        Returns
        -------
        dict with keys 'g2_values', 'mean', 'std'
        """
        g2_vals = self._g2_batch(int(T_run / self.dt_raw), n_trials)
        g2_vals = g2_vals[~np.isnan(g2_vals)]
        return {
            'g2_values': g2_vals,
            'mean':      float(g2_vals.mean()),
            'std':       float(g2_vals.std()),
        }

    def plot_g2_distribution(
        self,
        T_run: float,
        n_trials: int = 500,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Histogram of g^(2)(0) values over many simulated runs.
        """
        result    = self.g2_distribution(T_run, n_trials)
        g2_vals   = result['g2_values']
        mu, sigma = result['mean'], result['std']

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        ax.hist(g2_vals, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(mu,  color='crimson', linewidth=2,
                   label=f'Mean = {mu:.4f}')
        ax.axvline(1.0, color='gray',   linewidth=1.5, linestyle='--',
                   label='Coherent light limit = 1')
        ax.set_xlabel('$g^{(2)}(0)$')
        ax.set_ylabel('Counts')
        ax.set_title(
            f'Distribution of $g^{{(2)}}(0)$ over {n_trials} trials\n'
            f'Flux = {self.flux:.2e} ph/s,  $\\mu$ = {self.mu_raw:.4f},  '
            f'$T_{{\\rm run}}$ = {T_run:.2f} s\n'
            f'Mean = {mu:.4f},  Std = {sigma:.4f}'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def heralded_g2_zero(
        self,
        T_run: float,
        simulator: 'SPDCSimulator',
        P_pump: float = None,
    ) -> float:
        """
        Simulate the heralded g^(2)_H(0) measurement.

        Parameters
        ----------
        T_run     : run duration [s]
        simulator : SPDCSimulator instance
        P_pump    : pump power [W] (defaults to simulator.p.P_max)
        """
        if P_pump is None:
            P_pump = simulator.p.P_max

        n_bins  = int(T_run / self.dt_raw)
        mu_pair = simulator.pair_rate(P_pump) * self.dt_raw
        eta_h   = simulator.p.eta_1

        # sparse correlated pair events ---
        N_pairs = int(np.random.poisson(n_bins * mu_pair))
        if N_pairs == 0:
            return np.nan

        pair_bins = np.random.randint(0, n_bins, size=N_pairs)

        # herald --> each pair fires channel 0 with probability eta_h
        herald_mask = np.random.rand(N_pairs) < eta_h

        # signal arm: 50:50 HBT split of the same pairs
        ch1_mask = np.random.rand(N_pairs) < 0.5 # (true=ch1, false=ch2)

        # sparse uncorrelated background ---
        N_bg    = int(np.random.poisson(n_bins * self.mu_raw))
        bg_bins = np.random.randint(0, n_bins, size=N_bg) if N_bg > 0 else np.array([], dtype=int)
        bg_ch1  = np.random.rand(N_bg) < 0.5

        # occupied bin sets per channel
        herald_bins = np.unique(pair_bins[herald_mask])
        ch1_pair    = np.unique(pair_bins[ ch1_mask])
        ch2_pair    = np.unique(pair_bins[~ch1_mask])
        ch1_bg      = np.unique(bg_bins[bg_ch1])   if N_bg > 0 else np.array([], dtype=int)
        ch2_bg      = np.unique(bg_bins[~bg_ch1])  if N_bg > 0 else np.array([], dtype=int)
        ch1_bins    = np.unique(np.concatenate([ch1_pair, ch1_bg]))
        ch2_bins    = np.unique(np.concatenate([ch2_pair, ch2_bg]))

        N1   = len(herald_bins)
        N12  = len(np.intersect1d(herald_bins, ch1_bins, assume_unique=True))
        N13  = len(np.intersect1d(herald_bins, ch2_bins, assume_unique=True))
        ch12 = np.intersect1d(ch1_bins, ch2_bins, assume_unique=True)
        N123 = len(np.intersect1d(herald_bins, ch12, assume_unique=True))

        if N12 == 0 or N13 == 0:
            return np.nan

        return (N123 * N1) / (N12 * N13)

    def print_summary(self):
        niceprint(
            f"**PhotonBeamSimulator** <br>"
            f"Flux: {self.flux:.3e} ph/s <br>"
            f"$\\Delta t_{{\\rm raw}}$: {self.dt_raw:.3e} s <br>"
            f"$\\mu$ per raw bin: {self.mu_raw:.4f} <br>"
            f"Bins per second: {1/self.dt_raw:.3e}"
        )


class BellInequalitySimulator:
    """
    Simulates the CHSH Bell inequality experiment.
    """

    CHSH_ANGLES = {
        'singlet': {
            'alpha': (0.0,        np.pi / 4),
            'beta':  (-np.pi / 8, 5 * np.pi / 8),
        },
        'triplet': {
            'alpha': (0.0,        np.pi / 4),
            'beta':  (np.pi / 8, -np.pi / 8),
        },
    }

    def __init__(
        self,
        simulator: 'SPDCSimulator',
        state: str = 'singlet',
        visibility: float = 1.0,
    ):
        """
        Parameters
        ----------
        simulator  : SPDCSimulator — provides lambda_spdc, Rcoin_W(), SPDC params
        state      : 'singlet' for (|HH> - |VV>) / sqrt(2)
                     'triplet' for (|HH> + |VV>) / sqrt(2)
        visibility : fringe visibility in [0, 1].
                     P_obs = V * P_QM + (1 - V) * 1/4  (uniform background)
        """
        if state not in ('singlet', 'triplet'):
            raise ValueError("state must be 'singlet' or 'triplet'")
        self.sim        = simulator
        self.state      = state
        self.visibility = visibility
        self.angles     = self.CHSH_ANGLES[state]
        self._lam       = simulator.lambda_spdc

    def _arm_amplitudes(self, angle: float) -> tuple:
        """
        Internal. Calculates transmission amplitudes through a polarizer at angle for H and V.
        
        Returns
        -------
        t_H : complex amplitude for |H> transmitted through polarizer at angle
        t_V : complex amplitude for |V> transmitted through polarizer at angle
        """
        H_in = np.array([1.0, 0.0], dtype=complex)
        V_in = np.array([0.0, 1.0], dtype=complex)

        circ = OpticalCircuit(wavelength=self._lam)
        circ.add_polarizer(angle=angle)

        E_out_H = circ.propagate_polarization(H_in)
        E_out_V = circ.propagate_polarization(V_in)

        H_axis = np.array([np.cos(angle), np.sin(angle)], dtype=complex)
        t_H = complex(H_axis @ E_out_H)
        t_V = complex(H_axis @ E_out_V)
        return t_H, t_V

    def _joint_probs(self, alpha: float, beta: float) -> tuple:
        """
        Internal. Compute P_HH, P_HV, P_VH, P_VV.
        """
        H_in = np.array([1.0, 0.0], dtype=complex)
        V_in = np.array([0.0, 1.0], dtype=complex)

        # arm a
        circ_a  = OpticalCircuit(wavelength=self._lam)
        circ_a.add_polarizer(angle=alpha)
        E_aH    = circ_a.propagate_polarization(H_in)
        E_aV    = circ_a.propagate_polarization(V_in)
        H_a     = np.array([ np.cos(alpha),  np.sin(alpha)], dtype=complex)
        V_a     = np.array([-np.sin(alpha),  np.cos(alpha)], dtype=complex)
        t_aH, t_aV = complex(H_a @ E_aH), complex(H_a @ E_aV)
        r_aH, r_aV = complex(V_a @ E_aH), complex(V_a @ E_aV)

        # arm b
        circ_b  = OpticalCircuit(wavelength=self._lam)
        circ_b.add_polarizer(angle=beta)
        E_bH    = circ_b.propagate_polarization(H_in)
        E_bV    = circ_b.propagate_polarization(V_in)
        H_b     = np.array([ np.cos(beta),  np.sin(beta)], dtype=complex)
        V_b     = np.array([-np.sin(beta),  np.cos(beta)], dtype=complex)
        t_bH, t_bV = complex(H_b @ E_bH), complex(H_b @ E_bV)
        r_bH, r_bV = complex(V_b @ E_bH), complex(V_b @ E_bV)

        sign = -1.0 if self.state == 'singlet' else +1.0
        s2   = np.sqrt(2.0)

        A_HH = (t_aH * t_bH + sign * t_aV * t_bV) / s2
        A_HV = (t_aH * r_bH + sign * t_aV * r_bV) / s2
        A_VH = (r_aH * t_bH + sign * r_aV * t_bV) / s2
        A_VV = (r_aH * r_bH + sign * r_aV * r_bV) / s2

        P_HH, P_HV = abs(A_HH)**2, abs(A_HV)**2
        P_VH, P_VV = abs(A_VH)**2, abs(A_VV)**2

        V, bg = self.visibility, 0.25
        return (
            V * P_HH + (1 - V) * bg,
            V * P_HV + (1 - V) * bg,
            V * P_VH + (1 - V) * bg,
            V * P_VV + (1 - V) * bg,
        )

    def sample_counts(
        self,
        alpha: float,
        beta: float,
        P_pump: float = None,
        T_acq: float = 15.0,
    ) -> dict:
        """
        Simulate Poisson coincidence counts at one (alpha, beta) setting.

        Parameters
        ----------
        alpha, beta : polarizer angles [rad]
        P_pump      : pump power [W] (defaults to sim.p.P_max)
        T_acq       : acquisition time per setting [s]
        """
        if P_pump is None:
            P_pump = self.sim.p.P_max

        P_HH, P_HV, P_VH, P_VV = self._joint_probs(alpha, beta)
        R_coin = float(self.sim.Rcoin_W(np.array([P_pump]))[0])
        N_mean = R_coin * T_acq

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

    def run_chsh(self, P_pump: float = None, T_acq: float = 15.0) -> dict:
        """
        Simulate the full 4-setting CHSH experiment.
        
        Returns
        -------
        dict with per-setting results, S, sigma_S, S_theory, violates
        """
        if P_pump is None:
            P_pump = self.sim.p.P_max

        a1, a2 = self.angles['alpha']
        b1, b2 = self.angles['beta']

        chsh_settings = [
            (a1, b1, +1.0),
            (a1, b2, +1.0),
            (a2, b1, +1.0),
            (a2, b2, -1.0),
        ]

        results = {}
        S, var_S = 0.0, 0.0

        for alpha, beta, sign in chsh_settings:
            key = f'a={np.rad2deg(alpha):.1f}°, b={np.rad2deg(beta):.1f}°'
            r   = self.sample_counts(alpha, beta, P_pump, T_acq)
            results[key] = r
            S     += sign * r['E']
            var_S += r['sigma_E'] ** 2

        return {
            'settings':  results,
            'S':         S,
            'sigma_S':   np.sqrt(var_S),
            'S_theory':  self.visibility * 2.0 * np.sqrt(2.0),
            'violates':  abs(S) > 2.0,
        }

    def plot_correlation_sweep(
        self,
        fixed_alpha: float = 0.0,
        n_points: int = 73,
        P_pump: float = None,
        T_acq: float = 15.0,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot E(alpha, beta) vs. beta with alpha fixed.

        Parameters
        ----------
        fixed_alpha : fixed polarizer angle [rad]
        n_points    : resolution of smooth theory curve
        P_pump      : pump power [W]
        T_acq       : acquisition time per simulated data point [s]
        """
        if P_pump is None:
            P_pump = self.sim.p.P_max

        betas_theory = np.linspace(0, 2 * np.pi, n_points)
        E_theory = np.array([
            np.dot(self._joint_probs(fixed_alpha, b), [1, -1, -1, 1])
            for b in betas_theory
        ])

        betas_data = np.deg2rad(np.arange(0, 361, 5))
        E_data, E_err = [], []
        for b in betas_data:
            r = self.sample_counts(fixed_alpha, b, P_pump, T_acq)
            E_data.append(r['E'])
            E_err.append(r['sigma_E'])
        E_data, E_err = np.array(E_data), np.array(E_err)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.plot(np.rad2deg(betas_theory), E_theory,
                color='steelblue', linewidth=2, label='Theory', zorder=2)
        ax.errorbar(np.rad2deg(betas_data), E_data, yerr=E_err,
                    fmt='o', color='crimson', markersize=4,
                    elinewidth=0.8, capsize=2, label='Simulated data', zorder=3)

        b1, b2 = self.angles['beta']
        for b, lbl in [(b1, r'$\beta_1$'), (b2, r'$\beta_2$')]:
            ax.axvline(np.rad2deg(b % (2 * np.pi)),
                       color='gray', linestyle='--', linewidth=1.2, label=lbl)

        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlabel(r'$\beta$ (degrees)')
        ax.set_ylabel(r'$E(\alpha,\, \beta)$')
        ax.set_title(
            f'Correlation sweep — $|\\psi_{{{self.state[0]}}}\\rangle$,  '
            f'$\\alpha = {np.rad2deg(fixed_alpha):.1f}^\\circ$,  '
            f'$V = {self.visibility:.2f}$'
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        return ax

    def print_chsh_result(self, result: dict):
        label = 's' if self.state == 'singlet' else 't'
        niceprint(
            f"**CHSH Result — $|\\psi_{{{label}}}\\rangle$,  "
            f"$V = {self.visibility:.2f}$**", 4
        )
        niceprint(
            f"$\\langle S \\rangle = "
            f"{result['S']:.4f} \\pm {result['sigma_S']:.4f}$ <br>"
            f"QM prediction: $2\\sqrt{{2}}\\cdot V = {result['S_theory']:.4f}$ <br>"
            f"Classical bound: $|\\langle S \\rangle| \\leq 2$ <br>"
            f"Violates Bell inequality: {'**Yes**' if result['violates'] else 'No'}"
        )




