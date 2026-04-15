# orenge

Short for "overengineering"! This is my personal utility package for quantum computing and information. Built for use in Jupyter notebooks.

## Installation

```bash
pip install git+https://github.com/emmabacarra/personal-utils.git
```

## Import

```python
import orenge

# or import specific modules:
from orenge.general import *
from orenge.optics.simulators import GaussianBeamTool
```

If the coloring isn't recognized in VS Code, add this to your workspace settings:

```json
"python.analysis.extraPaths": [
			"[path to personal-utils folder]"
		]
```

---

## Modules

### `orenge.general`

Display and formatting utilities for Jupyter notebooks. Designed for clean LaTeX/Markdown rendering of quantum states, matrices, and expressions.

**Functions:**

- `niceprint(s, header_size, method)` — Renders a string as formatted Markdown or LaTeX in Jupyter. Supports headers, math, and multi-line LaTeX environments.
- `nicetable(headers, rows, align)` — Renders a Markdown table with optional column alignment. Compatible with LaTeX math inside cells.
- `cleandisp(qobj, format, return_str, ...)` — Formats a QuTiP `Qobj` or numpy array as a clean LaTeX expression. Automatically extracts common scalar factors, simplifies fractions and square roots, and optionally renders in Dirac notation.

---

### `orenge.sensing`

Quantum sensing simulations, including trajectory-based particle detection and stabilizer code analysis.

**Classes:**

- `TrajectorySimulator(num_qubits, theta, entangled_groups)` — Simulates qubit sensor arrays under particle trajectories. Handles entangled sensor groups, gate sequence application, Bloch sphere visualization, and state display.

**Functions:**

- `real_imag_bloch(state, bloch_obj)` — Adds real and imaginary Bloch vectors for a given qubit state to a QuTiP `Bloch` object.
- `find_syndrome_paulis(psi, trajectories, theta, n)` — Finds the minimal set of multi-qubit Pauli observables that uniquely identify each trajectory via syndrome measurement. Requires `theta = π/2`.
- `decode_syndrome(psi, trajectories, theta, n, ...)` — Builds a `pytket` syndrome measurement circuit using ancilla qubits and Pauli controlled gates. Returns the circuit, syndrome map, and Pauli strings.

---

### `orenge.nmr_sims`

NMR gate simulations and spectral analysis tools.

**Classes:**

- `NMRGates(J)` — Constructs NMR gate unitaries parameterized by J-coupling strength.

**Functions:**

- `locate_peaks(data, frequencies, pad)` — Detects and returns the top spectral peaks from NMR frequency-domain data, along with a frequency window around them.

---

### `orenge.optimizer`

A unified optimizer interface wrapping `scipy.optimize`, supporting a wide range of gradient-based, derivative-free, constrained, and global optimization methods.

**Supported methods include:** `BFGS`, `L-BFGS-B`, `Newton`, `CG`, `Adam`, `Nelder-Mead`, `Powell`, `COBYLA`, `DE` (Differential Evolution), `DA` (Dual Annealing), `SLSQP`, `trust-constr`, `Aug-Lag`, `LM` (Levenberg-Marquardt), `LS-TRF`, `Gauss-Newton`, `CMA-ES`, `Basin`, `SHGO`, and more.

---

## Subpackage: `orenge.optics`

Tools for classical and quantum optical system simulation.

---

### `orenge.optics.components`

Optical component classes implementing ABCD ray transfer matrices, Jones matrices for polarization, and quantum operators.


**Classes:**

- `FreeSpace(distance)` — Free-space propagation ABCD matrix.
- `ThinLens(focal_length)` — Thin lens ABCD matrix (converging if `f > 0`, diverging if `f < 0`).
- `CurvedMirror(radius)` — Curved mirror ABCD matrix with radius of curvature `R`.
- `FlatMirror()` — Flat mirror (identity ABCD matrix).
- `DielectricInterface(n1, n2)` — Planar dielectric interface between two refractive indices.
- `BeamSplitter(input_a, input_b)` — 50:50 beam splitter quantum operator (unitary transformation on path modes).
- `Mirror(path_index)` — Quantum mirror with π phase shift on specified path(s).
- `Polarizer(angle)` — Linear polarizer Jones matrix rotated to angle `θ`.
- `HalfWavePlate(fast_axis_angle, retardance_deviation)` — HWP Jones matrix with optional retardance error.
- `QuarterWavePlate(fast_axis_angle, retardance_deviation)` — QWP Jones matrix with optional retardance error.
- `PolarizingBeamSplitter(port)` — PBS Jones matrix for transmitted (H) or reflected (V) port.
- `RetroReflectiveMirror()` — Applies complex conjugation to a Jones vector (flips circular polarization handedness).

---

### `orenge.optics.params`

Dataclasses for parameterizing optical systems.


**Dataclasses:**

- `CavityGeometry` — Bow-tie cavity geometry: width, height, concave mirror focal length, mirror reflectivities, wavelength.
- `LaserBeam` — Beam waist, waist location, and wavelength.
- `Telescope` — Two-lens telescope focal lengths.
- `PiezoActuator` — Displacement per volt, voltage amplitude, frequency, and DC offset.
- `Photodetector` — Responsivity, load resistance, and amplifier gain.
- `SPDC` — SPDC source parameters: pump wavelength, pair generation efficiency, detection efficiencies, pump power, and laser diode operating currents.
- `BellMeasurement` — One row of a CHSH coincidence dataset: polarizer angles, singles counts, raw coincidences, and accidentals. Includes a `N_net` property for background-subtracted coincidences.

---

### `orenge.optics.simulators`

Simulation tools for Gaussian beam propagation, optical circuits, bow-tie cavities, SPDC sources, and photon beams.


**Classes:**

- `GaussianBeamTool(wavelength)` — Computes Gaussian beam parameters (q-parameter, waist, Rayleigh range) through ABCD systems.
- `OpticalCircuit` — Chains optical components and computes the overall system ABCD matrix and Jones matrix.
- `BowTieCavity(geometry)` — Models a bow-tie ring cavity: round-trip ABCD matrix, stability criterion, mode size at each mirror.
- `SPDCSimulator(params)` — Simulates SPDC pair generation: count rates, coincidences, and accidentals as a function of pump power and detection efficiency.
- `PhotonBeamSimulator` — Simulates photon beam propagation through an optical circuit.

---

### `orenge.optics.analyzers`

Optimization and analysis tools for optical system design and quantum state tomography.


**Classes:**

- `FiberModeMatchOptimizer(laser, telescope, f_collimator, w_fiber, d0)` — Optimizes telescope and collimator placement for maximum fiber coupling efficiency, using beam overlap integrals.

---

### `orenge.optics.helpers`

Low-level polarization helpers and density matrix tools, primarily used internally by `quTools`.


**Includes:** Pauli matrices, projection kets (H, V, D, A, R, L), quED tomography waveplate settings, single- and two-photon density matrix reconstruction, Stokes parameter contrast, and eigendecomposition utilities.

---

### `orenge.optics.quTools`

High-level API modeling the quTools quED entanglement demonstrator.


**Class:**

- `quED(spdc_params)` — Top-level instrument model with the following subcomponents:
  - `quED.pump` (`PumpBlock`) — 405 nm pump diode, waveplate, and BBO crystal.
  - `quED.block` (`DetectionBlock`) — Two-arm detection block.
    - `quED.block.arm1`, `quED.block.arm2` (`DetectionArm`) — Each arm contains a QWP (optional), linear polarizer, and APD.
  - `quED.SPDC` (`SPDCInterface`) — Count rates and power as a function of operating current.
  - `quED.stats` (`PhotonStatsInterface`) — Second-order coherence `g²(0)` and photon statistics.

---

## Dependencies

| Package | Purpose |
|:---|:---|
| `numpy` | Numerical arrays and linear algebra |
| `sympy` | Symbolic math and LaTeX formatting |
| `qutip` | Quantum state representation and simulation |
| `ipython` | Jupyter display utilities |
| `scipy` | Optimization, signal processing |
| `matplotlib` | Plotting |
| `pytket` | Quantum circuit construction (sensing module) |
| `tqdm` | Progress bars (optics helpers) |
| `pandas` | Data handling (NMR module) |

Optional (dev):

```bash
pip install "git+https://github.com/emmabacarra/personal-utils.git#egg=orenge[dev]"
```

Installs `jupyter` and `matplotlib` as additional dev dependencies.