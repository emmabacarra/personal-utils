from .general import *

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qutip.qip.circuit import QubitCircuit
from pytket.circuit import Circuit as _TketCircuit
from pytket.circuit.display import render_circuit_jupyter as _draw

from itertools import combinations
from scipy.optimize import linprog
import scipy.sparse as sp
from typing import Literal



def real_imag_bloch(state, bloch_obj):
    bloch_vec = [expect(sigmax(), state), 
                 expect(sigmay(), state), 
                 expect(sigmaz(), state)]
    
    real_vec = [np.real(bloch_vec[0]), np.real(bloch_vec[1]), np.real(bloch_vec[2])]
    bloch_obj.add_vectors([real_vec])
    bloch_obj.vector_color = ['b']
    
    imag_vec = [np.imag(bloch_vec[0]), np.imag(bloch_vec[1]), np.imag(bloch_vec[2])]
    if np.linalg.norm(imag_vec) > 1e-6:
        bloch_obj.add_vectors([imag_vec])
        bloch_obj.vector_color = ['b', 'r']
    else:
        bloch_obj.vector_color = ['b']


class TrajectorySimulator:

    def __init__(self, num_qubits, theta=np.pi/2, entangled_groups=None):
        """
        Initialize the trajectory simulator.

        Parameters
        ----------
        num_qubits : int
            Total number of qubits in the system.
        theta : float
            Rotation angle for the particle-sensor interaction (default: π/2).
        entangled_groups : list of tuples
            Pairs (control, target) defining which qubits are entangled.
            Each pair maps to one column in the sensor array.
            Group-relative index 0 = control, 1 = target within each pair.
        """
        self.num_qubits = num_qubits
        self.theta = theta
        self.theta_label = self._format_theta_label(theta)
        self.entangled_groups = entangled_groups
        self.psi_initial = tensor(*[basis(2, 0) for _ in range(num_qubits)])

    def _format_theta_label(self, theta):
        """
        Internal.
        Create a LaTeX label for theta.
        """
        if np.isclose(theta, np.pi/2):
            return r'$\frac{\pi}{2}$'
        elif np.isclose(theta, np.pi/4):
            return r'$\frac{\pi}{4}$'
        elif np.isclose(theta, np.pi):
            return r'$\pi$'
        else:
            return f'${theta:.3f}$'

    def _identify_entangled_groups(self):
        """
        Internal.
        Find connected components of qubits via depth-first search on entangled_groups.

        Returns
        -------
        groups : list of lists
            Each inner list is a set of qubit indices that are entangled together,
            sorted in ascending order. Groups are sorted by their minimum qubit index.
        """
        graph = {i: set() for i in range(self.num_qubits)}
        for control, target in self.entangled_groups:
            graph[control].add(target)
            graph[target].add(control)

        visited = set()
        groups = []

        def dfs(node, group):
            visited.add(node)
            group.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, group)

        for i in range(self.num_qubits):
            if i not in visited:
                group = []
                dfs(i, group)
                groups.append(sorted(group))

        groups.sort(key=lambda g: min(g))
        return groups

    def _apply_gate_sequence(self, circuit, gate_sequence):
        """
        Internal.
        Apply a gate sequence to every entangled group, mapping group-relative
        indices to absolute qubit indices.

        Parameters
        ----------
        circuit : QubitCircuit
            The circuit to add gates to.
        gate_sequence : list of tuples
            Each tuple is (gate_name, params_dict) where indices in 'controls'
            and 'targets' are relative to the group (0 = first qubit, 1 = second, ...).

        Supported gates
        ---------------
        'CNOT' : params with 'controls' and 'targets'
        'RZ'   : params with 'targets', optional 'theta' and 'theta_label'
        'H'    : params with 'targets'
        'X'    : params with 'targets'
        'Y'    : params with 'targets'
        'Z'    : params with 'targets'
        """
        groups = self._identify_entangled_groups()

        for group in groups:
            for gate_name, params in gate_sequence:
                abs_params = self._map_group_indices_to_absolute(params, group)

                if gate_name == 'CNOT':
                    for control, target in zip(abs_params['controls'],
                                               abs_params['targets']):
                        circuit.add_gate("CNOT", controls=[control], targets=[target])

                elif gate_name == 'RZ':
                    theta = abs_params.get('theta', self.theta)
                    theta_label = abs_params.get('theta_label', self.theta_label)
                    for target in abs_params['targets']:
                        circuit.add_gate("RZ", targets=[target],
                                         arg_value=theta, arg_label=theta_label)

                elif gate_name == 'H':
                    for target in abs_params['targets']:
                        circuit.add_gate("H", targets=[target])

                elif gate_name == 'X':
                    for target in abs_params['targets']:
                        circuit.add_gate("X", targets=[target])

                elif gate_name == 'Y':
                    for target in abs_params['targets']:
                        circuit.add_gate("Y", targets=[target])

                elif gate_name == 'Z':
                    for target in abs_params['targets']:
                        circuit.add_gate("Z", targets=[target])

                else:
                    raise ValueError(f"Unsupported gate: {gate_name}")

    def _map_group_indices_to_absolute(self, params, group):
        """
        Internal.
        Map group-relative indices to absolute qubit indices.

        Parameters
        ----------
        params : dict
            Keys 'controls' and 'targets' contain group-relative indices.
            Any other keys (e.g. 'theta', 'theta_label') are passed through unchanged.
        group : list
            Absolute qubit indices for this group, in sorted order.

        Returns
        -------
        absolute_params : dict
        """
        absolute_params = {}
        for key, value in params.items():
            if key in ['controls', 'targets']:
                absolute_params[key] = [group[i] for i in value]
            else:
                absolute_params[key] = value
        return absolute_params

    def _apply_trajectory(self, circuit, trajectory_qubits):
        """
        Internal.
        Apply RZ(theta) to each qubit in the trajectory.

        Parameters
        ----------
        trajectory_qubits : list, tuple, or None
            Absolute qubit indices hit by the particle. None or empty = no particle.
        """
        if trajectory_qubits is None or len(trajectory_qubits) == 0:
            return
        for qubit in trajectory_qubits:
            circuit.add_gate("RZ", targets=[qubit],
                             arg_value=self.theta, arg_label=self.theta_label)

    def simulate_trajectory(self, trajectory_qubits, prep_sequence, decoder_sequence,
                            trajectory_name=None, show_bloch=True, show_state=True):
        """
        Simulate a single trajectory through the quantum system.

        Parameters
        ----------
        trajectory_qubits : tuple, list, or None
            Absolute qubit indices hit by the particle. None = no trajectory (T0).
        prep_sequence : list of tuples
            Gate sequence applied to every entangled group during state preparation.
            Format: [(gate_name, params_dict), ...] with group-relative indices.
        decoder_sequence : list of tuples
            Gate sequence applied to every entangled group during decoding.
            Same format as prep_sequence.
        trajectory_name : str, optional
            Label for display.
        show_bloch : bool
        show_state : bool

        Returns
        -------
        psi_output : Qobj
        """
        circuit = QubitCircuit(self.num_qubits, num_cbits=0)

        # Stage 1: state preparation
        self._apply_gate_sequence(circuit, prep_sequence)

        # Stage 2: particle interaction
        self._apply_trajectory(circuit, trajectory_qubits)

        # Stage 3: decoding
        self._apply_gate_sequence(circuit, decoder_sequence)

        psi_output = circuit.run(self.psi_initial)

        if trajectory_name or (trajectory_qubits and len(trajectory_qubits) > 0):
            niceprint('---')
            if trajectory_name:
                niceprint(f'**Trajectory: {trajectory_name}**', 3)
            if trajectory_qubits and len(trajectory_qubits) > 0:
                niceprint(f'Qubits hit: {[q+1 for q in trajectory_qubits]} (1-indexed)')
            else:
                niceprint('No trajectory applied')

        if show_state:
            cleandisp(psi_output, format='Dirac')

        if show_bloch:
            self._visualize_bloch(psi_output, trajectory_name)

    def simulate_all_trajectories(self, trajectories, prep_sequence, decoder_sequence,
                                  show_bloch=True, show_state=True):
        """
        Simulate multiple trajectories.

        Parameters
        ----------
        trajectories : dict or list
            If dict: {name: qubit_indices}, e.g. {'T1': (0,1), 'T2': (1,2)}.
            If list: [qubit_indices, ...].
            Use None or empty tuple for no trajectory (T0).
        prep_sequence : list of tuples
            Passed to simulate_trajectory for every trajectory.
        decoder_sequence : list of tuples
            Passed to simulate_trajectory for every trajectory.
        show_bloch : bool
        show_state : bool

        Returns
        -------
        results : dict
            Maps trajectory names to output states.
        """

        if isinstance(trajectories, dict):
            traj_items = trajectories.items()
        else:
            traj_items = [(f'T{i+1}', traj) for i, traj in enumerate(trajectories)]

        for name, qubits in traj_items:
            self.simulate_trajectory(
                qubits,
                prep_sequence=prep_sequence,
                decoder_sequence=decoder_sequence,
                trajectory_name=name,
                show_bloch=show_bloch,
                show_state=show_state,
            )

    def get_initial_state(self, prep_sequence, show=True):
        """
        Get and optionally display the state after preparation (before trajectory).

        Parameters
        ----------
        prep_sequence : list of tuples
            Gate sequence for state preparation, same format as in simulate_trajectory.
        show : bool

        Returns
        -------
        psi : Qobj
        """
        circuit = QubitCircuit(self.num_qubits, num_cbits=0)
        self._apply_gate_sequence(circuit, prep_sequence)
        psi = circuit.run(self.psi_initial)

        if show:
            niceprint('State after preparation:', 3)
            cleandisp(psi, format='Dirac')

        return psi

    def _visualize_bloch(self, psi_output, title=None):
        groups = self._identify_entangled_groups()

        ncols = len(groups)
        nrows = max(len(g) for g in groups)

        bg_colors = ["#fafafa", "#e1eefa"]

        fig = plt.figure(layout='constrained', figsize=(3.5*ncols + 1, 3.5*nrows))
        fig.get_layout_engine().set(w_pad=0.05, h_pad=0.1, wspace=0.1, hspace=0.1)

        if title:
            fig.suptitle(title, fontsize=18, fontweight='bold')

        margin_size = 0.05
        width_ratios = [margin_size] + [1.0] * ncols + [margin_size]
        gs = fig.add_gridspec(1, ncols + 2, width_ratios=width_ratios)

        left_anchor = fig.add_subplot(gs[0, 0])
        left_anchor.axis('off')
        right_anchor = fig.add_subplot(gs[0, -1])
        right_anchor.axis('off')

        subfigs = []
        for i in range(ncols):
            subfigs.append(fig.add_subfigure(gs[0, i+1]))
        if ncols == 1:
            subfigs = [subfigs]

        for col_idx, (subfig, group) in enumerate(zip(subfigs, groups)):
            bg_color = bg_colors[col_idx % 2]
            subfig.patch.set_facecolor(bg_color)

            subfig.suptitle(f'Group {col_idx+1}', fontsize=16, fontweight='bold')
            qubit_subfigs = subfig.subfigures(len(group), 1)
            if len(group) == 1:
                qubit_subfigs = [qubit_subfigs]

            for qubit_subfig, qubit_idx in zip(qubit_subfigs, group):
                qubit_subfig.patch.set_facecolor(bg_color)
                qubit_subfig.suptitle(f'Qubit {qubit_idx+1}', fontsize=13)

                ax = qubit_subfig.subplots(1, 1, subplot_kw={'projection': '3d'})
                ax.set_facecolor(bg_color)

                b = Bloch(axes=ax)
                state_qi = psi_output.ptrace(qubit_idx)
                real_imag_bloch(state_qi, b)
                b.render()

        plt.show()




# ──────────────────────────────────────────────────────────────────
def _cyc_group(n):
    """
    Generate G = Z_n as a list of n permutation tuples (no n! search).
    z^j sends qubit i --> (i+j) mod n.
    """
    return [tuple((i + j) % n for i in range(n)) for j in range(n)]


def sym_minInt(n):
    """
    minimum interaction strength theta for symmetric trajectory sets
    """
    return ( (n-1)*np.pi )/n


def cyc_minInt(n, m):
    """
    minimum interaction strength theta for cyclic trajectory sets
    """
    return np.arccos(-1  + np.ceil(n/(2*m))**(-1))


# ──────────────────────────────────────────────
#  Trajectory set generators
# ──────────────────────────────────────────────

def _generate_Tsym(n, m):
    """All C(n,m) subsets of size m — T_sym(n, m)."""
    return {
        f"T{i}": frozenset(c)
        for i, c in enumerate(combinations(range(n), m))
    }


def _generate_Tcyc(n, m):
    """n consecutive-index trajectories (cyclic) — T_cyc(n, m)."""
    return {
        f"T{j}": frozenset((j + k) % n for k in range(m))
        for j in range(n)
    }


def _parse_trajectories(traj_dict):
    """Convert trajectory dict (tuples or sets as values) to list of frozensets."""
    return [frozenset(v) for v in traj_dict.values()]





# ──────────────────────────────────────────────
#  Symmetry group
# ──────────────────────────────────────────────


def _sym_bitstring_orbits(n):
    """
    Orbits of bit-strings under Sigma_tilde_n: W_nu union W_{n-nu}.
    Proposition 11 of [PRA]. Groups all 2^n strings by weight — no BFS needed.
    """
    by_weight = [[] for _ in range(n + 1)]
    for i in range(2 ** n):
        s = tuple(int(b) for b in format(i, f"0{n}b"))
        by_weight[s.count(1)].append(s)

    orbits = []
    visited = set()
    for nu in range(n // 2 + 1):
        if nu in visited:
            continue
        comp = n - nu
        merged = by_weight[nu] + ([] if comp == nu else by_weight[comp])
        orbits.append(frozenset(merged))
        visited.update([nu, comp])
    return orbits

def _compute_bitstring_orbits_cyc(n):
    """
    Fast orbit computation for G = Z_n using integer bit-rotation.

    Returns (frozenset_orbits, int_array_orbits) where:
      - frozenset_orbits : list of frozenset of bit-string tuples (for display)
      - int_array_orbits : list of np.ndarray of integers (for fast matrix build)
    """
    mask    = (1 << n) - 1
    visited = np.zeros(2 ** n, dtype=bool)
    fs_orbits  = []
    int_orbits = []

    def rotate_right(x, j):
        j = j % n
        return ((x >> j) | (x << (n - j))) & mask

    def flip(x):
        return (~x) & mask

    for s0 in range(2 ** n):
        if visited[s0]:
            continue
        orbit_ints = set()
        for j in range(n):
            r = rotate_right(s0, j)
            orbit_ints.add(r)
            orbit_ints.add(flip(r))
        for idx in orbit_ints:
            visited[idx] = True

        int_orbits.append(np.array(sorted(orbit_ints), dtype=np.int64))
        fs_orbits.append(frozenset(
            tuple(int(b) for b in format(i, f"0{n}b"))
            for i in orbit_ints
        ))

    return fs_orbits, int_orbits

# ──────────────────────────────────────────────
#  orbits  (Propositions 11, 12, 16 of [PRA])
# ──────────────────────────────────────────────



def _compute_traj_pair_orbits(trajectories, perm_group):
    """
    Partition T^2 = T x T into orbits under G~.

    G~ acts on (T, T') by permuting qubit indices (G) and/or swapping T <-> T' (X^n).
    The diagonal orbit {(T, T)} is placed first at index mu=0 per [PRA] convention.
    """
    all_pairs = [(T, Tp) for T in trajectories for Tp in trajectories]
    visited = set()
    diagonal_orbit = None
    other_orbits = []

    for T0, Tp0 in all_pairs:
        key0 = (T0, Tp0)
        if key0 in visited:
            continue
        orbit = set()
        stack = [key0]
        while stack:
            T, Tp = stack.pop()
            key = (T, Tp)
            if key in orbit:
                continue
            orbit.add(key)
            for perm in perm_group:
                nk = (frozenset(perm[q] for q in T), frozenset(perm[q] for q in Tp))
                if nk not in orbit:
                    stack.append(nk)
            swapped = (Tp, T)
            if swapped not in orbit:
                stack.append(swapped)
        visited |= orbit
        fs_orbit = frozenset(orbit)
        if T0 == Tp0:
            diagonal_orbit = fs_orbit
        else:
            other_orbits.append(fs_orbit)

    assert diagonal_orbit is not None
    return [diagonal_orbit] + other_orbits

def _sym_pair_orbits_fast(n, m, trajs):
    """
    One-representative-per-degree orbit list for T_sym^2 / Sigma_tilde_n.
    Degree mu = |T setminus T prime|. Proposition 12 of [PRA].
    Avoids enumerating all |T|^2 pairs.
    """
    M         = min(m, n - m) + 1
    T0        = frozenset(range(m))
    pair_reps = []
    for mu in range(M):
        Tp = frozenset(list(range(m - mu)) + list(range(m, m + mu)))
        pair_reps.append(frozenset([(T0, Tp)]))
    return pair_reps

def _build_ts_matrix_fast(n, bs_orbit_ints, pair_orbits, theta):
    """
    Vectorized A(theta) builder using integer bit-string representations.

    Parameters
    ----------
    bs_orbit_ints : list of np.ndarray
        Each array holds the integer indices of the bit-strings in that orbit.
    pair_orbits : list of frozensets
        Same structure as in _build_ts_matrix.
    """
    M = len(pair_orbits)
    N = len(bs_orbit_ints)
    A = np.zeros((M, N))

    for mu, pair_orbit in enumerate(pair_orbits):
        T, Tp   = next(iter(pair_orbit))
        neg_idx = list(T  - Tp)   # qubits contributing exp(-i*theta*(s_j-1/2))
        pos_idx = list(Tp - T)    # qubits contributing exp(+i*theta*(s_j-1/2))

        for nu, s_ints in enumerate(bs_orbit_ints):
            # s_ints : 1-D int array, each entry is a bit-string as an integer.
            # Bit j of integer x (MSB = qubit 0): (x >> (n-1-j)) & 1
            phase = np.ones(len(s_ints), dtype=complex)
            for j in neg_idx:
                bits   = (s_ints >> (n - 1 - j)) & 1
                phase *= np.exp(-1j * theta * (bits - 0.5))
            for j in pos_idx:
                bits   = (s_ints >> (n - 1 - j)) & 1
                phase *= np.exp(+1j * theta * (bits - 0.5))
            A[mu, nu] = phase.real.sum()

    return A

# ──────────────────────────────────────────────
#  A(theta)  (Theorem 4, Eq. 58 of [PRA])
# ──────────────────────────────────────────────



def _solve_ts_lp(A):
    """
    Check feasibility of A c = d, c >= 0, where d = [1, 0, ..., 0].
    Returns (is_feasible, c) using scipy HiGHS solver.
    """
    M, N = A.shape
    d = np.zeros(M)
    d[0] = 1.0
    result = linprog(
        c      = np.zeros(N),
        A_eq   = A,
        b_eq   = d,
        bounds = [(0.0, None)] * N,
        method = "highs",
    )
    if result.status == 0:
        return True, result.x
    return False, None


def _ts_state_from_lp(n, bs_orbits, c):
    """
    Build the 2^n state vector |psi> = sum_nu sqrt(c_nu) |nu> and normalise.
    Returns a QuTiP Qobj ket.
    """
    vec = np.zeros(2 ** n, dtype=complex)
    for orbit, coeff in zip(bs_orbits, c):
        amp = np.sqrt(max(coeff, 0.0))
        for s in orbit:
            idx = int("".join(map(str, s)), 2)
            vec[idx] += amp
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    return Qobj(vec, dims=[[2] * n, [1] * n])


def _sym_A_matrix(n, m, theta):
    """
    Build A(theta) analytically for T_sym(n, m) using Proposition 13 of [PRA],
    Eq. (71)
    """
    from math import comb
    N = n // 2 + 1
    M = min(m, n - m) + 1
    A = np.zeros((M, N))
    for mu in range(M):
        for nu in range(N):
            alpha = 1 if (n % 2 == 0 and nu == N - 1) else 2
            val   = 0.0
            for i in range(mu + 1):
                for ip in range(mu + 1):
                    inner = nu - (i + ip)
                    if 0 <= inner <= n - 2 * mu:
                        val += (comb(mu, i) * comb(mu, ip)
                                * comb(n - 2 * mu, inner)
                                * np.cos((i - ip) * theta))
            A[mu, nu] = alpha * val
    return A

def _sym_analytical_c(n, m, theta):
    """
    Analytical coefficients c_nu for T_sym(n, m=n/2), from Eq. (5) of [PRL]

    Skips the LP entirely. Returns None if m != n//2 or if infeasible at theta.
    Normalised so that A[0,:].c = 1 (the normalisation row).
    """
    if m != n // 2:
        return None
    N     = n // 2 + 1
    c_raw = np.array([(-1) ** (m - nu) * np.cos((m - nu) * theta)
                      for nu in range(N)])
    if np.any(c_raw < -1e-10):
        return None
    c_raw = np.maximum(c_raw, 0.0)
    A     = _sym_A_matrix(n, m, theta)
    norm  = A[0] @ c_raw
    if abs(norm) < 1e-12:
        return None
    return c_raw / norm




# ──────────────────────────────────────────────
#  ts solver
# ──────────────────────────────────────────────
class TSResult:
    """
    Container returned by solve_ts. Holds all orbit data and exposes a
    prepare() method for building state-preparation circuits.

    Backward-compatible: supports tuple unpacking
        theta_min, psi, orbits = solve_ts(n, m)
    via __iter__.

    Attributes
    ----------
    n         : int
    theta_min : float  (radians)
    psi       : Qobj ket, or None if infeasible
    orbits    : list of frozensets  (full G~-orbit partition)
    """

    def __init__(self, n, theta_min, theta, psi, orbits, c, traj_dict, no_hit_ok):   
        self.n         = n
        self.theta_min = theta_min
        self.theta     = theta
        self.psi       = psi
        self.orbits    = orbits
        self._c        = c   # LP coefficients — used to identify active orbits
        self.traj_dict  = traj_dict   # {label: frozenset of qubit indices}
        self.no_hit_ok  = no_hit_ok   # bool: |psi> ⊥ R^(T)|psi> for all T

    # ── Tuple-unpacking compatibility ────────────────────────────────────────
    def __iter__(self):
        return iter((self.theta_min, self.psi, self.orbits))

    # ── Active orbit lookup ──────────────────────────────────────────────────
    def _active_nus(self):
        """Return list of orbit indices nu where c_nu > 0."""
        if self._c is None:
            return []
        vec = self.psi.full().flatten()
        active = []
        for nu, orbit in enumerate(self.orbits):
            s0  = next(iter(orbit))
            idx = int("".join(map(str, s0)), 2)
            if abs(vec[idx]) > 1e-10:
                active.append(nu)
        return active

    # ── State preparation ────────────────────────────────────────────────────
    def prepare(self, nu=None, bitstrings=None, verbose=True):
        """
        Build and display a Clifford state-preparation circuit.

        Parameters
        ----------
        nu         : int, optional
            Orbit index from the solve_ts table. The bitstrings are taken
            directly from orbit omega_nu.
        bitstrings : list of str or list of tuples, optional
            Custom list of bitstrings, e.g. ["0011", "1100", "0110", "1001"].
            Displayed with label "custom".
        (default)  : if neither nu nor bitstrings is given, iterates through
            all active orbits.

        Returns
        -------
        If a single orbit/custom list:
            (circuit, psi)  — pytket Circuit and qutip Qobj ket
        If iterating all active orbits:
            list of (circuit, psi) tuples, one per active orbit
        """
        if nu is not None and bitstrings is not None:
            raise ValueError("Specify at most one of nu or bitstrings, not both.")

        if nu is not None:
            if nu >= len(self.orbits):
                raise ValueError(f"nu={nu} out of range (0 to {len(self.orbits)-1})")
            bs = ["".join(map(str, s)) for s in sorted(self.orbits[nu])]
            return _prepare_circuit(bs, self.n, label=f"$\\omega_{{{nu}}}$", verbose=verbose)

        if bitstrings is not None:
            return _prepare_circuit(bitstrings, self.n, label="custom", verbose=verbose)

        # Default: prepare full state (all orbits)
        return _prepare_circuit_general(
            self.psi.full().flatten(), self.n, label="full $|\\psi\\rangle$", verbose=verbose
        )
    
    # ── Decoder ──────────────────────────────────────────────────────────────
    def decode(self, outputs=None, no_hit=None, theta=None, verbose=True):
        """
        Build a decoder circuit U_decode satisfying:

            U_decode @ R^(T)(theta) |psi>  =  |output_T>

        for every trajectory T in the stored trajectory set.

        Parameters
        ----------
        outputs : dict, optional
            Custom output bitstring assignments {label: bitstring}.
            Keys must match result.traj_dict labels.
            Default: sequential T0->00..0, T1->00..01, T2->00..10, ...

        no_hit : bool or None
            Include no-hit (|psi> itself) as a decoder output.
            None = auto (include if result.no_hit_ok), True = force,
            False = exclude.

        theta : float, optional
            Interaction angle in radians for computing R^(T)(theta)|psi>.
            Default: result.theta (theta_min + eps). Override when the
            circuit uses a different interaction angle, e.g. result.theta_min.

        verbose : bool

        Returns
        -------
        (circuit, U_decode) : pytket Circuit, numpy ndarray (2^n, 2^n)
        """
        if self.psi is None:
            raise ValueError("No valid TS state found — LP was infeasible.")

        n          = self.n
        theta      = theta if theta is not None else self.theta
        traj_dict  = self.traj_dict

        # ── Resolve outputs ──────────────────────────────────────────────────────────────────
        stored_trajs = list(traj_dict.values())
        if outputs is None:
            stored_labels = list(traj_dict.keys())
            outputs = {lbl: f"{i:0{n}b}" for i, lbl in enumerate(stored_labels)}
            traj_for_decoder = {
                lbl: [sorted(qset), outputs[lbl]]
                for lbl, qset in traj_dict.items()
            }
        else:
            # User-supplied outputs: match by position, not label name.
            # Allows T1..T4 to line up with internally stored T0..T3.
            user_labels = list(outputs.keys())
            if len(user_labels) != len(stored_trajs):
                raise ValueError(
                    f"outputs has {len(user_labels)} entries but trajectory set "
                    f"has {len(stored_trajs)}. Provide one output per trajectory."
                )
            traj_for_decoder = {
                user_lbl: [sorted(stored_trajs[i]), outputs[user_lbl]]
                for i, user_lbl in enumerate(user_labels)
            }

        # ── Resolve no_hit ──────────────────────────────────────────────────────────────────
        if no_hit is None:
            no_hit = self.no_hit_ok
        if no_hit and not self.no_hit_ok:
            raise ValueError(
                "|psi> is not orthogonal to the trajectory states, so the "
                "no-hit case cannot be included as a distinct decoder output. "
                "Re-run solve_ts with no_hit=True to find a state that is."
            )

        # ── Build trajectories dict for build_decoder ──────────────────────
        # (traj_for_decoder already set above)
        # ── No-hit: find a free output slot ───────────────────────────────
        no_hit_output = None
        if no_hit:
            used = set(outputs.values())
            no_hit_output = next(
                (f"{i:0{n}b}" for i in range(2**n) if f"{i:0{n}b}" not in used),
                None
            )
            if no_hit_output is None:
                raise ValueError(
                    f"No free output bitstring for no-hit case — "
                    f"all {2**n} computational basis states are already assigned."
                )

        return build_decoder(
            trajectories  = traj_for_decoder,
            n             = n,
            theta         = theta,
            psi           = self.psi,
            no_hit_output = no_hit_output,
            verbose       = verbose,
        )

def _no_hit_row(T1, bs_orbit_ints, n, theta):
    """
    Compute the LP constraint row for the no-hit orthogonality condition.

    Parameters
    ----------
    T1            : frozenset — one representative trajectory
    bs_orbit_ints : list of np.ndarray — integer representations of each orbit
    n             : int
    theta         : float

    Returns
    -------
    row : np.ndarray, shape (N,) — one LP equality row with RHS = 0
    """
    T1_list = sorted(T1)
    row = np.zeros(len(bs_orbit_ints))
    for nu, s_ints in enumerate(bs_orbit_ints):
        phase = np.ones(len(s_ints), dtype=complex)
        for j in T1_list:
            bits   = (s_ints >> (n - 1 - j)) & 1
            phase *= np.exp(+1j * theta * (bits - 0.5))
        row[nu] = phase.real.sum()
    return row

def solve_ts(n, m, kind:Literal['cyclic', 'symmetric']="cyclic", theta: float = None, no_hit: bool = False, eps=1e-4, verbose=True):
    """
    Compute theta_min and find a valid TS state for T_cyc(n,m) or T_sym(n,m).

    Parameters
    ----------
    n    : int   — total sensor qubits
    m    : int   — qubits intercepted per trajectory
    kind : str   — 'cyclic' or 'symmetric'
    eps  : float — nudge above theta_min into the feasible interior

    Returns
    -------
    theta_min : float  (radians)
    psi       : Qobj ket, or None if infeasible
    orbits    : list of frozensets (the G~-orbit partition of bit-strings)
    """
    kind = kind.lower()
    assert kind in ("cyclic", "symmetric"), "kind must be 'cyclic' or 'symmetric'"

    if kind == "cyclic":
        traj_dict = _generate_Tcyc(n, m)
        tmin      = cyc_minInt(n, m)
        label     = rf"$\mathcal{{T}}_\text{{cyc}}({n},{m})$"
    else:
        traj_dict = _generate_Tsym(n, m)
        tmin      = sym_minInt(n)
        label     = rf"$\mathcal{{T}}_\text{{sym}}({n},{m})$"

    trajs = _parse_trajectories(traj_dict)
    theta = theta if theta is not None else tmin + eps

    if kind == "symmetric":
        bs_orbits   = _sym_bitstring_orbits(n)
        pair_orbits = _sym_pair_orbits_fast(n, m, trajs)
        A           = _sym_A_matrix(n, m, theta)
        c = _sym_analytical_c(n, m, theta)
        if c is None:
            valid, c = _solve_ts_lp(A)
        else:
            valid = True
        # For no_hit on symmetric: use integer orbit representations
        _, bs_orbit_ints = _compute_bitstring_orbits_cyc(n) if no_hit else (None, None)
    else:
        perm_group  = _cyc_group(n)
        bs_orbits, bs_orbit_ints = _compute_bitstring_orbits_cyc(n)
        pair_orbits = _compute_traj_pair_orbits(trajs, perm_group)
        A           = _build_ts_matrix_fast(n, bs_orbit_ints, pair_orbits, theta)
        valid, c       = _solve_ts_lp(A)

    # ── No-hit extra constraint ───────────────────────────────────────────
    # <psi| R^(T) |psi> = 0 for all T.
    # For G-transitive trajectory sets with G-invariant |psi>, all trajectories
    # give the same constraint — one representative T1 suffices.
    # We stack one extra equality row onto A and re-solve.
    no_hit_ok = False
    if valid and no_hit:
        if kind == "symmetric" and bs_orbit_ints is None:
            _, bs_orbit_ints = _compute_bitstring_orbits_cyc(n)
        T1       = next(iter(trajs))           # any representative trajectory
        nh_row   = _no_hit_row(T1, bs_orbit_ints, n, theta)
        A_nh     = np.vstack([A.toarray() if sp.issparse(A) else A, nh_row])
        d_nh     = np.zeros(A_nh.shape[0]); d_nh[0] = 1.0
        result_lp = linprog(
            c      = np.zeros(A_nh.shape[1]),
            A_eq   = A_nh,
            b_eq   = d_nh,
            bounds = [(0.0, None)] * A_nh.shape[1],
            method = "highs",
        )
        if result_lp.status == 0:
            valid    = True
            c        = result_lp.x
            no_hit_ok = True
        else:
            valid = False
            if verbose:
                niceprint(
                    "⚠ **No-hit constraint is infeasible at this $\\theta$** — "
                    "no TS state exists that also distinguishes the no-hit case. "
                    "Try a larger $\\theta$ or set `no_hit=False`."
                )
    elif valid:
        # Check post-hoc whether the solution is already no-hit orthogonal
        if bs_orbit_ints is not None:
            T1     = next(iter(trajs))
            nh_row = _no_hit_row(T1, bs_orbit_ints, n, theta)
            no_hit_ok = abs(float(nh_row @ c)) < 1e-6
    
    # ── Header ────────────────────────────────────────────────────────
    if verbose:
        niceprint(
            f"**Trajectory set:** {label} &nbsp;&nbsp; ({len(traj_dict)} trajectories) <br>"
            f"**$\\theta_\\text{{min}}$:** ${_frac_of_pi(tmin)}"
            f"\\approx {tmin:.6f}$ rad"
        )

    if not valid:
        if verbose:
            niceprint("**No TS state found** — LP infeasible at this $\\theta$.")
        
        return TSResult(
            n=n, theta_min=tmin, theta=theta, psi=None,
            orbits=bs_orbits, c=None, traj_dict=traj_dict, no_hit_ok=False
        )

    # ── Orbit table ───────────────────────────────────────────────────
    psi   = _ts_state_from_lp(n, bs_orbits, c)
    vec   = psi.full().flatten()

    if verbose:
        # per-orbit amplitude: every bit-string in orbit nu has the same amplitude
        unnorm_amps = np.array([np.sqrt(max(cv, 0.0)) for cv in c])
        unnorm_norm = np.linalg.norm(
            np.array([unnorm_amps[nu] * np.sqrt(len(orb))
                    for nu, orb in enumerate(bs_orbits)])
        )

        rows = []
        for nu, (orbit, coeff) in enumerate(zip(bs_orbits, c)):
            amp_norm = np.sqrt(max(coeff, 0.0)) / (unnorm_norm if unnorm_norm > 1e-12 else 1.0)
            if amp_norm == 0:
                continue # skip zero-amplitude orbits
            
            # ── Bit-string cell ────────────────────────────────────────────
            strings  = sorted(("".join(map(str, s)) for s in orbit), key=lambda s: int(s, 2))

            # Split into two halves by weight: lower-weight strings first,
            # their bit-flip mirrors second — separated by <br> within the cell.
            by_weight = {}
            for s in strings:
                by_weight.setdefault(s.count("1"), []).append(s)
            weights = sorted(by_weight)

            if len(weights) == 1:
                # nu = n/2 case: all strings same weight, no split needed
                cell_str = ", ".join(strings)
            else:
                upper = ", ".join(by_weight[weights[0]])
                lower = ", ".join(reversed(by_weight[weights[-1]]))
                cell_str = f"{upper},<br><br>{lower}"
            
            # ── Stabilizer & entanglement classification ───────────────────
            mat     = np.array([list(s) for s in sorted(orbit)], dtype=int)
            is_stab = _is_coset_f2(mat)
            is_prod = _is_product_f2(mat, n)

            ket = np.zeros(2**n, dtype=complex)
            for s in orbit:
                ket[int("".join(map(str, s)), 2)] = 1.0
            ket /= np.linalg.norm(ket)
            rho_A = Qobj(ket, dims=[[2]*n, [1]*n]).ptrace(list(range(n // 2)))
            eig   = np.real(rho_A.eigenenergies())
            eig   = eig[eig > 1e-12]
            S_ent = float(-np.sum(eig * np.log2(eig))) if len(eig) else 0.0

            rows.append((nu, cell_str, amp_norm, is_stab, is_prod, S_ent))

        header = "| $\\nu$ | Basis orbit $\\omega_\\nu$ (bit-strings) | Amplitude  $\\sqrt{c_\\nu}$ | Clifford? | Entanglement | $S_{\\mathrm{ent}}$ |"
        sep    = "|:---:|:---|:---:|:---:|:---:|:---:|"
        lines_out = [header, sep]
        for nu, cell_str, amp, is_stab, is_prod, S_ent in rows:
            amp_str  = f"{amp:.4e}" if abs(amp) < 1e-4 else f"{amp:.4f}"
            stab_str = "\u2713" if is_stab else "\u2717"
            ent_str  = "product" if is_prod else "entangled"
            s_str    = f"{S_ent:.3f}"
            lines_out.append(f"| {nu} | {cell_str} | {amp_str} | {stab_str} | {ent_str} | {s_str} |")
        niceprint("\n".join(lines_out))
        niceprint("\u2713 = stabilizer state (Clifford-preparable via H + CNOT + X)")

    return TSResult(
        n=n, theta_min=tmin, theta=theta, psi=psi,
        orbits=bs_orbits, c=c, traj_dict=traj_dict, no_hit_ok=no_hit_ok
    )




def _frac_of_pi(theta):
    """Return a LaTeX string for theta as a multiple of pi."""
    from fractions import Fraction
    frac = Fraction(theta / np.pi).limit_denominator(64)
    if frac.denominator == 1:
        return rf"{frac.numerator}\pi"
    if frac.numerator == 1:
        return rf"\frac{{\pi}}{{{frac.denominator}}}"
    return rf"\frac{{{frac.numerator}\pi}}{{{frac.denominator}}}"


# ──────────────────────────────────────────────────────────────────


# orbit id for state prep

def _rref_f2(rows):
    """
    Row-reduce a list of F_2 row-vectors to reduced row-echelon form.
    Returns (G, pivot_cols) where G contains only the non-zero rows.
    """
    mat = np.array(rows, dtype=int) % 2
    m, ncols = mat.shape
    pivot_cols = []
    row = 0
    for col in range(ncols):
        pivot = next((r for r in range(row, m) if mat[r, col] == 1), None)
        if pivot is None:
            continue
        pivot_cols.append(col)
        mat[[row, pivot]] = mat[[pivot, row]]
        for r in range(m):
            if r != row and mat[r, col] == 1:
                mat[r] = (mat[r] + mat[row]) % 2
        row += 1
        if row >= m:
            break
    return mat[:row], pivot_cols


def _is_coset_f2(mat):
    """
    Check if the rows of binary matrix mat form a coset of a linear subspace
    of F_2^n.
    """
    K = len(mat)
    if K == 0:
        return False
    if K & (K - 1) != 0:               # not a power of 2
        return False
    v = mat[0]
    translated = (mat ^ v[None, :]) % 2
    trans_set = set(map(tuple, translated.tolist()))
    rows_list = list(trans_set)
    for i in range(len(rows_list)):
        for j in range(i + 1, len(rows_list)):
            xij = tuple((np.array(rows_list[i]) ^ np.array(rows_list[j])).tolist())
            if xij not in trans_set:
                return False
    return True


def _is_product_f2(mat, n):
    """
    Check if the equal-amplitude state sum_{s in S} |s> is a product state.
    """
    from itertools import product as iprod
    K = len(mat)
    per_qubit = [set(mat[:, k]) for k in range(n)]
    cart_size = 1
    for s in per_qubit:
        cart_size *= len(s)
    if cart_size != K:
        return False
    s_set = set(map(tuple, mat.tolist()))
    return all(tuple(combo) in s_set for combo in iprod(*per_qubit))


# ──────────────────────────────────────────────────────────────────────────────
def _qutip_to_tket(qutip_circuit, n):
    """
    Convert a QuTiP QubitCircuit (H, CNOT, X, RZ gates) to a pytket Circuit.
    """
    circ = _TketCircuit(n)
    for gate in qutip_circuit.gates:
        name = gate.name.upper()
        if name == "H":
            circ.H(gate.targets[0])
        elif name == "CNOT":
            circ.CX(gate.controls[0], gate.targets[0])
        elif name == "X":
            circ.X(gate.targets[0])
        elif name == "RZ":
            t = gate.arg_value / np.pi   # radians -> half-turns
            circ.Rz(t, gate.targets[0])
        else:
            raise ValueError(f"_qutip_to_tket: unsupported gate '{gate.name}'")
    return circ


def _prepare_circuit_general(psi_vec, n, label=None, verbose=True):
    """
    Prepare an arbitrary n-qubit state |psi> from |00...0> using the quantum
    Shannon decomposition — works for any state, Clifford or not.

    Parameters
    ----------
    psi_vec : array-like, shape (2^n,)
        Target state vector (will be normalised internally).
    n       : int
    label   : str, optional
    verbose : bool

    Returns
    -------
    circuit : pytket Circuit
    psi     : Qobj ket
    """
    from pytket.passes import DecomposeBoxes as _DBX, FullPeepholeOptimise as _FPO

    d       = 2 ** n
    psi_vec = np.array(psi_vec, dtype=complex)
    psi_vec = psi_vec / np.linalg.norm(psi_vec)

    # ── Build U_prep: first column = psi_vec ─────────────────────────────
    # SVD of a (d,1) matrix: psi_vec = U[:,0] * s[0] * Vh[0,0]
    # So U[:,0] = psi_vec / (s[0] * Vh[0,0]).  Fix phase so col 0 = psi_vec.
    U_prep, s, Vh = np.linalg.svd(psi_vec.reshape(-1, 1), full_matrices=True)
    U_prep[:, 0] *= s[0] * Vh[0, 0]   # now U_prep[:,0] = psi_vec exactly

    # ── Synthesize ────────────────────────────────────────────────────────
    circ = _TketCircuit(n)
    _synth_unitary(U_prep, list(range(n)), circ)
    _DBX().apply(circ)
    _FPO().apply(circ)

    psi_qobj = Qobj(psi_vec, dims=[[2] * n, [1] * n])

    if verbose:
        label_str = f" ({label})" if label else ""
        niceprint(
            f"**Non-Clifford preparation{label_str}** "
            f"(quantum Shannon decomposition) <br>"
            f"{circ.n_gates} gates, depth {circ.depth()}, "
            f"{circ.n_2qb_gates()} two-qubit gates"
        )
        niceprint("**Prepared state $|\\psi\\rangle$:**")
        cleandisp(psi_qobj, format='Dirac')
        _draw(circ)

    return circ, psi_qobj

def _prepare_circuit(target_bitstrings, n, label=None, verbose=True):
    """
    Build the shallowest circuit that prepares the equal superposition.

    If the target is NOT a linear coset, no Clifford circuit exists.
    The target state vector is still returned.

    Parameters
    ----------
    target_bitstrings : list of str  e.g. ["0011", "1100", "0110", "1001"]
                        or list of tuples of ints
    n                 : int

    Returns
    -------
    circuit : QubitCircuit (Clifford) | None (non-stabilizer target)
    psi     : Qobj ket  (the prepared state)
    """
    # ── Parse ──────────────────────────────────────────────────────────────
    if isinstance(target_bitstrings[0], str):
        mat = np.array([[int(b) for b in s] for s in target_bitstrings], dtype=int)
    else:
        mat = np.array([list(s) for s in target_bitstrings], dtype=int)
    K = len(mat)

    # ── Build target state vector ──────────────────────────────────────────
    vec = np.zeros(2**n, dtype=complex)
    for row in mat:
        vec[int("".join(map(str, row.tolist())), 2)] = 1.0
    vec /= np.linalg.norm(vec)
    psi = Qobj(vec, dims=[[2]*n, [1]*n])

    # ── Clifford-preparable? ──────────────────────────────────────────────
    if not _is_coset_f2(mat):
        if verbose:
            niceprint(
                "State prep not Clifford constructable."
            )
            cleandisp(psi, format='Dirac')
        return _prepare_circuit_general(vec, n, label=label, verbose=verbose)

    # ── Encoding circuit construction ──────────────────────────────────────
    v = mat[0]                                  # offset vector
    C = (mat ^ v[None, :]) % 2                  # linear subspace
    nonzero_C = C[np.any(C != 0, axis=1)]       # remove zero row

    circuit     = QubitCircuit(n)
    pivot_cols  = []
    check_cols  = list(range(n))
    G           = np.zeros((0, n), dtype=int)
    k           = 0
    n_cnots     = 0

    if len(nonzero_C) > 0:
        G, pivot_cols = _rref_f2(nonzero_C)
        k          = len(pivot_cols)
        check_cols = [c for c in range(n) if c not in pivot_cols]

        # Step 1 — H on each free qubit
        for p in pivot_cols:
            circuit.add_gate("H", targets=[p])

        # Step 2 — CNOT(free -> check) for each 1-entry in G
        for c in check_cols:
            for i, p in enumerate(pivot_cols):
                if G[i, c] == 1:
                    circuit.add_gate("CNOT", controls=[p], targets=[c])
                    n_cnots += 1

    # Step 3 — X gates for the offset v
    n_x = int(np.sum(v))
    for j in range(n):
        if v[j] == 1:
            circuit.add_gate("X", targets=[j])

    if verbose:
        label_str = f" (orbit {label})" if label else ""
        niceprint(
            f"**Clifford preparation{label_str}** for $|\\psi\\rangle$: "
            f"{K} bitstrings, linear dimension $k = {k}$ <br><br>"
            f"Prep Gate Count: <br>"
            f"Hadamards — {k} <br>"
            f"CNOTs — {n_cnots} <br>"
            f"X gates — {n_x} <br>"
            "──────────────<br>"
            f"**Total gates** — **{k + n_cnots + n_x}**"
        )

        psi0    = tensor(*[basis(2, 0) for _ in range(n)])
        psi_out = circuit.run(psi0)
        fid     = float(abs(np.vdot(psi_out.full().flatten(), psi.full().flatten()))**2)
    
        niceprint(f"Optimized Prep vs Target: Fidelity = {fid:.8f}")
        niceprint("**Prepared state $|\\psi\\rangle$:**")
        cleandisp(psi, format='Dirac')

    tket_circ = _qutip_to_tket(circuit, n)

    return tket_circ, psi




def _synth_unitary(U, qubits, circ):
    """
    Recursively decompose the unitary U acting on `qubits` into H, CNOT, Rz,
    Ry gates and add them to `circ`, using the quantum Shannon decomposition
    (cosine-sine decomposition).
    """
    from scipy.linalg import cossin as _cossin
    from pytket.circuit import (Unitary1qBox as _U1, Unitary2qBox as _U2,
                                 Unitary3qBox as _U3,
                                 MultiplexedRotationBox as _MRB,
                                 OpType as _OT)
    n = len(qubits)
    d = 2**n
    assert U.shape == (d, d), f"_synth_unitary: expected {d}×{d}, got {U.shape}"

    if n == 1:
        circ.add_unitary1qbox(_U1(U), qubits[0])
        return
    if n == 2:
        circ.add_unitary2qbox(_U2(U), qubits[0], qubits[1])
        return
    if n == 3:
        circ.add_unitary3qbox(_U3(U), qubits[0], qubits[1], qubits[2])
        return

    half = d // 2
    (u1, u2), thetas, (v1h, v2h) = _cossin(U, p=half, q=half, separate=True)

    # ── Right block: diag(v1h, v2h) ────────────────────────────────────────
    _synth_unitary(v1h, qubits[1:], circ)
    _synth_controlled(v2h @ v1h.conj().T, qubits[0], qubits[1:], circ)

    # ── Centre: multiplexed Ry on qubits[0], controlled by qubits[1:] ──────
    angles_ht = list(2.0 * thetas / np.pi)     # radians → half-turns
    box = _MRB(angles_ht, _OT.Ry)
    circ.add_multiplexedrotation(box, list(qubits[1:]) + [qubits[0]])

    # ── Left block: diag(u1, u2) ────────────────────────────────────────────
    _synth_unitary(u1, qubits[1:], circ)
    _synth_controlled(u2 @ u1.conj().T, qubits[0], qubits[1:], circ)


def _synth_controlled(W, ctrl, targs, circ):
    """
    Add a controlled-W gate to circ:.

    n_t = 1, 2 : direct Unitary2qBox / Unitary3qBox (control + targets).
    n_t >= 3   : eigendecompose W = V D V^dag, implement as
                     V^dag  -->  controlled-diag(I, D)  -->  V
                 where the diagonal box is pytket's DiagonalBox.
    """
    from pytket.circuit import (Unitary2qBox as _U2, Unitary3qBox as _U3,
                                 DiagonalBox as _DB)
    n_t = len(targs)
    d_t = 2**n_t

    if n_t == 1:
        CW = np.block([[np.eye(2, dtype=complex), np.zeros((2, 2))],
                       [np.zeros((2, 2)), W]])
        circ.add_unitary2qbox(_U2(CW), ctrl, targs[0])
        return

    if n_t == 2:
        CW = np.block([[np.eye(4, dtype=complex), np.zeros((4, 4))],
                       [np.zeros((4, 4)), W]])
        circ.add_unitary3qbox(_U3(CW), ctrl, targs[0], targs[1])
        return

    # n_t >= 3: W = V D V^dag
    evals, V = np.linalg.eig(W)
    _synth_unitary(V.conj().T, targs, circ)
    diag = np.concatenate([np.ones(d_t, dtype=complex), evals])
    circ.add_diagonal_box(_DB(diag), [ctrl] + list(targs))
    _synth_unitary(V, targs, circ)


# decoder

def _rz_matrix(theta):
    """Standard R_Z(theta) = diag(e^{-i*theta/2}, e^{i*theta/2})."""
    return np.array([[np.exp(-1j * theta / 2), 0.0],
                     [0.0, np.exp( 1j * theta / 2)]], dtype=complex)


def _apply_trajectory(qubits_hit, n, thetas, psi_vec):
    """
    Apply trajectory to psi_vec.

    thetas : float  — uniform angle applied to every qubit in qubits_hit
             list   — per-qubit angles, same order as qubits_hit

    Qubit ordering: qubit 0 is the most-significant bit (pytket convention).
    Reference: [PRA] Eq. (2); [PRL] Eq. (1).
    """
    if isinstance(thetas, (int, float, np.floating)):
        theta_map = {q: float(thetas) for q in qubits_hit}
    else:
        theta_map = dict(zip(qubits_hit, thetas))
    ops = [_rz_matrix(theta_map[q]) if q in theta_map else np.eye(2, dtype=complex)
           for q in range(n)]
    R = ops[0]
    for o in ops[1:]:
        R = np.kron(R, o)
    return R @ psi_vec

def build_decoder(trajectories, n, theta, psi, no_hit_output=None, verbose=True):
    """
    Build a pytket circuit implementing the decoder unitary U_decode for every trajectory T in the dict.

    Parameters
    ----------
    trajectories : dict
        {label: [qubit_indices, output_bitstring]}
        qubit_indices : list of 0-based ints hit by the particle for label T
        output_bitstring : n-bit string giving the measurement output for T
        e.g. {'T0': [[], '0000'], 'T1': [[0,1], '0001'], ...}
    n             : int   — number of qubits
    theta         : float — interaction angle in radians
    psi           : Qobj ket — TS initial state
    no_hit_output : str or None
        If given, |psi> itself (the no-hit state) is mapped to this output
        bitstring. Must be orthogonal to all trajectory states.
    verbose       : bool

    Returns
    -------
    circuit  : pytket Circuit
    U_decode : numpy ndarray, shape (2^n, 2^n) — the decoder unitary matrix
    """
    from pytket.passes import DecomposeBoxes as _DB, FullPeepholeOptimise as _FPO

    d       = 2**n
    psi_vec = psi.full().flatten()

    # ── Step 1: post-trajectory state vectors ────────────────────────────────
    # theta may be:
    #   float                  — uniform angle for all trajectories
    #   dict {label: float}    — per-trajectory uniform angle
    #   dict {label: list}     — per-trajectory, per-qubit angles
    traj_vecs = {}   # label -> (col_idx, unit_vec)
    for label, (qidx, outbits) in trajectories.items():
        if isinstance(theta, dict):
            t = theta.get(label, theta.get(list(theta.keys())[0]))
        else:
            t = theta
        v = _apply_trajectory(qidx, n, t, psi_vec)
        v = v / np.linalg.norm(v)
        traj_vecs[label] = (int(outbits, 2), v)

    # ── Step 2: no-hit case ──────────────────────────────────────────────────
    no_hit_v = psi_vec / np.linalg.norm(psi_vec)
    if no_hit_output is not None:
        # Verify orthogonality
        max_ov = max(abs(np.dot(no_hit_v.conj(), v)) for _, v in traj_vecs.values())
        if max_ov > 1e-4:
            raise ValueError(
                f"|psi> has overlap {max_ov:.4f} with a trajectory state — "
                "it cannot be used as a distinct no-hit output. "
                "Re-run solve_ts with no_hit=True."
            )
        traj_vecs['no_hit'] = (int(no_hit_output, 2), no_hit_v)
        if verbose:
            niceprint(f"✓ No-hit state included as output `{no_hit_output}`.")
    else:
        # Inform user post-hoc whether no-hit is available
        max_ov = max(abs(np.dot(no_hit_v.conj(), v)) for _, v in traj_vecs.values())
        if verbose and max_ov < 1e-6:
            niceprint(
                "ℹ No-hit state is orthogonal to all trajectory states — "
                "pass `no_hit=True` to result.decode() to include it."
            )

    # ── Step 3: verify mutual orthogonality ─────────────────────────────────
    vecs      = [v for _, v in traj_vecs.values()]
    cross     = np.abs(np.array([[np.dot(vecs[i].conj(), vecs[j])
                                   for j in range(len(vecs))]
                                  for i in range(len(vecs))]))
    np.fill_diagonal(cross, 0.0)
    max_cross = cross.max()
    if max_cross > 1e-4 and verbose:
        niceprint(
            f"⚠ States not fully orthogonal (max off-diagonal overlap = {max_cross:.2e}). "
            "Verify theta is the correct interaction angle."
        )

    # ── Step 4: build U_decode† ──────────────────────────────────────────────
    # Column output_T of U_decode† = v_T.  Remaining columns from SVD complement.
    target_to_vec = {col: v for _, (col, v) in traj_vecs.items()}
    U_dag         = np.zeros((d, d), dtype=complex)
    for col, v in target_to_vec.items():
        U_dag[:, col] = v

    K             = len(target_to_vec)
    V_mat         = np.array(list(target_to_vec.values())).T   # (d, K)
    U_svd, _s, _  = np.linalg.svd(V_mat, full_matrices=True)
    complement     = U_svd[:, K:]                               # (d, d-K)
    free_cols      = [c for c in range(d) if c not in target_to_vec]
    for i, col in enumerate(free_cols):
        U_dag[:, col] = complement[:, i]

    from scipy.linalg import polar as _polar
    U_dag, _ = _polar(U_dag)
    U_decode = U_dag.conj().T

    # ── Unitarity check ──────────────────────────────────────────────────────
    unit_err = float(np.max(np.abs(U_decode @ U_decode.conj().T - np.eye(d))))

    # ── Step 5: fidelity table ───────────────────────────────────────────────
    if verbose:
        rows = []
        for label, (col, v) in traj_vecs.items():
            out    = U_decode @ v
            target = np.zeros(d, dtype=complex); target[col] = 1.0
            fid    = float(abs(np.dot(out.conj(), target))**2)
            rows.append((label, f"{col:0{n}b}", fid))
        lines = ["| Trajectory | Output | Fidelity |", "|:---:|:---:|:---:|"]
        for label, bits, fid in rows:
            lines.append(f"| {label} | `{bits}` | {fid:.8f} |")
        niceprint("\n".join(lines))
        niceprint(f"Unitarity error: ${unit_err:.2e}$")

    # ── Step 6: synthesize ───────────────────────────────────────────────────
    if verbose and n > 6:
        niceprint(
            f"⚠ n={n}: decoder is a ${d}\\times{d}$ unitary — "
            "circuit synthesis via quantum Shannon decomposition is exact but "
            "produces a large number of gates."
        )

    from pytket.circuit import Circuit as _TK
    circ = _TK(n)
    _synth_unitary(U_decode, list(range(n)), circ)
    _DB().apply(circ)
    _FPO().apply(circ)

    if verbose:
        niceprint(
            f"**Decoder circuit:** {circ.n_gates} gates, "
            f"depth {circ.depth()}, {circ.n_2qb_gates()} two-qubit gates"
        )
        _draw(circ)

    return circ, U_decode


def build_ts_circuit(result, intercepted_qubits, theta_turns=None,
                     bitstrings=None, decoder_circuit=None, verbose=False):
    """
    Assemble the full sensing circuit:  prep --> trajectory Rzs --> decoder.

    Parameters
    ----------
    result             : TSResult from solve_ts
    intercepted_qubits : list of int — 0-based qubit indices hit by particle
    theta_turns        : float or list of float, optional
        Interaction strength in pytket half-turns (= theta / pi).
        - float : uniform angle applied to every intercepted qubit
        - list  : per-qubit angles, same order as intercepted_qubits
        Default: result.theta_min / pi (uniform).
        When using a list, pass a matching theta dict to result.decode() so
        the decoder is built from the same per-qubit angles.
    bitstrings         : list of str, optional
        Custom bitstrings for result.prepare(bitstrings=...).
        If None, prepares result.psi directly.
    decoder_circuit    : pytket Circuit, optional
        Pre-built decoder circuit to append.  Pass this when building circuits
        for several trajectories to avoid synthesising the same decoder
        repeatedly — build it once with result.decode(), then reuse:

            circ_decode, U = result.decode(theta=theta_min, verbose=False)
            for name, qubits in trajectories.items():
                circ = build_ts_circuit(result, qubits,
                                        decoder_circuit=circ_decode)

    verbose            : bool

    Returns
    -------
    circuit : pytket Circuit
    """
    n        = result.n
    theta_ht = theta_turns if theta_turns is not None else result.theta_min / np.pi
    if isinstance(theta_ht, (list, np.ndarray)):
        theta_rad = [t * np.pi for t in theta_ht]
    else:
        theta_rad = float(theta_ht) * np.pi

    # ── State preparation ─────────────────────────────────────────────────
    if bitstrings is not None:
        circ, _ = result.prepare(bitstrings=bitstrings, verbose=verbose)
    else:
        circ, _ = result.prepare(verbose=verbose)

    all_qubits = list(range(n))
    circ.add_barrier(all_qubits)

    # ── Particle interaction ──────────────────────────────────────────────
    # theta_turns may be a float (uniform) or list (per-qubit, same order as
    # intercepted_qubits).  theta_ht is only used for the uniform case.
    if isinstance(theta_turns, (list, np.ndarray)):
        if len(theta_turns) != len(intercepted_qubits):
            raise ValueError(
                f"theta_turns has {len(theta_turns)} values but "
                f"intercepted_qubits has {len(intercepted_qubits)} qubits."
            )
        for q, ht in zip(intercepted_qubits, theta_turns):
            circ.Rz(float(ht), q)
    else:
        for q in intercepted_qubits:
            circ.Rz(theta_ht, q)

    circ.add_barrier(all_qubits)

    # ── Decoder ───────────────────────────────────────────────────────────
    if decoder_circuit is None:
        decoder_circuit, _ = result.decode(theta=theta_rad, verbose=verbose)
    circ.append(decoder_circuit)

    return circ



