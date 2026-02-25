from .general import *

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qutip.qip.circuit import QubitCircuit, Gate
from pytket.circuit import Circuit as _TketCircuit
from pytket.circuit.display import render_circuit_jupyter as _draw

from itertools import combinations
from scipy.linalg import null_space
from scipy.optimize import linprog
import scipy.sparse as sp
import re
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
        """Create a LaTeX label for theta."""
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
        Apply a gate sequence to every entangled group, mapping group-relative
        indices to absolute qubit indices.

        This is the single general-purpose method used for both state preparation
        and decoding. Passing a prep_sequence here during the prep stage and a
        decoder_sequence during the decode stage is the entire mechanism — no
        separate prep or decode logic exists.

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


def _precompute_exponents(sets, n, N):
    """
    Compute the integer exponent array for every trajectory pair.

    Returns
    -------
    pairs : list of (i, j) index pairs
    exponents : int8 array of shape (n_pairs, N)
        exponents[p, j] is the exponent used to compute d^{T,T'}_j.
    """
    k_range = np.arange(n)
    j_range = np.arange(N, dtype=np.int64)

    # bits[j, k] = (j >> (n-1-k)) & 1,  shape (N, n)
    bits  = ((j_range[:, None] >> (n - 1 - k_range[None, :])) & 1).astype(np.int8)
    signs = (1 - 2 * bits)  # shape (N, n), entries +/-1

    pairs = list(combinations(range(len(sets)), 2))
    exponents = np.empty((len(pairs), N), dtype=np.int8)
    for idx, (i, j_idx) in enumerate(pairs):
        c = np.array(
            [(1 if k in sets[i] else 0) - (1 if k in sets[j_idx] else 0)
             for k in range(n)],
            dtype=np.int8,
        )
        exponents[idx] = signs @ c  # shape (N,)

    return pairs, exponents


def _build_from_exponents(pairs, exponents, N, theta):
    """
    Build the LP constraint matrix for a given theta using precomputed exponents.
    All N-dimensional work is a single vectorised np.exp() call.

    Returns a sparse CSR matrix A and dense rhs array b.
    """
    # d_all[p, j] = exp(i*theta/2 * exponents[p, j]),  shape (n_pairs, N)
    d_all = np.exp((1j * theta / 2) * exponents.astype(np.float32))

    # Stack real and imaginary rows for each pair, plus normalisation
    n_pairs = len(pairs)
    dense_rows = np.empty((2 * n_pairs + 1, N), dtype=np.float64)
    dense_rows[0:2*n_pairs:2]  = d_all.real
    dense_rows[1:2*n_pairs:2]  = d_all.imag
    dense_rows[-1]              = 1.0  # normalisation

    rhs = np.zeros(2 * n_pairs + 1)
    rhs[-1] = 1.0

    return sp.csr_matrix(dense_rows), rhs


def _build(sets, n, N, theta):
    """Convenience wrapper used by ts_solver (single-theta call, no caching needed)."""
    pairs, exponents = _precompute_exponents(sets, n, N)
    A, b = _build_from_exponents(pairs, exponents, N, theta)
    return A, b, pairs


# ──────────────────────────────────────────────────────────────────
def _cyc_group(n):
    """
    Generate G = Z_n as a list of n permutation tuples (no n! search).
    z^j sends qubit i -> (i+j) mod n.
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


def _infer_n(trajectories, n=None):
    """Infer n from the maximum qubit index across all trajectories."""
    if n is not None:
        return n
    return max(q for T in trajectories for q in T) + 1


# ──────────────────────────────────────────────
#  Symmetry group
# ──────────────────────────────────────────────

def _find_symmetry_group(trajectories, n):
    """
    Return all permutations pi of [n] that map T -> T as a set.
    Each permutation is a length-n tuple where pi[i] = j means qubit i -> j.
    """
    from itertools import permutations as _permutations
    traj_set = frozenset(trajectories)
    group = []
    for perm in _permutations(range(n)):
        mapped = frozenset(frozenset(perm[q] for q in T) for T in trajectories)
        if mapped == traj_set:
            group.append(perm)
    return group

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

    For cyclic G, the orbit of integer s under G~ is:
        {rotate(s, j) for j in 0..n-1}  union  {flip(rotate(s, j)) for j in 0..n-1}
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
#  Orbit computations  (Propositions 11, 12, 16 of [PRA])
# ──────────────────────────────────────────────

def _perm_act_on_bitstring(perm, s):
    """Apply permutation pi to bit-string s: s'[pi[i]] = s[i]."""
    n = len(s)
    inv = [0] * n
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(s[inv[k]] for k in range(n))


def _compute_bitstring_orbits(n, perm_group):
    """
    Partition all 2^n bit-strings into orbits under G~ = G x {I, X^n}.

    G~ acts by permuting qubit positions (G) and/or flipping all bits (X^n).
    Returns a list of frozensets of bit-string tuples.
    """
    all_strings = [
        tuple(int(b) for b in format(i, f"0{n}b"))
        for i in range(2 ** n)
    ]
    visited = set()
    orbits = []

    for s0 in all_strings:
        if s0 in visited:
            continue
        orbit = set()
        stack = [s0]
        while stack:
            s = stack.pop()
            if s in orbit:
                continue
            orbit.add(s)
            for perm in perm_group:
                ps = _perm_act_on_bitstring(perm, s)
                if ps not in orbit:
                    stack.append(ps)
            flipped = tuple(1 - b for b in s)
            if flipped not in orbit:
                stack.append(flipped)
        orbits.append(frozenset(orbit))
        visited |= orbit

    return orbits


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

    Replaces the per-string Python loop in _build_ts_matrix with numpy
    operations: bit extraction via right-shift, vectorised exp, and sum —
    giving ~100x speedup for large n.

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
#  Matrix A(theta)  (Theorem 4, Eq. 58 of [PRA])
# ──────────────────────────────────────────────

def _ts_matrix_element(T, Tp, s, theta):
    r"""
    Compute <s| R^{(T,T')}(theta) |s> for a single Z-eigenstate s.

    Diagonal in the Z-eigenbasis:
      each qubit j in T\T' contributes exp(-i*theta*(s_j - 1/2))
      each qubit j in T'\T contributes exp(+i*theta*(s_j - 1/2))

    The imaginary part cancels when summed over a bit-flip-closed orbit.
    """
    phase = complex(1.0)
    for j in T - Tp:
        phase *= np.exp(-1j * theta * (s[j] - 0.5))
    for j in Tp - T:
        phase *= np.exp(+1j * theta * (s[j] - 0.5))
    return phase.real


def _build_ts_matrix(n, bitstring_orbits, pair_orbits, theta):
    """
    Build the M x N matrix A(theta) from Theorem 4 of [PRA].

    A[mu, nu] = sum_{s in omega_nu} <s| R^{(T,T')}(theta) |s>
    Row mu=0 is the normalisation row (diagonal orbit gives A[0,nu] = |omega_nu|).
    """
    M = len(pair_orbits)
    N = len(bitstring_orbits)
    A = np.zeros((M, N))
    for mu, pair_orbit in enumerate(pair_orbits):
        T, Tp = next(iter(pair_orbit))
        for nu, bs_orbit in enumerate(bitstring_orbits):
            A[mu, nu] = sum(_ts_matrix_element(T, Tp, s, theta) for s in bs_orbit)
    return A


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
    Eq. (71):

    A_{mu,nu}(theta) = alpha_nu * sum_{i=0}^{mu} sum_{i'=0}^{mu}
                         C(mu,i) * C(mu,i') * C(n-2mu, nu-i-i') * cos[(i-i')*theta]

    Avoids all group computation and the 2^n matrix-element loop entirely.
    M = min(m, n-m)+1 rows, N = floor(n/2)+1 columns.
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
    Analytical coefficients c_nu for T_sym(n, m=n/2), from Eq. (5) of [PRL]:

        c_nu  proportional to  (-1)^{m-nu} * cos[(m-nu)*theta]

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

    def __init__(self, n, theta_min, psi, orbits, c):
        self.n         = n
        self.theta_min = theta_min
        self.psi       = psi
        self.orbits    = orbits
        self._c        = c   # LP coefficients — used to identify active orbits

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

        # Default: iterate all active orbits
        results = []
        for nu_i in self._active_nus():
            bs = ["".join(map(str, s)) for s in sorted(self.orbits[nu_i])]
            results.append(_prepare_circuit(bs, self.n, label=f"$\\omega_{{{nu_i}}}$", verbose=verbose))
        return results


def solve_ts(n, m, kind:Literal['cyclic', 'symmetric']="cyclic", eps=1e-4, verbose=True):
    """
    Compute theta_min and find a valid TS state for T_cyc(n,m) or T_sym(n,m).

    Uses the group-theoretic LP reduction of Theorem 4 from [PRA]:
    a TS state exists at theta iff  A(theta) c = d, c >= 0  is feasible.
    The TS state is then  |psi> = sum_nu sqrt(c_nu) |nu>,  where |nu> is the
    equal superposition of all bit-strings in the nu-th G~-orbit.

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
    theta = tmin + eps

    if kind == "symmetric":
        # Fast path (Proposition 13 of [PRA]): build A analytically,
        # skip the O(n!) group search, O(|T|^2) pair enumeration,
        # and O(2^n) matrix-element loop entirely.
        bs_orbits   = _sym_bitstring_orbits(n)
        pair_orbits = _sym_pair_orbits_fast(n, m, trajs)
        A           = _sym_A_matrix(n, m, theta)
        # For m=n/2: closed-form c_nu from Eq. (5) of [PRL] — skip LP too
        c = _sym_analytical_c(n, m, theta)
        if c is None:
            valid, c = _solve_ts_lp(A)
        else:
            valid = True
    else:
        # Cyclic fast path: Z_n group (n elements) + integer-rotation orbit computation
        perm_group  = _cyc_group(n)
        bs_orbits, bs_orbit_ints = _compute_bitstring_orbits_cyc(n)
        pair_orbits = _compute_traj_pair_orbits(trajs, perm_group)
        A           = _build_ts_matrix_fast(n, bs_orbit_ints, pair_orbits, theta)
        valid, c       = _solve_ts_lp(A)

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
        return tmin, None, bs_orbits

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
                w = s.count("1")
                by_weight.setdefault(w, []).append(s)
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

    return TSResult(n=n, theta_min=tmin, psi=psi, orbits=bs_orbits, c=c)




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


# ══════════════════════════════════════════════════════════════════════════════
#  Orbit classification  &  state preparation
# ══════════════════════════════════════════════════════════════════════════════

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

    For an equal-amplitude state sum_{s in S} |s> to be a stabilizer state,
    the support S must be a coset of a linear code over F_2^n.
    (Mike & Ike §10.5; [PRA] Section V C — stabilizer TS codes.)

    Two necessary conditions: |S| must be a power of 2, and S - v must be
    closed under XOR for some (any) representative v in S.
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

    The state is separable iff S = S_0 x S_1 x ... x S_{n-1} (Cartesian
    product of per-qubit subsets), i.e. the qubits are statistically independent.
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

    Gate mapping:
      H    -> H
      CNOT -> CX  (pytket uses CX for controlled-X)
      X    -> X
      RZ   -> Rz  (pytket Rz takes angle in half-turns, i.e. t where U = e^{i*pi*t/2}*Rz(pi*t))
                  QuTiP arg_value is in radians; pytket wants turns = arg_value / pi
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


def _prepare_circuit(target_bitstrings, n, label=None, verbose=True):
    """
    Build the shallowest circuit that prepares the equal superposition:

        |psi>  =  (1/sqrt(K))  sum_{s in target}  |s>

    If the target forms a coset of a linear code over F_2^n, uses only
    Hadamard, CNOT, and X gates — the standard stabilizer encoding circuit.
    (Mike & Ike §10.5.3, Problem 10.3; [PRA] Section IV B.)

    Encoding circuit derivation:
      1. Let v = any element of target (offset).
      2. C = {s XOR v : s in target} is a k-dimensional linear subspace.
      3. RREF of C over F_2 gives generator matrix G and pivot (free) columns.
      4. Circuit:
           H on each free qubit  f_i
           CNOT(f_i -> c_j)  for each check qubit c_j where G[i, c_j] = 1
           X on qubit j  where v[j] = 1  (applies the offset)

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
                "State prep not supported by just Clifford gates."
            )
            cleandisp(psi, format='Dirac')
        return None, psi

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
            f"Hadamards — {k} <br>"
            f"CNOTs — {n_cnots} <br>"
            f"X gates — {n_x} <br>"
            "──────────────<br>"
            f"**Total gates** — **{k + n_cnots + n_x}**"
        )

        psi0    = tensor(*[basis(2, 0) for _ in range(n)])
        psi_out = circuit.run(psi0)
        fid     = float(abs(np.vdot(psi_out.full().flatten(), psi.full().flatten()))**2)
    
        niceprint(f"Fidelity = {fid:.8f}")
        niceprint("**Prepared state $|\\psi\\rangle$:**")
        cleandisp(psi, format='Dirac')

    tket_circ = _qutip_to_tket(circuit, n)

    return tket_circ, psi

def prepare_ts_state(target_bitstrings, n):
    return _prepare_circuit(target_bitstrings, n, label=None)


