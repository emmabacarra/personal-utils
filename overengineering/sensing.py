from .general import *

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qutip.qip.circuit import QubitCircuit, Gate

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

def _lbl(j,n):
    return format(j, f"0{n}b")

def insert_phases(s):
    def replacer(m):
        j = int(m.group(1), 2)          # binary string -> index j
        return f"e^{{i\\phi_{{{j}}}}}|{m.group(1)}\\rangle"
    return re.sub(r'\|(\d+)\\rangle', replacer, s)

def _cyclic_shift(j: int, n: int) -> int:
    """Right-rotate the n-bit integer j by 1 position."""
    return ((j >> 1) | ((j & 1) << (n - 1))) & ((1 << n) - 1)

def _cyclic_orbits(n: int) -> list:
    """Return orbits of {0,...,2^n-1} under the cyclic group Z_n."""
    N, visited, orbits = 2 ** n, set(), []
    for j in range(N):
        if j not in visited:
            orbit, cur = [], j
            while cur not in visited:
                visited.add(cur); orbit.append(cur); cur = _cyclic_shift(cur, n)
            orbits.append(orbit)
    return orbits

def _cyclic_equality_rows(n: int, N: int):
    """
    Build equality constraint rows enforcing cyclic symmetry on p.

    For each orbit [j0, j1, ...] under Z_n, adds rows:
        p[j0] - p[j1] = 0
        p[j1] - p[j2] = 0
        ...
    Stacking these with the orthogonality system restricts the LP
    to cyclic-invariant distributions.

    Returns a *sparse* CSR matrix and a dense rhs vector.
    Each row has exactly 2 non-zero entries, so sparse storage
    avoids the O(n_rows * N) memory footprint of the dense version.
    (For 16 qubits this is the difference between ~30 GB and ~1 MB.)
    """
    orbits = _cyclic_orbits(n)

    # Pre-count rows so we can pre-allocate COO data arrays
    n_rows = sum(len(orbit) - 1 for orbit in orbits)

    data    = np.empty(2 * n_rows, dtype=np.float64)
    row_idx = np.empty(2 * n_rows, dtype=np.int32)
    col_idx = np.empty(2 * n_rows, dtype=np.int32)
    rhs     = np.zeros(n_rows, dtype=np.float64)

    ptr = 0
    row = 0
    for orbit in orbits:
        for i in range(len(orbit) - 1):
            data[ptr]     =  1.0;  row_idx[ptr]     = row;  col_idx[ptr]     = orbit[i]
            data[ptr + 1] = -1.0;  row_idx[ptr + 1] = row;  col_idx[ptr + 1] = orbit[i + 1]
            ptr += 2
            row += 1

    A_cyc = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n_rows, N))
    return A_cyc, rhs

def ts_solver(trajectories: dict, theta: float, cyclic_constraint: bool, verbose: Literal[True, False, 'off']=True) -> dict:
    """
    Solves for a valid trajectory-sensing initial state (if one exists) for the given trajectories and interaction strength.
    
    Parameters
    ----------
    trajectories : dict
        Maps trajectory names to sets of qubit indices. Example: {'T1': {0,1}, 'T2': {1,2}} or {'T1': (0,1), 'T2': (1,2)}.
        Use None or empty set for no trajectory (T0).
    
    theta : float
        Interaction strength (rotation angle) for the particle-sensor interaction.
    
    cyclic_constraint : bool
        If True, constrains the solution to be cyclically symmetric (i.e., all qubits have equal probability).
    
    verbose : bool or 'off'
        If True, prints details about the solution. If False, prints only the final results without the details. If 'off', suppresses all output.
    """
    
    names = list(trajectories.keys())
    sets  = [set(trajectories[k]) for k in names]
    n = max(q for s in sets for q in s) + 1
    N = 2 ** n

    A, b, pairs = _build(sets, n, N, theta)
    if cyclic_constraint:
        A_cyc, b_cyc = _cyclic_equality_rows(n, N)
        A = sp.vstack([A, A_cyc], format='csr')
        b = np.hstack([b, b_cyc])

    # Null space: degrees of freedom in the probability distribution
    NULLSPACE_MAX_N = 12
    if n <= NULLSPACE_MAX_N:
        V = null_space(A.toarray() if sp.issparse(A) else A, rcond=1e-9)
        null_dim = V.shape[1]
    else:
        V = None
        null_dim = None

    # find solution p_j ≥ 0 via linear programming
    lp = linprog(
        c=np.zeros(N),
        A_eq=A, b_eq=b,
        bounds=[(0, None)] * N,
        method="highs",
    )
    
    theta_formatted = cleandisp(theta/np.pi, return_str = 'Latex')

    if not lp.success:
        if verbose != 'off':
            niceprint(f"Initial state with trajectories {names} at $\\theta = {theta_formatted}\\pi$ does not exist.")
        return
    else:
        
        p_feasible = lp.x
        solution_valid = np.linalg.norm(A @ p_feasible - b) # double checking solution validity
        is_valid = solution_valid < 1e-8 and not np.any(p_feasible < -1e-9)
        if not is_valid:
            if verbose != 'off':
                niceprint(f"Solution found but failed validity check (residual {solution_valid:.2e}, min p_j = {p_feasible.min():.2e}).")
            return
        
        else:
            if verbose:
                null_dim_str = f"{null_dim}" if null_dim is not None else f"(skipped for n={n})"
                niceprint(f"**Solutions for trajectories {names} at $\\theta = {theta_formatted}\\pi$** <br>" +
                        f"{n} qubits $\\rightarrow$ Hilbert space dimension (# unknowns) {N} <br>" +
                        f"{len(names)} trajectories $\\rightarrow$ {len(pairs)} pairs of orthogonality constraints <br>" +
                        f"# of equations: {2*len(pairs)} from orthogonality + 1 from normalisation = {2*len(pairs)+1} <br>" +
                        f"Null space dimension: {null_dim_str}  (degrees of freedom in the probability distribution space)"
                        )

            if verbose != 'off':
                nz = [(j, p_feasible[j]) for j in range(N) if p_feasible[j] > 1e-9]
                nz.sort(key=lambda x: x[1]) # sort nz by increasing probability
                terms = ""
                for idx, (j, pj) in enumerate(nz): # break up into multiple lines
                    terms += f"$\\quad{cleandisp(np.sqrt(pj), return_str = 'Latex')}\\,e^{{i\\phi_{{{j}}}}}|{_lbl(j,n)}\\rangle$"
                    if (idx + 1) % 4 == 0:
                        terms += "<br>"
                    else:
                        terms += ", "
                niceprint(f"**Valid (orthogonal) states of $p_{{\\text{{feasible}}}}$ with arbitrary phase for $\\theta = {theta_formatted}\\pi$ ({len(nz)} total)**: <br>" +
                        f"{terms}"
                        )
                
                # Xs = [_lbl(j, n) for j, pj in nz] # states |j>
                # Ys = [pj for j, pj in nz] # probabilities p_j
                # fig, ax = plt.subplots(figsize=(6, 3))
                # ax.bar(Xs, Ys)
                # ax.set_xlabel("Basis states $|j\\rangle$")
                # ax.set_ylabel("Probability")
                # ax.set_title(f"valid probability distribution $p_{{\\text{{feasible}}}}$ for $\\theta$ = {theta_formatted}")
                # plt.show()
                
                null_space_size = null_dim
                if null_space_size is None:
                    niceprint(f"(Null space dimension not computed for n={n} qubits to avoid memory issues.)")
                else:
                    if null_space_size > 3:
                        pdist_str = f"$p = p_{{\\text{{feasible}}}} + \\alpha_1 \\cdot v_1 + \\ldots + \\alpha_{{{null_space_size}}} \\cdot v_{{{null_space_size}}}$"
                    else:
                        terms = ' + '.join(f'\\alpha_{{{k+1}}} \\cdot v_{{{k+1}}}' for k in range(null_space_size))
                        pdist_str = f"$p = p_{{\\text{{feasible}}}} + {terms}$"
                    niceprint(f"For the full family of valid initial states, the probability distribution $p$ is: <br>" +
                            f"$\\quad$ {pdist_str} <br>" +
                            (f"where $v_k$ are the vectors of the probability distribution (null space of the linear system A·p = b) and and $\\alpha_k$ are arbitrary real numbers (choose them so that $p_j \\geq 0$ for every $j$)" if verbose else "")
                            )
    
    return lp.success, p_feasible, n


def find_min_theta(trajectories: dict, n_iters: int = 60, cyclic_constraint: bool=False, verbose: bool = True):
    """
    Find the minimum interaction strength θ_min ∈ (0, π] for which a valid
    trajectory-sensing initial state exists, via binary search on LP feasibility.

    Performance note: exponents and (if cyclic_constraint) A_cyc are precomputed
    once and reused across all binary-search iterations. Each feasibility check
    only recomputes the np.exp() call and solves the LP.
    """
    
    clean = {k: set(v) for k, v in trajectories.items()
             if v is not None and len(v) > 0}

    if len(clean) < 2:
        niceprint("Need at least 2 non-empty trajectories to define orthogonality constraints.")
        return

    names = list(clean.keys())
    sets  = list(clean.values())
    n = max(q for s in sets for q in s) + 1
    N = 2 ** n

    # ── precompute once ────────────────────────────────────────────────────────
    pairs, exponents = _precompute_exponents(sets, n, N)

    A_cyc, b_cyc = (_cyclic_equality_rows(n, N) if cyclic_constraint
                    else (None, None))
    # ──────────────────────────────────────────────────────────────────────────

    def feasible(theta):
        A, b = _build_from_exponents(pairs, exponents, N, theta)
        if cyclic_constraint:
            A = sp.vstack([A, A_cyc], format='csr')
            b = np.hstack([b, b_cyc])
        lp = linprog(np.zeros(N), A_eq=A, b_eq=b,
                     bounds=[(0, None)] * N, method="highs")
        return lp.success

    if not feasible(np.pi):
        niceprint(f"No valid initial state exists for trajectories {names} at any $\\theta \\in (0, \\pi]$.")
        return

    # Binary search: maintain lo = infeasible, hi = feasible
    lo, hi = 0.0, np.pi
    for _ in range(n_iters):
        mid = (lo + hi) / 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    theta_min = hi

    theta_formatted = cleandisp(theta_min / np.pi, return_str='Latex')
    if verbose:
        niceprint(
            f"**Minimum $\\theta$ for trajectories {names}** <br>" +
            f"$\\quad \\theta_{{\\min}} = {theta_formatted}\\pi$" 
        )

    return theta_min


