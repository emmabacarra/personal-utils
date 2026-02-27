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

def _infer_n(trajectories, n=None):
    """Infer n from the maximum qubit index across all trajectories."""
    if n is not None:
        return n
    return max(q for T in trajectories for q in T) + 1

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




