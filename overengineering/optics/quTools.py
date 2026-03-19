"""
API built for quTools simulations.

Hierarchy
------------------
  quED
  ├── pump   : PumpBlock   — 405 nm diode + waveplate + BBO crystal
  ├── block  : DetectionBlock
  │            ├── arm1 : DetectionArm   — QWP (opt.) + polarizer + APD
  │            └── arm2 : DetectionArm
  ├── SPDC   : SPDCInterface   — count_rates(), power()
  └── stats  : PhotonStatsInterface — g2()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, Dict, List
from qutip import Qobj

from ..general   import niceprint, cleandisp, nicetable
from .params     import SPDC
from .components import QuarterWavePlate
from .simulators import SPDCSimulator, OpticalCircuit, PhotonBeamSimulator
from .helpers    import (
    _LABELS_1Q, _SETTINGS, _analysis_circuit,
    density_matrix_1photon, density_matrix_2photon,
    rho_eigensystem, rho_properties, _stokes_contrast,
)

_PALETTE = ['mediumvioletred', 'dodgerblue', 'mediumseagreen', 'gold', 'mediumpurple']

_SINGLES_COLOR = {
    'Arm 1':       'mediumvioletred',
    'Arm 2':       'dodgerblue',
    'Coincidence': 'mediumseagreen',
}

# optimal CHSH angles
_CHSH_ANGLES = {
    'singlet': { #PHI-
        'alpha': (0.0,          np.pi / 4),
        'beta':  (-np.pi / 8,    5*np.pi / 8),
    },
    'triplet': { #PHI+
        'alpha': (0.0,          3*np.pi / 4),
        'beta':  (-np.pi / 8,    5*np.pi / 8),
    },
}

def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_title(title,   fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.spines[['top', 'right']].set_visible(False)


# ────────────────────────────────────────────────────────────────────────────────────
# output containers
# ────────────────────────────────────────────────────────────────────────────────────

class TomoResult:
    """
    Return value from ``quED.tomo1q()`` and ``quED.tomo2q()``.

    Attributes
    ----------
    rho_ideal   : Qobj   - ideal density matrix
    rho_recon   : Qobj   - linear-inversion reconstructed density matrix
    counts      : ndarray - shape (6,) for 1-qubit or (6,6) for 2-qubit
    fidelity    : float
    purity      : float
    eigenvalues : ndarray
    """

    def __init__(self, rho_ideal: Qobj, rho_recon: Qobj,
                 counts: np.ndarray, n_qubits: int):
        self.rho_ideal   = rho_ideal
        self.rho_recon   = rho_recon
        self.counts      = counts
        self.n_qubits    = n_qubits
        self.properties  = rho_properties(rho_recon)
        self.eigenvalues, self.eigenvectors = rho_eigensystem(rho_recon)
        self.purity      = self.properties['purity']

        rho_r = rho_recon.full() if isinstance(rho_recon, Qobj) else np.asarray(rho_recon)
        rho_i = rho_ideal.full() if isinstance(rho_ideal, Qobj) else np.asarray(rho_ideal)
        self.fidelity = float(np.real(np.trace(rho_i @ rho_r)))

        if n_qubits == 1:
            N = counts
            self.stokes = {
                'S1': _stokes_contrast(N[2], N[3]),   # D / A
                'S2': _stokes_contrast(N[4], N[5]),   # R / L
                'S3': _stokes_contrast(N[0], N[1]),   # H / V
            }

    def summary(self, label: str = ''):
        n_str = 'Single-Qubit' if self.n_qubits == 1 else 'Two-Qubit'
        title = f'**Quantum State Tomography — {n_str}**'
        if label:
            title += f': {label}'

        niceprint('---')
        niceprint(title, 3)

        # ── counts ──────────────────────────────────────────────────────────────
        if self.n_qubits == 1:
            N = self.counts.astype(int)
            niceprint('<u> Measurement Counts </u>', 5)
            nicetable(
                headers=['Projection', 'H', 'V', 'D', 'A', 'R', 'L'],
                rows=[['Counts',
                       f'{N[0]:,}', f'{N[1]:,}', f'{N[2]:,}',
                       f'{N[3]:,}', f'{N[4]:,}', f'{N[5]:,}']],
                align='lcccccc',
            )

            S = self.stokes
            niceprint('<u> Stokes Parameters </u>', 5)
            niceprint(
                f'$S_1$ (D/A): {S["S1"]:+.4f} <br>'
                f'$S_2$ (R/L): {S["S2"]:+.4f} <br>'
                f'$S_3$ (H/V): {S["S3"]:+.4f}'
            )

        else:
            C = self.counts.astype(int)
            niceprint('<u> Coincidence Count Matrix (6x6) </u>', 5)
            basis_labels = ['H', 'V', 'D', 'A', 'R', 'L']
            rows = [
                [f'**{basis_labels[i]}**'] + [f'{C[i, j]:,}' for j in range(6)]
                for i in range(6)
            ]
            nicetable(
                headers=[''] + basis_labels,
                rows=rows,
                align='l' + 'c'*6,
            )

        # ── reconstructed ρ ─────────────────────────────────────────────────────
        niceprint('<u> Reconstructed $\\rho$ </u>', 5)
        niceprint(cleandisp(self.rho_recon, return_str='Markdown'))

        # ── state properties ─────────────────────────────────────────────────────
        niceprint('<u> State Properties </u>', 5)
        eig_str = ' &nbsp;&nbsp; '.join(f'{v:+.4f}' for v in self.eigenvalues)
        niceprint(
            f'Fidelity $F = \\mathrm{{Tr}}[\\rho_{{\\mathrm{{ideal}}}}'
            f'\\rho_{{\\mathrm{{recon}}}}]$: {self.fidelity:.4f} <br>'
            f'Purity $\\mathrm{{Tr}}[\\rho^2]$: {self.purity:.4f} <br>'
            f'Eigenvalues: {eig_str}'
        )
        niceprint('---')

    def plot(self, label: str = '', ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Bar chart of Re/Im parts of every element of the reconstructed ρ."""
        rho = (self.rho_recon.full() if isinstance(self.rho_recon, Qobj)
            else np.asarray(self.rho_recon))
        n = rho.shape[0]

        basis   = (['HH', 'HV', 'VH', 'VV'] if n == 4 else ['H', 'V'])
        xlabels = [f'$\\rho_{{{r},{c}}}$' for r in basis for c in basis]
        re_vals = [rho[i, j].real for i in range(n) for j in range(n)]
        im_vals = [rho[i, j].imag for i in range(n) for j in range(n)]

        spacing = 1.5 if n == 4 else 0.8
        x       = np.arange(len(xlabels)) * spacing
        w       = 0.30

        if ax is None:
            _, ax = plt.subplots(figsize=(6.75, 2.2))

        ax.bar(x - w/2, re_vals, w, label='Re', color=_PALETTE[0], alpha=0.88)
        ax.bar(x + w/2, im_vals, w, label='Im', color=_PALETTE[1], alpha=0.88)
        ax.set_xticks(x)
        if n == 4:
            ax.set_xticklabels(xlabels, fontsize=12, rotation=45, ha='center')
        else:
            ax.set_xticklabels(xlabels, fontsize=12, rotation=0,  ha='center')
        ax.axhline(0, color='k', linewidth=0.6)
        ax.set_ylim(-0.65, 0.65)
        ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
        ax.set_yticklabels(['-0.5', '-0.25', '0', '0.25', '0.5'], fontsize=12)
        ax.set_ylabel('Matrix element', fontsize=10)
        n_str  = 'Single-Qubit' if self.n_qubits == 1 else 'Two-Qubit'
        title  = f'Reconstructed $\\rho$ ({n_str})'
        if label:
            title += f' — {label}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=4)
        ax.legend(fontsize=9, framealpha=0.7)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        return ax



class G2Result:
    """
    Return value from ``quED.stats.g2()``.

    ``value()``   - (mean, std)
    ``summary()`` - formatted niceprint output
    ``plot()``    - histogram
    """

    def __init__(self, g2_vals: np.ndarray, mode: str, mu: float, flux: float):
        self._vals = g2_vals[~np.isnan(g2_vals)]
        self.mode  = mode
        self.mu    = mu
        self.flux  = flux

    def value(self) -> Tuple[float, float]:
        """Returns (mean, std) of the simulated g^2(0) distribution."""
        return float(np.mean(self._vals)), float(np.std(self._vals))

    def summary(self):
        mu_val, std_val = self.value()
        mode_str = 'Heralded' if self.mode == 'heralded' else 'Unheralded'
        is_nc    = mu_val < 0.5

        niceprint('---')
        niceprint(f'**Photon Statistics — {mode_str} $g^{{(2)}}(0)$**', 3)

        niceprint('<u> Simulation Parameters </u>', 5)
        niceprint(
            f'Mode: {mode_str} <br>'
            f'Trials: {len(self._vals):,} <br>'
            f'Mean photons per raw bin $\\mu$: {self.mu:.4e} <br>'
            f'Photon flux: {self.flux:.3e} ph/s'
        )

        niceprint('<u> Result </u>', 5)
        verdict = (r'$\color{green}\checkmark$ **Non-classical** ($< 0.5$)'
                   if is_nc else r'$\color{red}\times$ Classical ($\geq 0.5$)')
        niceprint(
            f'$g^{{(2)}}(0) = {mu_val:.4f} \\pm {std_val:.4f}$ <br>'
            f'{verdict}'
        )
        nicetable(
            headers=['Quantity', 'Value'],
            rows=[
                ['Mean $g^{(2)}(0)$', f'{mu_val:.4f}'],
                ['Std $g^{(2)}(0)$',  f'{std_val:.4f}'],
                ['Coherent limit',    '1.0000'],
                ['Non-classical?',    'Yes' if is_nc else 'No'],
            ],
            align='lc',
        )
        niceprint('---')

    def plot(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        mu_val, std_val = self.value()
        ax.hist(self._vals, bins=40, color=_PALETTE[0], edgecolor='white', alpha=0.85)
        ax.axvline(mu_val, color=_PALETTE[2], linewidth=2,
                   label=f'Mean = {mu_val:.4f}')
        ax.axvline(1.0, color='gray', linewidth=1.5, linestyle='--',
                   label='Coherent limit = 1')
        if mu_val < 0.5:
            ax.axvspan(0, 0.5, alpha=0.06, color=_PALETTE[2],
                       label='Non-classical (< 0.5)')
        ax.legend(fontsize=9)
        mode_str = 'Heralded' if self.mode == 'heralded' else 'Unheralded'
        _style_ax(ax,
                  title=f'{mode_str} $g^{{(2)}}(0)$ Distribution '
                        f'({len(self._vals)} trials)',
                  xlabel='$g^{(2)}(0)$', ylabel='Count')
        plt.tight_layout()
        return ax



class CHSHResult:
    """
    Return value from ``quED.CHSH()``.

    ``value()``        - (S, sigma_S)
    ``summary()``      - formatted niceprint output with per-correlator table
    ``correlations()`` - E(alpha, beta) vs beta sweep plot
    """

    def __init__(self, state: str, S: float, sigma_S: float,
                 settings: dict, vis: float, angles: dict, analyzer):
        self.state      = state
        self.S          = S
        self.sigma_S    = sigma_S
        self.settings   = settings
        self.visibility = vis
        self.angles     = angles
        self._ana       = analyzer

    def value(self) -> Tuple[float, float]:
        """Returns (S, sigma_S)."""
        return self.S, self.sigma_S

    def summary(self):
        lbl_state = ('singlet $|\\Phi^-\\rangle$' if self.state == 'singlet'
                     else 'triplet $|\\Phi^+\\rangle$')
        niceprint('---')
        niceprint(f'**CHSH Bell Inequality — {lbl_state}**', 3)

        # ── per-setting table ────────────────────────────────────────────────────
        niceprint('<u> Per-Setting Results </u>', 5)
        rows = []
        for key, r in self.settings.items():
            rows.append([
                f'${key}$',
                f"{r['N_HH']:,}", f"{r['N_HV']:,}",
                f"{r['N_VH']:,}", f"{r['N_VV']:,}",
                f"{r['E']:+.4f}",
                f"{r['sigma_E']:.4f}",
            ])
        nicetable(
            headers=['Setting', '$N_{HH}$', '$N_{HV}$',
                     '$N_{VH}$', '$N_{VV}$', '$E$', '$\\sigma_E$'],
            rows=rows,
            align='lcccccc',
        )

        # ── CHSH parameter ───────────────────────────────────────────────────────
        niceprint('<u> CHSH Parameter </u>', 5)
        S_theory    = self.visibility * 2.0 * np.sqrt(2.0)
        violates    = abs(self.S) > 2.0
        sigma_above = (abs(self.S) - 2.0) / self.sigma_S if self.sigma_S > 0 else 0.0
        verdict = (
            r'$\color{green}\checkmark$ **Bell inequality violated** '
            f'(${sigma_above:.1f}\\sigma$ above LHV bound)'
            if violates else r'$\color{red}\times$ No violation detected'
        )
        niceprint(
            f'$\\langle\\hat{{S}}\\rangle = {self.S:.4f} \\pm {self.sigma_S:.4f}$ <br>'
            f'QM prediction $2\\sqrt{{2}}\\cdot V = {S_theory:.4f}$ <br>'
            f'Classical bound $|\\langle\\hat{{S}}\\rangle| \\leq 2$ <br>'
            f'{verdict}'
        )

        # ── angle table ──────────────────────────────────────────────────────────
        niceprint('<u> Measurement Angles </u>', 5)
        a1 = np.rad2deg(self.angles['alpha'][0])
        a2 = np.rad2deg(self.angles['alpha'][1])
        b1 = np.rad2deg(self.angles['beta'][0])
        b2 = np.rad2deg(self.angles['beta'][1])
        from ..general import cleandisp
        a1r, a2r = self.angles['alpha'][0], self.angles['alpha'][1]
        b1r, b2r = self.angles['beta'][0],  self.angles['beta'][1]

        def _rad_str(x):
            return f'${cleandisp(x, return_str="Latex")}$'

        nicetable(
            headers=['', '$\\alpha_1$', '$\\alpha_2$', '$\\beta_1$', '$\\beta_2$'],
            rows=[
                ['Angle ($^\\circ$)',   f'${a1:.1f}^\\circ$',  f'${a2:.1f}^\\circ$',
                                f'${b1:.1f}^\\circ$',  f'${b2:.1f}^\\circ$'],
                ['Angle (rad)', _rad_str(a1r), _rad_str(a2r),
                                _rad_str(b1r), _rad_str(b2r)],
            ],
            align='lcccc',
        )
        niceprint('---')

    def correlations(
        self,
        angle:       Optional[Tuple[float, float, float]] = None,
        fixed_alpha: Optional[float]   = None,
        ax:          Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot E(alpha, beta) vs beta with alpha held fixed.

        Parameters
        ----------
        angle       : (start_deg, stop_deg, step_deg).  Defaults to (0, 360, 3).
        fixed_alpha : Alice's fixed angle [degrees].
                      Defaults to first optimal alpha for the state.
        """
        start, stop, step = angle if angle is not None else (0, 360, 3)
        betas_rad = np.deg2rad(np.arange(start, stop, step))
        alpha_rad = (np.deg2rad(fixed_alpha) if fixed_alpha is not None
                     else self.angles['alpha'][0])

        E_theory = np.array([
            np.dot(self._ana._joint_probs(alpha_rad, b), [1, -1, -1, 1])
            for b in betas_rad
        ])

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.plot(np.rad2deg(betas_rad), E_theory,
                color=_PALETTE[0], linewidth=2, label='Theory')

        b1_deg = np.rad2deg(self.angles['beta'][0]) % 360
        b2_deg = np.rad2deg(self.angles['beta'][1]) % 360
        for b_deg, lbl in [(b1_deg, r'$\beta_1$'), (b2_deg, r'$\beta_2$')]:
            ax.axvline(b_deg, color='gray', linestyle='--',
                       linewidth=1.2, label=lbl)

        ax.axhline(0, color='k', linewidth=0.6)
        lbl_state = ('singlet $|\\Phi^-\\rangle$' if self.state == 'singlet'
                    else 'triplet $|\\Phi^+\\rangle$')
        a_lbl = (fixed_alpha if fixed_alpha is not None
                 else np.rad2deg(alpha_rad))
        _style_ax(ax,
                  title=f'Correlation Sweep — {lbl_state},  '
                        f'$\\alpha = {a_lbl:.1f}^\\circ$,  $V = {self.visibility:.2f}$',
                  xlabel=r'$\beta$ (degrees)',
                  ylabel=r'$E(\alpha,\,\beta)$')
        ax.set_xlim(start, stop)
        ax.set_ylim(-1.15, 1.15)
        ax.legend(fontsize=9)
        plt.tight_layout()
        return ax


# ────────────────────────────────────────────────────────────────────────────────────
#  physical config blocks
# ────────────────────────────────────────────────────────────────────────────────────

class PumpBlock:
    """
    SPDC pump assembly: 405 nm diode + HWP + BBO crystal.

    The HWP angle selects which Bell state is produced.
    Default is triplet.
      triplet |phi+> = (|HH> + |VV>)/sqrt2  - HWP at 0°
      singlet |phi-> = (|HH> − |VV>)/sqrt2  - HWP at 22.5°
    """

    _KETS = {
        'singlet': np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),  # phi-
        'triplet': np.array([1, 0, 0,  1], dtype=complex) / np.sqrt(2),  # phi+
    }
    _HWP_ANGLES = {'triplet': 0.0, 'singlet': 22.5}   # degrees

    def __init__(self, params: SPDC, sim: SPDCSimulator):
        self._params     = params
        self._sim        = sim
        self._state_name = 'triplet'

    def waveplate(self, state: str):
        """
        Rotate the pump HWP to produce the given Bell state.

        Parameters
        ----------
        state : 'triplet' - |phi+>,  'singlet' - |phi->
        """
        if state not in self._KETS:
            raise ValueError(f"state must be 'triplet' or 'singlet', got '{state!r}'")
        self._state_name = state
        ket_str = ('$|\\Phi^+\\rangle = \\frac{{1}}{{\\sqrt{{2}}}}(|HH\\rangle + |VV\\rangle)$'
                   if state == 'triplet'
                   else '$|\\Phi^-\\rangle = \\frac{{1}}{{\\sqrt{{2}}}}(|HH\\rangle - |VV\\rangle)$')
        niceprint(f'state: {ket_str}')

    @property
    def state(self) -> str:
        return self._state_name

    @property
    def ket(self) -> np.ndarray:
        """4-component Bell state ket in {HH, HV, VH, VV} ordering."""
        return self._KETS[self._state_name].copy()

    @property
    def rho(self) -> np.ndarray:
        """Ideal 4x4 density matrix of the prepared Bell state."""
        psi = self.ket
        return np.outer(psi, psi.conj())

    @property
    def pair_rate(self) -> float:
        return self._sim.pair_rate(self._params.P_max)

    @property
    def coincidence_rate(self) -> float:
        return float(self._sim.Rcoin_W(np.array([self._params.P_max]))[0])

    def summary(self):
        niceprint('---')
        niceprint('**Pump Block**', 3)

        lbl = ('triplet $|\\Phi^+\\rangle$' if self._state_name == 'triplet'
               else 'singlet $|\\Phi^-\\rangle$')

        niceprint('<u> Source Configuration </u>', 5)
        niceprint(
            f'Bell state: {lbl} <br>'
            f'Pump HWP angle: {self._HWP_ANGLES[self._state_name]:.1f}° <br>'
            f'Pump wavelength: {self._params.lambda_pump*1e9:.0f} nm <br>'
            f'SPDC wavelength: {2*self._params.lambda_pump*1e9:.0f} nm <br>'
            f'SPDC efficiency: {self._params.spdc_efficiency:.0e}'
        )
        from .helpers import photon_energy
        E_pump = photon_energy(self._params.lambda_pump)
        E_spdc = photon_energy(2 * self._params.lambda_pump)
        N_pump = self._sim.pump_photon_rate(self._params.P_max)
        N_pair = self.pair_rate

        niceprint('<u> Rates at Operating Power </u>', 5)
        niceprint(
            f'Pump power $P_{{\\mathrm{{pump}}}}$: {self._params.P_max*1e3:.1f} mW <br>'
            f'Pump photon energy $E_{{\\mathrm{{pump}}}}$: {E_pump:.3e} J <br>'
            f'SPDC photon energy $E_{{\\mathrm{{SPDC}}}}$: {E_spdc:.3e} J <br>'
            f'Pump photon rate $\\dot{{N}}_{{\\mathrm{{pump}}}}$: {N_pump:.3e} photons/s <br>'
            f'Pair rate $\\dot{{N}}_{{\\mathrm{{pairs}}}}$: {N_pair:.3e} pairs/s <br>'
            f'Expected coincidence rate: {self.coincidence_rate:.0f} counts/s'
        )

        niceprint('<u> Bell State </u>', 5)
        niceprint(cleandisp(Qobj(self.ket), format='Dirac', return_str='Markdown'))
        niceprint('---')

    def __repr__(self) -> str:
        return (f"PumpBlock(state='{self._state_name}', "
                f"pair_rate={self.pair_rate:.2e} pairs/s)")



class DetectionArm:
    """
    One detection arm: [QWP (optional)] - HWP - PBS - fiber coupler - APD.

    Uses the QWP + HWP + PBS analysis circuit (Altepeter/Kwiat Eq. 77-78),
    which is operationally equivalent to the quED-TOM QWP + polarizer protocol
    (Table 2.1 of quED-TOM Manual V1.0).
    """

    def __init__(self, arm_id: int, wavelength: float = 810e-9):
        self.arm_id      = arm_id
        self._wavelength = wavelength
        self._qwp_angle: Optional[float] = None
        self._hwp_angle: float           = 0.0

    def insert(self, component: Union[QuarterWavePlate, float]):
        """Insert a QWP (QuarterWavePlate instance or fast-axis angle [rad])."""
        if isinstance(component, QuarterWavePlate):
            self._qwp_angle = component.angle
        elif isinstance(component, (int, float)):
            self._qwp_angle = float(component)
        else:
            raise TypeError(f"Expected QuarterWavePlate or float, got {type(component)}")

    def reset(self):
        """Remove QWP and reset HWP to 0 deg."""
        self._qwp_angle = None
        self._hwp_angle = 0.0

    def set_angles(self, qwp: float, hwp: float):
        """Set QWP and HWP fast-axis angles [rad]."""
        self._qwp_angle = qwp
        self._hwp_angle = hwp

    def set_projection(self, label: str):
        """Configure arm to project onto one of 'H', 'V', 'D', 'A', 'R', 'L'."""
        if label not in _SETTINGS:
            raise ValueError(f"label must be one of {list(_SETTINGS)}")
        qwp, hwp = _SETTINGS[label]
        self.set_angles(qwp, hwp)

    def get_projection_ket(self) -> np.ndarray:
        """
        Effective projection ket |m> for the current arm settings.
        Built as U^dag|H> where U = U_HWP(theta_HWP) U_QWP(theta_QWP).
        """
        circ = OpticalCircuit(wavelength=self._wavelength)
        if self._qwp_angle is not None:
            circ.add_qwp(fast_axis_angle=self._qwp_angle)
        circ.add_hwp(fast_axis_angle=self._hwp_angle)
        circ.add_pbs(port='transmitted')
        J    = circ.get_total_jones_matrix()
        proj = J.conj().T[:, 0]
        proj /= (np.linalg.norm(proj) + 1e-15)
        return proj

    def summary(self):
        qwp_str = (f'${np.rad2deg(self._qwp_angle):.1f}^\\circ$'
                   if self._qwp_angle is not None else 'not inserted')
        niceprint(
            f'Arm {self.arm_id}: '
            f'QWP = {qwp_str}, '
            f'HWP = {np.rad2deg(self._hwp_angle):.1f}°'
        )

    def __repr__(self) -> str:
        qwp_str = (f'{np.rad2deg(self._qwp_angle):.1f}°'
                   if self._qwp_angle is not None else 'none')
        return (f"DetectionArm(id={self.arm_id}, "
                f"QWP={qwp_str}, HWP={np.rad2deg(self._hwp_angle):.1f}°)")



class DetectionBlock:
    """
    Container for both detection arms.

    Usage
    -----
    app.block.insert('arm1', QuarterWavePlate(np.deg2rad(45)))
    app.block.reset('arm1')
    app.block.reset()
    app.block.summary()
    """

    def __init__(self, wavelength: float = 810e-9):
        self.arm1 = DetectionArm(1, wavelength)
        self.arm2 = DetectionArm(2, wavelength)

    def insert(self, arm: str, component: Union[QuarterWavePlate, float]):
        self._get_arm(arm).insert(component)

    def reset(self, arm: Optional[str] = None):
        """Reset arm(s) to default (no QWP, HWP at 0 deg). arm=None resets both."""
        if arm is None:
            self.arm1.reset(); self.arm2.reset()
        else:
            self._get_arm(arm).reset()

    def summary(self):
        niceprint('---')
        niceprint('**Detection Block**', 3)
        niceprint('<u> Arm Configuration </u>', 5)
        self.arm1.summary()
        self.arm2.summary()
        niceprint('---')

    def _get_arm(self, arm: str) -> DetectionArm:
        if arm == 'arm1': return self.arm1
        if arm == 'arm2': return self.arm2
        raise ValueError(f"arm must be 'arm1' or 'arm2', got '{arm}'")

    def __repr__(self) -> str:
        return f"DetectionBlock(\n  {self.arm1},\n  {self.arm2}\n)"


# ────────────────────────────────────────────────────────────────────────────────────
#  interfaces
# ────────────────────────────────────────────────────────────────────────────────────

class SPDCInterface:
    """SPDC diagnostic plots — attached to a ``quED`` instance as ``app.SPDC``."""

    _HV_SETTINGS   = [('HH',   0.0,   0.0), ('VV',  90.0,  90.0),
                      ('HV',   0.0,  90.0), ('VH',  90.0,   0.0)]
    _DIAG_SETTINGS = [('PP',  45.0,  45.0), ('MM', 135.0, 135.0),
                      ('PM',  45.0, 135.0), ('MP', 135.0,  45.0)]

    def __init__(self, parent: 'quED'):
        self._q = parent

    def _p_coin(self, alpha_deg, beta_deg):
       a, b = np.deg2rad(alpha_deg), np.deg2rad(beta_deg)
       if self._q.pump.state == 'triplet': return np.cos(a - b)**2 / 2.0
       else: return np.cos(a + b)**2 / 2.0

    def _simulate_basis(
        self, settings: List[Tuple], V: float, T_acq: float,
        n_trials: int, rng: np.random.Generator, R_background: float = 0.0
    ) -> Dict[str, Dict]:
        """
        Parameters
        ----------
        R_background : per-arm background rate [counts/s].  Default 0 (no change).
        """
        p      = self._q._params
        sim    = self._q._sim
        R_pair = sim.pair_rate(p.P_max)
    
        # Singles: background + SPDC signal (marginal state is I/2 - factor 0.5)
        R_s1  = R_background + 0.5 * p.eta_1 * R_pair
        R_s2  = R_background + 0.5 * p.eta_2 * R_pair
        R_max = p.eta_1 * p.eta_2 * R_pair   # coincidence: background negligible
    
        out = {}
        for label, alpha, beta in settings:
            P_c = V * self._p_coin(alpha, beta) + (1 - V) * 0.25
            out[label] = {
                'Arm 1':       rng.poisson(R_s1 * T_acq, size=n_trials).astype(float) / T_acq,
                'Arm 2':       rng.poisson(R_s2 * T_acq, size=n_trials).astype(float) / T_acq,
                'Coincidence': rng.poisson(P_c * R_max * T_acq, size=n_trials).astype(float) / T_acq,
            }
        return out

    def count_rates(
        self,
        basis:      str   = 'HV',
        visibility: float = 0.97,
        T_acq:      float = 15.0,
        n_trials:   int   = 3,
        R_background: float = 0.0,
        ax:         Optional[plt.Axes] = None,
        seed:       Optional[int]      = None,
    ) -> plt.Axes:
        """
        Simulate and plot count rates for the four polarizer settings in a basis.

        Parameters
        ----------
        basis      : 'HV' (H/V basis) or 'diag' (diagonal basis)
        visibility : fringe visibility [0, 1]
        T_acq      : acquisition time per setting [s]
        n_trials   : Poisson samples per setting (mimics repeated measurements)
        """
        settings = self._HV_SETTINGS if basis == 'HV' else self._DIAG_SETTINGS
        rng      = np.random.default_rng(seed)
        data = self._simulate_basis(settings, visibility, T_acq, n_trials, rng, R_background)

        channels = ['Arm 1', 'Arm 2', 'Coincidence']
        x        = np.arange(len(data))
        width    = 0.25

        if ax is None:
            _, ax = plt.subplots(figsize=(6.75, 2.2))

        for i, ch in enumerate(channels):
            offset = (i - len(channels) / 2 + 0.5) * width
            means  = [d[ch].mean() for d in data.values()]
            sems   = [d[ch].std(ddof=1) / np.sqrt(n_trials) if n_trials > 1
                      else 0.0 for d in data.values()]
            ax.bar(x + offset, means, yerr=sems,
                   width=width, label=ch, color=_SINGLES_COLOR[ch],
                   capsize=4, ecolor='black')

        if basis == 'HV':
            N_corr = (data['HH']['Coincidence'].mean() + data['VV']['Coincidence'].mean())
            N_anti = (data['HV']['Coincidence'].mean() + data['VH']['Coincidence'].mean())
        else:
            N_corr = (data['PP']['Coincidence'].mean() + data['MM']['Coincidence'].mean())
            N_anti = (data['PM']['Coincidence'].mean() + data['MP']['Coincidence'].mean())
        V_meas = (N_corr - N_anti) / (N_corr + N_anti) if (N_corr + N_anti) > 0 else 0.0

        basis_str = 'H/V' if basis == 'HV' else 'Diagonal'
        state_str = ('triplet $|\\Phi^+\\rangle$' if self._q.pump.state == 'triplet'
                     else 'singlet $|\\Phi^-\\rangle$')
        ax.set_xticks(x)
        ax.set_xticklabels(list(data.keys()), fontsize=12)
        ax.set_xlabel(f'Visibility $V = {V_meas:.1%}$', fontsize=14, fontweight='bold')
        _style_ax(ax,
                  title=f'Simulated Count Rates — {basis_str} Basis, {state_str}',
                  ylabel='Count Rate/s')
        ax.legend(fontsize=11)
        plt.tight_layout()
        return ax

    def power(
        self,
        range:    Tuple[int, int, int] = (27, 38, 1),
        basis:    str   = 'HV',
        T_acq:    float = 15.0,
        n_trials: int   = 3,
        R_background: float = 0.0,
        ax:       Optional[plt.Axes] = None,
        seed:     Optional[int]      = None,
    ) -> plt.Axes:
        """
        Simulate and plot Arm 1, Arm 2, and Coincidence rates vs pump current.

        Parameters
        ----------
        range : (start_mA, stop_mA, step_mA) — pump current sweep, inclusive
        basis : 'HV' or 'diag'
        """
        start, stop, step = range
        Is  = list(np.arange(start, stop + 1, step))[::-1]
        rng = np.random.default_rng(seed)

        p   = self._q._params
        sim = self._q._sim
        (_, alpha_c, beta_c) = (self._HV_SETTINGS[0] if basis == 'HV'
                                else self._DIAG_SETTINGS[0])

        channels  = ['Arm 1', 'Arm 2', 'Coincidence']
        all_means = {ch: [] for ch in channels}
        all_sems  = {ch: [] for ch in channels}

        for I_mA in Is:
            P      = float(sim.mA_to_W(np.array([I_mA]))[0])
            if P <= 0:
                for ch, R in zip(channels, [R_background, R_background, 0.0]):
                    samp = rng.poisson(R * T_acq, size=n_trials).astype(float) / T_acq
                    all_means[ch].append(samp.mean())
                    all_sems[ch].append(samp.std(ddof=1) / np.sqrt(n_trials)
                                        if n_trials > 1 else 0.0)
                continue
            R_pair = sim.pair_rate(P)
            R_s1 = R_background + p.eta_1 * R_pair
            R_s2 = R_background + p.eta_2 * R_pair
            P_c    = 0.97 * self._p_coin(alpha_c, beta_c) + (1 - 0.97) * 0.25
            R_coin = p.eta_1 * p.eta_2 * R_pair * P_c

            for ch, R in zip(channels, [R_s1, R_s2, R_coin]):
                samp = rng.poisson(R * T_acq, size=n_trials).astype(float) / T_acq
                all_means[ch].append(samp.mean())
                all_sems[ch].append(samp.std(ddof=1) / np.sqrt(n_trials)
                                    if n_trials > 1 else 0.0)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        for ch in channels:
            ax.errorbar(Is, all_means[ch], yerr=all_sems[ch],
                        fmt='o-', color=_SINGLES_COLOR[ch],
                        capsize=10, capthick=1.5, elinewidth=1.5,
                        linewidth=2, markersize=5, label=ch)

        ax.invert_xaxis()
        ax.set_xticks(Is)
        ax.set_xticklabels([str(I) for I in Is])
        basis_str = 'H/V' if basis == 'HV' else 'Diagonal'
        _style_ax(ax,
                  title=f'Simulated Count Rates vs Pump Power ({basis_str} Basis)',
                  xlabel='Pump Laser Current (mA)',
                  ylabel='Count Rate per Second')
        ax.legend(fontsize=11)
        plt.tight_layout()
        return ax
    
    def correlation_sweep(
        self,
        R_max:      Optional[float]                      = None,
        visibility: float                                = 0.97,
        beta_range: Tuple[float, float, float]           = (0, 180, 5),
        T_acq:      float                                = 1.0,
        ax:         Optional[plt.Axes]                   = None,
        seed:       Optional[int]                        = None,
    ) -> plt.Axes:
        """
        Simulate and plot coincidence counts vs Bob's angle beta for each of
        Alice's four canonical fixed angles {H, P, V, M}.
    
        Matches the quED experimental correlation measurement format with
        sinusoidal fits and CHSH angle markers.
    
        Parameters
        ----------
        R_max      : peak unpolarized coincidence rate [counts/s].
                    Defaults to η₁ η₂ R_pair at operating power.
        visibility : fringe visibility V [0, 1]
        beta_range : (start, stop, step) for Bob's angle sweep [degrees]
        T_acq      : acquisition time per data point [s]
                    (scales coincidence rate - total counts for Poisson sampling)
        """
        from scipy.optimize import curve_fit
    
        p   = self._q._params
        sim = self._q._sim
    
        if R_max is None:
            R_max = float(sim.Rcoin_W(np.array([p.P_max]))[0])
    
        state = self._q.pump.state
        rng   = np.random.default_rng(seed)
    
        start, stop, step = beta_range
        betas = np.arange(start, stop + step / 2, step)
    
        # Four Alice fixed angles matching your channels dict
        alice_settings = [
            ('H', 0.0,    r'H ($0^\circ$)',    'mediumvioletred'),
            ('P', 45.0,   r'P ($45^\circ$)',   'dodgerblue'),
            ('V', 90.0,   r'V ($90^\circ$)',   'mediumseagreen'),
            ('M', 135.0,  r'M ($135^\circ$)',  'gold'),
        ]
    
        # ── theory formula ────────────────────────────────────────────────────────
        # triplet |phi+>: R(alpha,beta) = (R_max/4)(1 + V cos 2(alpha−beta))
        # singlet |phi->: R(alpha,beta) = (R_max/4)(1 + V cos 2(alpha+beta))
        def r_theory(alpha_deg: float, beta_arr: np.ndarray) -> np.ndarray:
            a = np.deg2rad(alpha_deg)
            b = np.deg2rad(beta_arr)
            arg = 2 * (a - b) if state == 'triplet' else 2 * (a + b)
            return (R_max / 4) * (1 + visibility * np.cos(arg))
    
        # sinusoid model for fitting (same as your experimental code)
        def sinusoid(x, A, phi, C):
            return A * np.cos(2 * np.deg2rad(x) + phi) + C
    
        # CHSH optimal beta angles and labels (same for both states)
        chsh_betas  = [22.5, 67.5, 112.5, 157.5]
        chsh_labels = [r'$\beta$', r"$\beta'$",
                    r'$\beta^\perp$', r"$\beta'^\perp$"]
    
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
    
        beta_fit = np.linspace(start, stop, 500)
    
        for name, alpha, label, color in alice_settings:
            R_th  = r_theory(alpha, betas)
            # Poisson-sample simulated counts
            y_sim = rng.poisson(R_th * T_acq).astype(float)
    
            p0 = [R_max * visibility / 4, 0.0, R_max / 4]
            try:
                popt, _ = curve_fit(sinusoid, betas, y_sim, p0=p0, maxfev=5000)
            except RuntimeError:
                popt = p0
    
            ax.plot(betas, y_sim, 'o', color=color, markersize=4)
            ax.plot(beta_fit, sinusoid(beta_fit, *popt), '-',
                    color=color, linewidth=2, label=label)
    
        # CHSH angle markers (after curves are drawn so ylim is set)
        ax.autoscale(enable=True, axis='y')
        for bv, lbl in zip(chsh_betas, chsh_labels):
            if start <= bv <= stop:
                ax.axvline(bv, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
                # use axes-coordinate y so text sits at 93% height regardless of scale
                ax.text(bv - 3, 0.93, lbl, fontsize=13, color='k',
                        transform=ax.get_xaxis_transform())
    
        state_str = ('triplet $|\\Phi^+\\rangle$' if state == 'triplet'
                    else 'singlet $|\\Phi^-\\rangle$')
        def _fmt_tacq(t):
            """Format T_acq without rounding to zero for small values."""
            if t >= 1:
                return f'{t:.1f} s'
            elif t >= 1e-3:
                return f'{t*1e3:.1f} ms'
            else:
                return f'{t*1e6:.1f} $\\mu s$'
        _style_ax(ax,
                title=f'Correlation Sweep — {state_str},  $V = {visibility:.2f}$',
                xlabel=r'Bob angle $\beta$ (degrees)',
                ylabel=f'Coincidence counts ($T_{{\\rm acq}} = {_fmt_tacq(T_acq)}$)')
        ax.legend(fontsize=11, loc='upper right')
        plt.tight_layout()
        return ax



class PhotonStatsInterface:
    """Photon-statistics interface — attached as ``app.stats``."""

    def __init__(self, parent: 'quED'):
        self._q = parent

    def g2(
        self,
        mode:     str   = 'heralded',
        T_run:    float = 1.0,        # NOTE: 1 s is plenty; see below
        n_trials: int   = 200,
        seed:     Optional[int] = None,
    ) -> 'G2Result':
        """
        Simulate g^2(0).
 
        Parameters
        ----------
        mode     : 'heralded' or 'unheralded'
        T_run    : run duration per trial [s].
                   1 s with 200 trials is fast and converges well.
        n_trials : number of independent trials
 
        Expected results
        ----------------
        unheralded : g^2 ~ 1.0  (multi-mode SPDC - Poisson statistics)
        heralded   : g^2 << 0.5  (single-photon character; ideal - 0)
        """
        rng = np.random.default_rng(seed)
        p   = self._q._params
        sim = self._q._sim
 
        # ── calibrate time-bins from PAIR rate (not coincidence rate) ──────────
        R_pair  = sim.pair_rate(p.P_max)
        mu_tgt  = 0.01                          # <<1 to stay in Poissonian
        dt      = mu_tgt / R_pair               # ~ 27 ns per bin
        n_bins  = int(T_run / dt)
 
        if mode == 'unheralded':
            vals = self._sim_unheralded(rng, n_bins, mu_tgt, n_trials)
        else:
            vals = self._sim_heralded(rng, n_bins, dt, R_pair, p, n_trials)
 
        flux = R_pair
        mu   = mu_tgt
        return G2Result(g2_vals=vals, mode=mode, mu=mu, flux=flux)
 
    @staticmethod
    def _sim_unheralded(
        rng: np.random.Generator, n_bins: int,
        mu: float, n_trials: int,
    ) -> np.ndarray:
        """
        Unheralded HBT simulation for a Poisson (multi-mode SPDC) source.
        For Poisson light g2(0) = 1 by definition.
 
        The formula  g2 = N_12 * n_bins / (N1 * N2)  should converge to 1.
        All arithmetic is in float64 to prevent overflow.
        """
        # Each arm gets half the photons on average.
        N_total = rng.poisson(float(n_bins) * mu, size=n_trials).astype(np.float64)
        N1      = rng.binomial(N_total.astype(np.int64), 0.5).astype(np.float64)
        N2      = N_total - N1
 
        # Expected coincidences for Poisson: N_12 = N1 * N2 / n_bins
        N12     = rng.poisson(N1 * N2 / float(n_bins)).astype(np.float64)
 
        valid   = (N1 > 0) & (N2 > 0)
        g2      = np.full(n_trials, np.nan)
        g2[valid] = (N12[valid] * float(n_bins)) / (N1[valid] * N2[valid])
        return g2
 
    @staticmethod
    def _sim_heralded(
        rng: np.random.Generator, n_bins: int, dt: float,
        R_pair: float, p: 'SPDC', n_trials: int,
    ) -> np.ndarray:
        """
        Heralded g2_H(0) simulation.
        """
        mu_pair = R_pair * dt    # ~ 0.01 pairs/bin by construction
 
        g2_vals = []
        for _ in range(n_trials):
            N_pairs = int(rng.poisson(float(n_bins) * mu_pair))
            if N_pairs == 0:
                continue
 
            pair_bins = rng.integers(0, n_bins, size=N_pairs)
 
            # Herald arm (APD1): Bernoulli detection with eta_1
            h_mask  = rng.random(N_pairs) < p.eta_1
            # Signal arm: detection with eta_2, then 50:50 BS split
            s_mask  = rng.random(N_pairs) < p.eta_2
            s2_mask = rng.random(N_pairs) < 0.5   # - APD2 (ch 2)
            s3_mask = ~s2_mask                     # - APD3 (ch 3)
 
            h_bins   = set(pair_bins[h_mask].tolist())
            s2_bins  = set(pair_bins[s_mask &  s2_mask].tolist())
            s3_bins  = set(pair_bins[s_mask &  s3_mask].tolist())
 
            N1   = len(h_bins)
            N12  = len(h_bins  & s2_bins)
            N13  = len(h_bins  & s3_bins)
            N123 = len(h_bins  & s2_bins & s3_bins)
 
            if N12 > 0 and N13 > 0:
                g2_H = float(N123) * float(N1) / (float(N12) * float(N13))
                g2_vals.append(g2_H)
 
        return np.array(g2_vals) if g2_vals else np.array([np.nan])


# ────────────────────────────────────────────────────────────────────────────────────
#  grand unified quED!!!
# ────────────────────────────────────────────────────────────────────────────────────

class quED:
    """
    Unified model of the qutools quED entanglement demonstrator.

    Parameters
    ----------
    params : SPDC dataclass.  Defaults to SPDC() with quED-3 values.

    Examples
    --------
    >>> app = quED(SPDC(I_operating=38, eta_1=0.18, eta_2=0.22))
    >>> app.pump.waveplate('triplet')
    >>> app.summary()

    >>> app.SPDC.count_rates(basis='HV')
    >>> app.SPDC.power(range=(27, 38, 1))

    >>> result = app.tomo2q()
    >>> result.summary()

    >>> g2h = app.stats.g2('heralded')
    >>> g2h.summary()

    >>> chsh = app.CHSH('triplet')
    >>> chsh.summary()
    >>> chsh.correlations(angle=(0, 360, 5))
    """

    def __init__(self, params: Optional[SPDC] = None):
        self._params  = params or SPDC()
        self._sim     = SPDCSimulator(self._params)
        self._lam     = self._sim.lambda_spdc

        self.pump  = PumpBlock(self._params, self._sim)
        self.block = DetectionBlock(self._lam)
        self.SPDC  = SPDCInterface(self)
        self.stats = PhotonStatsInterface(self)

    # ── apparatus summary ────────────────────────────────────────────────────────
    def summary(self):
        """Full apparatus summary: source parameters, SPDC rates, current config."""
        p = self._params

        niceprint('---')
        niceprint('**quED Entanglement Demonstrator**', 3)

        niceprint('<u> Source Parameters </u>', 5)
        niceprint(
            f'Pump wavelength: {p.lambda_pump*1e9:.0f} nm <br>'
            f'SPDC wavelength: {self._lam*1e9:.0f} nm <br>'
            f'SPDC efficiency: {p.spdc_efficiency:.0e} <br>'
            f'Detection efficiencies: '
            f'$\\eta_1 = {p.eta_1:.2f},\\; \\eta_2 = {p.eta_2:.2f}$ <br>'
            f'Operating current: {p.I_operating:.1f} mA <br>'
            f'Threshold current: {p.I_threshold:.1f} mA'
        )

        from .helpers import photon_energy
        E_pump = photon_energy(p.lambda_pump)
        E_spdc = photon_energy(self._lam)
        N_pump = self._sim.pump_photon_rate(p.P_max)
        N_pair = self._sim.pair_rate(p.P_max)
        N_spdc = 2.0 * N_pair
        P_spdc = self._sim.spdc_W(p.P_max)
        R_max  = float(self._sim.Rcoin_W(np.array([p.P_max]))[0])  # unpolarized
        R_HH   = R_max * 0.5    # with polarizers at HH for triplet (P_coin = 1/2)
    
        niceprint('<u> SPDC Calculations </u>', 5)
        niceprint(
            f'Power: <br>'
            f'$\\quad P_{{\\mathrm{{pump}}}}$ = {p.P_max*1e3:.1f} mW <br>'
            f'$\\quad P_{{\\mathrm{{SPDC}}}}$ = {P_spdc*1e3:.1f} mW <br>'
            f'Energies: <br>'
            f'$\\quad E_{{\\mathrm{{pump}}}}$ = {E_pump:.3e} J <br>'
            f'$\\quad E_{{\\mathrm{{SPDC}}}}$ = {E_spdc:.3e} J <br>'
            f'Photon rates: <br>'
            f'$\\quad \\dot{{N}}_{{\\mathrm{{pump}}}}$ = {N_pump:.3e} photons/s <br>'
            f'$\\quad \\dot{{N}}_{{\\mathrm{{SPDC}}}}$ = {N_spdc:.3e} photons/s <br>'
            f'$\\quad \\dot{{N}}_{{\\mathrm{{pairs}}}}$ = {N_pair:.3e} pairs/s <br>'
            f'Max coincidence rate (no polarizers): {R_max:.0f} counts/s <br>'
            f'Expected HH/VV rate (with polarizers, $V=1$): {R_HH:.0f} counts/s'
        )

        state_lbl = ('triplet $|\\Phi^+\\rangle$' if self.pump.state == 'triplet'
                     else 'singlet $|\\Phi^-\\rangle$')
        arm1_qwp = (f'{np.rad2deg(self.block.arm1._qwp_angle):.1f}°'
                    if self.block.arm1._qwp_angle is not None else 'not inserted')
        arm2_qwp = (f'{np.rad2deg(self.block.arm2._qwp_angle):.1f}°'
                    if self.block.arm2._qwp_angle is not None else 'not inserted')

        niceprint('<u> Current Configuration </u>', 5)
        niceprint(
            f'Bell state: {state_lbl} <br>'
            f'Arm 1 — QWP: {arm1_qwp}, '
            f'HWP: {np.rad2deg(self.block.arm1._hwp_angle):.1f}° <br>'
            f'Arm 2 — QWP: {arm2_qwp}, '
            f'HWP: {np.rad2deg(self.block.arm2._hwp_angle):.1f}°'
        )
        niceprint('---')

    # ── tomography ───────────────────────────────────────────────────────────────

    def tomo1q(
        self,
        state,
        n_counts:        int   = 20_000,
        angle_error_rad: float = 0.0,
        seed:            Optional[int] = None,
    ) -> TomoResult:
        """
        Single-qubit quantum state tomography (quED-TOM 6-setting protocol,
        Table 2.1).  Uses QWP + HWP + PBS in one arm.

        Parameters
        ----------
        state           : Qobj (ket or dm) or np.ndarray
        n_counts        : mean photon counts per measurement setting
        angle_error_rad : uniform waveplate angle error [rad]
        seed            : random seed
        """
        rng     = np.random.default_rng(seed)
        rho_arr = self._to_rho(state)
        counts  = np.zeros(6, dtype=float)

        for k, label in enumerate(_LABELS_1Q):
            qwp, hwp = _SETTINGS[label]
            if angle_error_rad > 0:
                qwp += rng.uniform(-angle_error_rad, angle_error_rad)
                hwp += rng.uniform(-angle_error_rad, angle_error_rad)
            proj      = self._proj_ket(qwp, hwp)
            p_        = float(np.real(proj.conj() @ rho_arr @ proj))
            counts[k] = rng.poisson(max(p_, 0) * n_counts)

        rho_recon = density_matrix_1photon(counts)
        return TomoResult(rho_ideal=Qobj(rho_arr), rho_recon=rho_recon,
                          counts=counts, n_qubits=1)

    def tomo2q(
        self,
        state=None,
        n_counts:        int   = 20_000,
        angle_error_rad: float = 0.0,
        seed:            Optional[int] = None,
    ) -> TomoResult:
        """
        Two-qubit quantum state tomography (quED-TOM 36-setting protocol,
        section 2.2.1).  Uses QWP + HWP + PBS in both arms.

        Parameters
        ----------
        state           : Qobj / ndarray, shape 4x1 or 4x4.
                          If None, uses ``pump.ket`` (current Bell state).
        n_counts        : mean coincidence counts per measurement setting
        angle_error_rad : uniform waveplate angle error [rad]
        seed            : random seed
        """
        if state is None:
            psi   = self.pump.ket
            state = np.outer(psi, psi.conj())

        rng      = np.random.default_rng(seed)
        rho_arr  = self._to_rho(state)
        counts36 = np.zeros((6, 6), dtype=float)

        for a, la in enumerate(_LABELS_1Q):
            qwp_a, hwp_a = _SETTINGS[la]
            if angle_error_rad > 0:
                qwp_a += rng.uniform(-angle_error_rad, angle_error_rad)
                hwp_a += rng.uniform(-angle_error_rad, angle_error_rad)
            proj_a = self._proj_ket(qwp_a, hwp_a)

            for b, lb in enumerate(_LABELS_1Q):
                qwp_b, hwp_b = _SETTINGS[lb]
                if angle_error_rad > 0:
                    qwp_b += rng.uniform(-angle_error_rad, angle_error_rad)
                    hwp_b += rng.uniform(-angle_error_rad, angle_error_rad)
                proj_b = self._proj_ket(qwp_b, hwp_b)

                kab            = np.kron(proj_a, proj_b)
                p_             = float(np.real(kab.conj() @ rho_arr @ kab))
                counts36[a, b] = rng.poisson(max(p_, 0) * n_counts)

        rho_recon = density_matrix_2photon(counts36)
        return TomoResult(rho_ideal=Qobj(rho_arr), rho_recon=rho_recon,
                          counts=counts36, n_qubits=2)

    # ── CHSH ────────────────────────────────────────────────────────────────────

    def CHSH(
        self,
        state:      Optional[str]  = None,
        angles:     Optional[dict] = None,
        visibility: float          = 1.0,
        T_acq:      float          = 15.0,
    ) -> CHSHResult:
        """
        Simulate the CHSH Bell inequality experiment.

        Parameters
        ----------
        state      : 'singlet' or 'triplet'.  Defaults to ``pump.state``.
        angles     : dict with 'alpha' and 'beta' tuples [rad].
                     Defaults to the known optimal CHSH angles for the state.
        visibility : fringe visibility [0, 1]
        T_acq      : acquisition time per setting [s]
        """
        from .analyzers import BellInequalityAnalyzer

        state  = state  or self.pump.state
        angles = angles or _CHSH_ANGLES[state]

        ana        = BellInequalityAnalyzer(state=state, visibility=visibility,
                                            simulator=self._sim)
        ana.angles = angles

        result = ana.run_chsh(T_acq=T_acq)
        return CHSHResult(state=state, S=result['S'], sigma_S=result['sigma_S'],
                          settings=result['settings'], vis=visibility,
                          angles=angles, analyzer=ana)

    # ── internal helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_rho(state) -> np.ndarray:
        if isinstance(state, Qobj):
            return (state * state.dag()).full() if state.type == 'ket' else state.full()
        arr = np.asarray(state, dtype=complex)
        if arr.ndim == 1:
            return np.outer(arr, arr.conj())
        return arr

    def _proj_ket(self, qwp_angle: float, hwp_angle: float) -> np.ndarray:
        circ = _analysis_circuit(qwp_angle, hwp_angle, self._lam)
        J    = circ.get_total_jones_matrix()
        proj = J.conj().T[:, 0]
        proj /= (np.linalg.norm(proj) + 1e-15)
        return proj

    def __repr__(self) -> str:
        return (f"quED(\n"
                f"  pump  = {self.pump!r},\n"
                f"  block = {self.block!r}\n"
                f")")



