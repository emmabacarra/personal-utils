import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.special import jn_zeros, jv


PTC10_COLS = ("heater", "thermistor", "outloop", "ambient")
COLORS = ["mediumvioletred", "deepskyblue", "darkorange", "forestgreen"]
CH_COLORS = {"thermistor": COLORS[0], "outloop": COLORS[1], "heater": COLORS[2], "ambient": COLORS[3]}

C_LIGHT = 2.998e8                                                  # m/s
LAM = 729e-9                                                       # m
NU_LASER = C_LIGHT / LAM                                           # Hz, optical frequency at 729 nm
N_SILICA = 1.4547                                                  # refractive index of fused silica at 729 nm
DNDT = 8.7839e-6                                                   # 1/K, Leviton & Frey (2008), interpolated to 729 nm, 300 K
BETA = DNDT / N_SILICA                                             # 1/K, thermorefractive coefficient
KB = 1.380649e-23                                                  # J/K
L_CAV = 9.5e-3                                                     # m, cavity length
R_CAV = 12.5e-3                                                    # m, cavity radius
CP_VOL = 746.0 * 2202.0                                            # J/(m^3 K), volumetric heat capacity
KAPPA = 1.38                                                       # W/(m K)


# ------------------------------------------------------------------ spectra
def psd_fft(y, fs):
    n = np.arange(y.size)
    y = y - np.polyval(np.polyfit(n, y, 1), n)                    # remove mean + linear drift
    w = np.hanning(y.size)                                         # Hann window
    S = 2.0 * np.abs(np.fft.rfft(w * y)) ** 2 / (fs * np.sum(w ** 2))   # units^2/Hz, one-sided
    S[0] /= 2.0                                                    # DC bin is not doubled
    if y.size % 2 == 0:
        S[-1] /= 2.0                                               # Nyquist bin is not doubled
    return np.fft.rfftfreq(y.size, 1.0 / fs), S                    # Hz, units^2/Hz



def log_bin(x, S, n_per_dec=10):
    edges = 10 ** np.arange(np.log10(x[x > 0].min()), np.log10(x.max()) + 1.0 / n_per_dec, 1.0 / n_per_dec)
    idx = np.digitize(x, edges)
    bins = np.unique(idx)                                          # only occupied bins
    xb = np.array([np.exp(np.mean(np.log(x[idx == i]))) for i in bins])   # geometric bin centre
    Sb = np.array([np.mean(S[idx == i]) for i in bins])                   # mean PSD in bin
    return xb, Sb


def extrapolate_psd(fb, Sb, f0, fit_dec=1.0, f_to=1e4, n_per_dec=10):
    win = (fb >= f0 / 10 ** fit_dec) & (fb < f0)                   # fit window: fit_dec decades just below f0
    if np.count_nonzero(win) < 2:
        raise ValueError(f"need >= 2 binned points in [{f0 / 10 ** fit_dec:.3g}, {f0:.3g}) Hz to extrapolate; widen extpol_fit_dec or lower extpol_from_Hz")
    p = np.polyfit(np.log10(fb[win]), np.log10(Sb[win]), 1)        # log10 S = p[0]*log10 f + p[1]  (power law S = A f^alpha)
    f_ext = 10 ** np.arange(np.log10(f0), np.log10(f_to) + 0.5 / n_per_dec, 1.0 / n_per_dec)   # Hz, log grid f0 -> f_to
    keep = fb < f0                                                 # measured bins kept below f0
    f_out = np.concatenate([fb[keep], f_ext])                      # Hz
    S_out = np.concatenate([Sb[keep], 10 ** np.polyval(p, np.log10(f_ext))])   # units^2/Hz
    return f_out, S_out, p                                         # p[0] = power-law exponent alpha


# ------------------------------------------------------------------ PTC10 log
def load_ptc10(path, dt=None, cols=PTC10_COLS, n_skip=0, n_per_dec=10, rename=None,
               extpol_from_Hz=None, extpol_fit_dec=1.0, extpol_to_Hz=1e4, **read_kw):
    df = pd.read_csv(path, sep=None, engine="python", **read_kw)   # read_kw: e.g. skiprows=, comment=
    units = {}
    heads = {}
    for c in df.columns:
        m = re.match(r"^(.*) \((.*)\)$", str(c).strip())        # "thermistor (K)" -> ("thermistor", "K")
        name = m.group(1) if m else str(c).strip()
        heads[c] = name
        if m:
            units[name] = m.group(2)
    df = df.rename(columns=heads)
    if rename is not None:
        df = df.rename(columns=rename)                             # {file header: channel name}
        units = {rename.get(k, k): v for k, v in units.items()}
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{missing} not in {path}; columns are {list(df.columns)}; pass rename={{'file header': 'channel'}}")
    if dt is None:
        dt = np.median(np.diff(df["time_ms"].to_numpy(dtype=float)[n_skip:])) / 1000.0   # s, from logged timestamps
    df = df[list(cols)].apply(pd.to_numeric, errors="coerce").dropna(how="all").iloc[n_skip:]   # drop blank/footer rows, then transient
    N = len(df)                                                    # number of samples
    n = np.arange(N)                                               # sample index
    y = {}
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        good = np.isfinite(v)
        y[c] = np.interp(n, n[good], v[good])                      # channel units; fill dropouts
    fs = 1.0 / dt                                                  # Hz
    f = psd_fft(y[cols[0]], fs)[0]                                 # Hz
    keep = f > 0                                                   # drop DC
    S = {c: psd_fft(y[c], fs)[1][keep] for c in cols}              # units^2/Hz
    Sb = {c: log_bin(f[keep], S[c], n_per_dec) for c in cols}      # (f_bin Hz, S_bin units^2/Hz)
    Sb_meas = Sb                                                   # binned measured PSD, never extrapolated
    extpol = None
    if extpol_from_Hz is not None:
        f0 = float(extpol_from_Hz)                                 # Hz
        out = {c: extrapolate_psd(*Sb[c], f0, extpol_fit_dec, extpol_to_Hz, n_per_dec) for c in cols}
        Sb = {c: (out[c][0], out[c][1]) for c in cols}             # measured below f0 + power law above f0
        extpol = {"from_Hz": f0, "to_Hz": extpol_to_Hz, "fit_dec": extpol_fit_dec, "alpha": {c: out[c][2][0] for c in cols}}
    return {"path": path, "dt": dt, "N": N, "t": n * dt, "y": y, "units": units, "f": f[keep], "S": S,
            "Sb": Sb, "Sb_meas": Sb_meas, "extpol": extpol}


def psd_ratio(log, num, den):
    return log["Sb"][num][0], log["Sb"][num][1] / log["Sb"][den][1]   # Hz, dimensionless


# ------------------------------------------------------------------ Allan deviation
def adev_m(N, n_points=60):
    return np.unique(np.round(np.logspace(0, np.log10((N - 1) // 3), n_points)).astype(int))


def oadev(y, dt, m_list):
    x = np.concatenate([[0.0], np.cumsum(y) * dt])                 # K*s, integrated "phase"
    sig = np.array([np.sqrt(np.sum((x[2*m:] - 2*x[m:-m] + x[:-2*m]) ** 2) / (2.0 * (m*dt) ** 2 * (x.size - 2*m))) for m in m_list])
    return m_list * dt, sig                                        # s, K


# ------------------------------------------------------------------ COMSOL + can/cavity model
def fit_cooldown_tau(path, T0=300, n_sim=121, T_min=200, T_step=20, dt_sim=1.0):
    T_all = pd.read_csv(path, sep=",", header=None)[1].to_numpy(dtype=float)   # K
    i = list(T_min + T_step * np.arange(len(T_all) // n_sim)).index(T0)       # which start temperature
    T = T_all[i * n_sim:(i + 1) * n_sim]                                       # K
    t = np.arange(n_sim) * dt_sim                                              # s
    p, _ = curve_fit(lambda t, Ta, Ts, tau: Ts + (Ta - Ts) * np.exp(-t / tau), t, T, p0=[T[0], T[-1], t[-1] / 5.0])
    return p[2]                                                                # s


def cavity_filter(omega, tau):
    return 1.0 / (1.0 + (omega * tau) ** 2)                        # dimensionless, first-order low-pass |H|^2


def S_Tcan(f, sigma=1e-3, f_c=0.1):
    A = 2.0 * sigma ** 2 / (np.pi * f_c)                           # K^2/Hz
    return A / (1.0 + (f / f_c) ** 2)                              # K^2/Hz


def S_Tcav(f, tau, sigma=1e-3, f_c=0.1):
    return S_Tcan(f, sigma, f_c) * cavity_filter(2.0 * np.pi * f, tau)   # K^2/Hz


# ------------------------------------------------------------------ frequency noise
def S_nu_env(S_T):
    return (NU_LASER * BETA) ** 2 * S_T                            # Hz^2/Hz, from S_T in K^2/Hz
 
 
def K_n(a_n, R, r0):
    integrand = lambda xi: jv(0, a_n * xi) * np.exp(-((R / r0) * xi) ** 2) * xi
    return quad(integrand, 0, 1, epsabs=1e-12, epsrel=1e-10)[0]    # dimensionless
 
 
def Sz_hom(f, T0, H, R, r0, Cp, kappa, beta, N_modes=400):
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)               # rad/s
    a = jn_zeros(1, N_modes)                                       # dimensionless, zeros of J_1
    k = a / R                                                      # 1/m
    K = np.array([K_n(a_n, R, r0) for a_n in a])                   # dimensionless
    J0 = jv(0, a)                                                  # dimensionless
    prefactor = (16.0 / np.pi) * (KB * T0 ** 2) * (H * R ** 2 / r0 ** 4) * (kappa * beta ** 2) / Cp ** 2   # m^4/s
    denominator = omega[..., np.newaxis] ** 2 + (kappa / Cp) ** 2 * k ** 4    # 1/s^2
    return prefactor * np.sum((k ** 2 / J0 ** 2) * K ** 2 / denominator, axis=-1)   # m^2/Hz
 
 
def Sdv_over_v(f, L_cavity, T0, H, R, r0, Cp, kappa, beta, N_modes=400):
    return Sz_hom(f, T0, H, R, r0, Cp, kappa, beta, N_modes) / L_cavity ** 2   # 1/Hz
 
 
def S_nu_intrinsic(f, T0, r0, L_cavity=L_CAV, R=R_CAV, Cp=CP_VOL, kappa=KAPPA, beta=BETA, N_modes=400):
    return NU_LASER ** 2 * Sdv_over_v(f, L_cavity, T0, L_cavity, R, r0, Cp, kappa, beta, N_modes)   # Hz^2/Hz


# ------------------------------------------------------------------ plots
def _mark_extpol(ax, log):
    ext = log.get("extpol")
    if ext is not None:
        ax.axvline(ext["from_Hz"], color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
                   label=fr"power-law extrapolation above {ext['from_Hz']:.3g} Hz")

def plot_psd(log, chans=("thermistor", "outloop", "ambient"), ref="ambient", colors=CH_COLORS):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax_ratio = ax.twinx()
    for c in chans:
        fb, Sb = log["Sb"][c]                                      # Hz, K^2/Hz
        ax.loglog(log["f"], log["S"][c], color=colors[c], alpha=0.15, linewidth=0.5)
        ax.loglog(fb, Sb, color=colors[c], linewidth=1.5, label=f"{c}")
        if ref is not None and c != ref:
            ax_ratio.loglog(*psd_ratio(log, c, ref), color=colors[c], linestyle="--", linewidth=1.5, label=fr"{c} / {ref} ($S_{{\mathrm{{{c}}}}}/S_{{\mathrm{{{ref}}}}}$)")
    _mark_extpol(ax, log)
    ax.set_xlabel(r"frequency $f$ (Hz)")
    ax.set_ylabel("temperature noise PSD (K$^2$/Hz)")
    ax_ratio.set_ylabel("PSD ratio (dimensionless)")
    handles = ax.get_legend_handles_labels()
    handles_ratio = ax_ratio.get_legend_handles_labels()
    ax.legend(handles[0] + handles_ratio[0], handles[1] + handles_ratio[1], loc="lower left", fontsize=8)
    fig.tight_layout(); fig.show()
    return fig, ax


def plot_ratios(log, pairs=(("thermistor", "outloop"), ("thermistor", "ambient")), colors=CH_COLORS):
    fig, ax = plt.subplots(figsize=(7, 5))
    for num, den in pairs:
        ax.loglog(*psd_ratio(log, num, den), color=colors[den], linewidth=2, label=fr"{num} / {den} ($S_{{\mathrm{{{num}}}}}/S_{{\mathrm{{{den}}}}}$)")
    fb = log["Sb"][pairs[0][0]][0]                                 # Hz, includes extrapolated range
    ax.set_xlim(min(log["f"][0], fb[0]), max(log["f"][-1], fb[-1]))
    _mark_extpol(ax, log)
    ax.set_xlabel(r"frequency $f$ (Hz)")
    ax.set_ylabel("PSD ratio (dimensionless)")
    ax.set_title("temperature noise ratios")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.show()
    return fig, ax


def plot_adev(log, chans=("ambient", "outloop", "thermistor"), n_points=60, colors=CH_COLORS):
    m = adev_m(log["N"], n_points)
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in chans:
        tau, sig = oadev(log["y"][c], log["dt"], m)                # s, K
        ax.loglog(tau, sig, "o-", color=colors[c], markersize=3, label=c)
    ax.set_xlabel(r"averaging time $\tau$ (s)")
    ax.set_ylabel(r"$\sigma_T(\tau)$ (K)")
    ax.set_title("Allan deviation")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.show()
    return fig, ax


def plot_can_cavity(log, tau, can_col="thermistor", outloop_col="outloop", sigma=1e-3, f_c=0.1, f=np.logspace(-4, 4, 400)):
    f_corner = 1.0 / (2.0 * np.pi * tau)                           # Hz
    fig, ax = plt.subplots(figsize=(8, 5))
    # ax.loglog(log["f"], log["S"][can_col], color="gray", alpha=0.25, linewidth=0.8, label=f"{can_col} data (raw)", zorder=1)
    ax.loglog(f, S_Tcan(f, sigma, f_c), color="forestgreen", alpha=0.4, linewidth=2, label=fr"can, simulation ($f_c$ = {f_c:.2g} Hz)", zorder=3)
    ax.loglog(f, S_Tcav(f, tau, sigma, f_c), color="goldenrod", alpha=0.4, linewidth=2, label="cavity, simulation", zorder=2)
    for col, tag, color in ((can_col, "in-loop", COLORS[0]), (outloop_col, "out-of-loop", COLORS[1])):
        if col is None or (col == outloop_col and col == can_col and tag == "out-of-loop"):
            continue
        fb, Sb = log["Sb"][col]                                    # Hz, K^2/Hz, already extrapolated (if requested)
        ax.loglog(fb, Sb, color=color, linewidth=2, label=f"can, measured ({tag})", zorder=3)
        ax.loglog(fb, Sb * cavity_filter(2.0 * np.pi * fb, tau), color=color, linestyle="--", linewidth=2,
                  label=f"cavity, estimate ({tag})", zorder=2)   # transfer function applied after extrapolation
    # ax.axvline(f_corner, color="magenta", linestyle=":", linewidth=1.5, alpha=0.4, label=fr"simulated $1/(2\pi\tau)$ = {f_corner:.2g} Hz", zorder=2)
    _mark_extpol(ax, log)
    ax.set_xlabel(r"frequency $f$ (Hz)")
    ax.set_ylabel(r"$S_T(f)$ (K$^2$/Hz)")
    ax.set_title("temperature noise: simulation vs data")
    ax.legend(fontsize=7, handlelength=4)                          # long handles so -- vs -. are distinguishable
    fig.tight_layout(); fig.show()
    return fig, ax







