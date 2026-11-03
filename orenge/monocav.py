import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PTC10_COLS = ("heater", "thermistor", "outloop", "ambient")
COLORS = ["mediumvioletred", "deepskyblue", "darkorange", "forestgreen"]
CH_COLORS = {"thermistor": COLORS[0], "outloop": COLORS[1], "heater": COLORS[2], "ambient": COLORS[3]}


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


# ------------------------------------------------------------------ PTC10 log
def load_ptc10(path, dt=None, cols=PTC10_COLS, n_skip=0, n_per_dec=10, rename=None, **read_kw):
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
    omega = 2.0 * np.pi * f[keep]                                  # rad/s
    S = {c: psd_fft(y[c], fs)[1][keep] for c in cols}              # units^2/Hz
    Sb = {c: log_bin(omega, S[c], n_per_dec) for c in cols}        # (omega_bin rad/s, S_bin units^2/Hz)
    return {"path": path, "dt": dt, "N": N, "t": n * dt, "y": y, "units": units, "f": f[keep], "omega": omega, "S": S, "Sb": Sb}


def psd_ratio(log, num, den):
    return log["Sb"][num][0], log["Sb"][num][1] / log["Sb"][den][1]   # rad/s, dimensionless


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


# ------------------------------------------------------------------ plots
def plot_psd(log, chans=("thermistor", "outloop", "ambient"), ref="ambient", colors=CH_COLORS):
    fig, ax = plt.subplots(figsize=(9, 4))
    for c in chans:
        wb, Sb = log["Sb"][c]
        ax.loglog(log["omega"], log["S"][c], color=colors[c], alpha=0.15, linewidth=0.5)
        ax.loglog(wb, Sb, color=colors[c], linewidth=1.5, label=f"{c}")
        if ref is not None and c != ref:
            ax.loglog(*psd_ratio(log, c, ref), color=colors[c], linestyle="--", linewidth=1.5, label=fr"{c} / {ref} ($S_{{\mathrm{{{c}}}}}/S_{{\mathrm{{{ref}}}}}$)")
    ax.set_xlabel(r"angular frequency $\omega = 2\pi f$ (rad/s)")
    ax.set_ylabel("thermal noise PSD (K$^2$/Hz)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.show()
    return fig, ax


def plot_ratios(log, pairs=(("thermistor", "outloop"), ("thermistor", "ambient")), colors=CH_COLORS):
    fig, ax = plt.subplots(figsize=(7, 5))
    for num, den in pairs:
        ax.loglog(*psd_ratio(log, num, den), color=colors[den], linewidth=2, label=fr"{num} / {den} ($S_{{\mathrm{{{num}}}}}/S_{{\mathrm{{{den}}}}}$)")
    ax.set_xlim(log["omega"][0], log["omega"][-1])
    ax.set_xlabel(r"angular frequency $\omega = 2\pi f$ (rad/s)")
    ax.set_ylabel("PSD ratio")
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


def plot_can_cavity(log, tau, can_col="outloop", sigma=1e-3, f_c=0.1, f=np.logspace(-4, 4, 400)):
    omega_th = 2.0 * np.pi * f                                     # rad/s
    wb, Sb = log["Sb"][can_col]                                    # rad/s, K^2/Hz
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(log["omega"], log["S"][can_col], color="gray", alpha=0.25, linewidth=0.8, label=f"{can_col} data (raw)", zorder=1)
    ax.loglog(omega_th, S_Tcan(f, sigma, f_c), color="forestgreen", linewidth=2, label=fr"can, simulation ($f_c$ = {f_c:.2g} Hz)", zorder=3)
    ax.loglog(omega_th, S_Tcav(f, tau, sigma, f_c), color="magenta", linewidth=2, label="cavity, simulation", zorder=2)
    ax.loglog(wb, Sb, color="forestgreen", linestyle="--", linewidth=2, label="can, measured", zorder=3)
    ax.loglog(wb, Sb * cavity_filter(wb, tau), color="magenta", linestyle="--", linewidth=2, label="cavity, estimate", zorder=2)
    ax.axvline(1.0 / tau, color="magenta", linestyle=":", linewidth=1.5, alpha=0.4, label=fr"simulated $1/\tau$ = {1.0/tau:.2g} rad/s", zorder=2)
    ax.set_xlabel(r"angular frequency $\omega = 2\pi f$ (rad/s)")
    ax.set_ylabel(r"$S_T(\omega)$ (K$^2$/Hz)")
    ax.set_title("temperature noise: simulation vs data")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.show()
    return fig, ax