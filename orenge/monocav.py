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

def lowpass_fft(y, dt, f_cut):
    Y = np.fft.rfft(y)                                             # measured spectrum (amplitude + phase)
    Y[np.fft.rfftfreq(y.size, dt) > f_cut] = 0.0                   # brick-wall: remove f > f_cut, keep DC (mean)
    return np.fft.irfft(Y, y.size)                                 # channel units

def extrapolate_psd(fb, Sb, f0, fit_dec=1.0, f_to=1e4, n_per_dec=10):
    win = (fb >= f0 / 10 ** fit_dec) & (fb < f0)                   # fit window: fit_dec decades just below f0
    
    p = np.polyfit(np.log10(fb[win]), np.log10(Sb[win]), 1)        # log10 S = p[0]*log10 f + p[1]  (power law S = A f^alpha)
    
    f_ext = 10 ** np.arange(np.log10(f0), np.log10(f_to) + 0.5 / n_per_dec, 1.0 / n_per_dec)   # Hz, log grid f0 -> f_to
    
    keep = fb < f0                                                 # measured bins kept below f0
    f_out = np.concatenate([fb[keep], f_ext])                      # Hz
    S_out = np.concatenate([Sb[keep], 10 ** np.polyval(p, np.log10(f_ext))])   # units^2/Hz
    return f_out, S_out, p                                         # p[0] = power-law exponent alpha


# ------------------------------------------------------------------ PTC10 log
def load_ptc10(path, dt=None, cols=PTC10_COLS, n_skip=0, n_per_dec=10, rename=None,
               lowpass_Hz=None, lowpass_amb_Hz=None, extpol_fit_dec=1.0, extpol_to_Hz=1e4, **read_kw):
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
    lowpass = {c: (lowpass_amb_Hz if c == "ambient" else lowpass_Hz) for c in cols}   # Hz per channel, None = no low-pass
    y_lp = {c: y[c] if lowpass[c] is None else lowpass_fft(y[c], dt, lowpass[c]) for c in cols}   # channel units
    lp_cols = [c for c in cols if lowpass[c] is not None]         # channels that get extrapolated
    extpol = None
    if lp_cols:
        out = {c: extrapolate_psd(*Sb_meas[c], float(lowpass[c]), extpol_fit_dec, extpol_to_Hz, n_per_dec) for c in lp_cols}
        Sb = {c: (out[c][0], out[c][1]) if c in out else Sb_meas[c] for c in cols}   # measured below cutoff + power law above
        extpol = {"from_Hz": {c: float(lowpass[c]) for c in lp_cols}, "to_Hz": extpol_to_Hz, "fit_dec": extpol_fit_dec,
                  "alpha": {c: out[c][2][0] for c in lp_cols}}
    return {"path": path, "dt": dt, "N": N, "t": n * dt, "y": y, "y_lp": y_lp, "units": units, "f": f[keep], "S": S,
            "Sb": Sb, "Sb_meas": Sb_meas, "extpol": extpol, "lowpass": lowpass}



def psd_ratio(log, num, den):
    fn, Sn = log["Sb"][num]                                        # Hz, units^2/Hz
    fd, Sd = log["Sb"][den]                                        # Hz, units^2/Hz (grid may differ if cutoffs differ)
    Sd = 10 ** np.interp(np.log10(fn), np.log10(fd), np.log10(Sd), left=np.nan, right=np.nan)   # den on num grid; nan outside
    return fn, Sn / Sd                                             # Hz, dimensionless


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
def _mark_extpol(ax, log, chans=None):
    ext = log.get("extpol")
    if ext is None:
        return
    f0 = {c: v for c, v in ext["from_Hz"].items() if chans is None or c in chans}   # Hz per channel
    for v in sorted(set(f0.values())):
        who = ", ".join(c for c in f0 if f0[c] == v)
        ax.axvline(v, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
                   label=fr"power-law extrapolation above {v:.3g} Hz ({who})")

def plot_psd(log, chans=("thermistor", "outloop", "ambient"), ref="ambient", colors=CH_COLORS):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax_ratio = ax.twinx()
    blank = lambda: plt.Line2D([], [], linestyle="none")           # invisible legend handle (titles / padding)
    cols = []                                                      # one legend column per channel: [(handle, label), ...]
    for c in chans:
        fb, Sb = log["Sb"][c]                                      # Hz, K^2/Hz
        fm, Sm = log["Sb_meas"][c]                                 # Hz, K^2/Hz, original binned data
        ax.loglog(log["f"], log["S"][c], color=colors[c], alpha=0.15, linewidth=0.5)
        f_cut = log.get("lowpass", {}).get(c)                      # Hz or None
        col = [(blank(), c)]                                       # column title
        if f_cut is None:
            col.append((ax.loglog(fm, Sm, color=colors[c], linewidth=1.5)[0], "raw data"))
        else:
            lo = fm <= f_cut                                       # below the cutoff: kept as measured
            hi = fm >= fm[lo][-1] if np.any(lo) else np.ones_like(lo)   # above the cutoff: removed by the low pass (shares one point)
            ext = fb >= f_cut                                      # power-law extrapolation
            col.append((ax.loglog(fm[lo], Sm[lo], color=colors[c], linewidth=1.5)[0], "raw data"))
            col.append((ax.loglog(fm[hi], Sm[hi], color=colors[c], linestyle=":", linewidth=1.5)[0], f"{f_cut:.3g} Hz low pass"))
            col.append((ax.loglog(fb[ext], Sb[ext], color=colors[c], linestyle="-.", linewidth=1.5)[0], "extrapolation"))
        cols.append(col)
        if ref is not None and c != ref:
            ax_ratio.loglog(*psd_ratio(log, c, ref), color=colors[c], linestyle="--", linewidth=1.5, label=fr"{c} / {ref} ($S_{{\mathrm{{{c}}}}}/S_{{\mathrm{{{ref}}}}}$)")
    # _mark_extpol(ax, log, chans)
    ax.set_xlabel(r"frequency $f$ (Hz)")
    ax.set_ylabel("temperature noise PSD (K$^2$/Hz)")
    ax_ratio.set_ylabel("PSD ratio (dimensionless)")
    rows = max(len(col) for col in cols)
    cols = [col + [(blank(), "")] * (rows - len(col)) for col in cols]   # pad so every column has the same height
    entries = [e for col in cols for e in col]                     # column-major order (matplotlib fills legends by column)
    leg = ax.legend([h for h, _ in entries], [l for _, l in entries], ncol=len(cols), loc="lower left", fontsize=8, handlelength=3)
    for k, text in enumerate(leg.get_texts()):
        if k % rows == 0:
            text.set_fontweight("bold")                            # column titles
    if ax_ratio.get_legend_handles_labels()[0]:
        ax_ratio.legend(loc="upper right", fontsize=8)             # ratios in their own legend
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


def plot_freq_noise(log, tau, can_col="thermistor", outloop_col="outloop"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, tag, color in ((can_col, "in-loop", COLORS[0]), (outloop_col, "out-of-loop", COLORS[1])):
        if col is None or (col == outloop_col and col == can_col and tag == "out-of-loop"):
            continue
        fb, Sb = log["Sb"][col]                                    # Hz, K^2/Hz, already extrapolated (if requested)
        ax.loglog(fb, S_nu_env(Sb), color=color, linewidth=2, label=f"can, measured ({tag})")   # Hz^2/Hz
        ax.loglog(fb, S_nu_env(Sb * cavity_filter(2.0 * np.pi * fb, tau)), color=color, linestyle="--", linewidth=2,
                  label=f"cavity, estimate ({tag})")               # Hz^2/Hz, transfer function then thermorefractive conversion
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label=r"1 Hz$^2$/Hz")
    ax.axvline(100.0, color="gray", linestyle=":", linewidth=1.5, label="100 Hz")
    ax.set_xlabel(r"frequency $f$ (Hz)")
    ax.set_ylabel(r"$S_\nu(f)$ (Hz$^2$/Hz)")
    ax.set_title(r"frequency noise from temperature noise")
    ax.legend(fontsize=7, handlelength=4)
    fig.tight_layout(); fig.show()
    return fig, ax


# ------------------------------------------------------------------ time domain
def synth_ifft(log, col, fs=None, T=None, n_max=2**21, seed=0):
    ext = log["extpol"]
    extrapolated = ext is not None and col in ext["from_Hz"]       # this channel has a power-law tail
    f_top = ext["to_Hz"] if extrapolated else 0.5 / log["dt"]      # Hz, top of available PSD
    if T is None:
        T = log["N"] * log["dt"]                                   # s, measured record length
    if fs is None:
        fs = min(2.0 * f_top, n_max / T)                           # Hz, Nyquist = top of PSD, lowered so all of T fits in n_max                       # s, measured record length
    n = int(min(round(T * fs), n_max))                             # number of synthesized samples (capped)
    n -= n % 2                                                     # even length -> real Nyquist bin
    fb, Sb = log["Sb"][col]                                        # Hz, units^2/Hz (measured + extrapolated)
    f = np.fft.rfftfreq(n, 1.0 / fs)                               # Hz, uniform grid
    S = np.zeros_like(f)                                           # units^2/Hz, DC left at 0 (zero mean)
    S[1:] = 10 ** np.interp(np.log10(f[1:]), np.log10(fb), np.log10(Sb))   # log-log interpolation, held flat outside fb
    rng = np.random.default_rng(seed)
    X = np.sqrt(S * fs * n / 4.0) * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))   # Timmer & Koenig (1995): E|X|^2 = S fs n / 2
    X[-1] = X[-1].real                                             # Nyquist bin must be real
    return np.arange(n) / fs, np.fft.irfft(X, n)                   # s, units (zero-mean fluctuation)


def plot_ifft(log, chans=("thermistor", "outloop"), fs=None, T=None, n_max=2**21, seed=0, colors=CH_COLORS):
    fig, axes = plt.subplots(len(chans), 1, figsize=(9, 2.2 * len(chans) + 1), sharex=True, squeeze=False)
    for k, (ax, c) in enumerate(zip(axes[:, 0], chans)):
        t, x = synth_ifft(log, c, fs, T, n_max, seed + k)          # s, units; own seed per channel (independent phases)
        ax.plot(t / 3600.0, x, color=colors[c], linewidth=0.5, label=c)   # h
        ax.set_ylabel(fr"$\delta T$ ({log['units'].get(c, 'K')})")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)   # full numbers, no 1e-5 multiplier
        ax.legend(loc="upper right", fontsize=8)
    ext = log["extpol"]
    f0 = {c: ext["from_Hz"][c] for c in chans if ext is not None and c in ext["from_Hz"]}   # Hz per channel
    src = "PSD (extrapolated)" if f0 else "PSD (no extrapolation)"
    axes[0, 0].set_title(fr"IFFT of {src}, $T$ = {(t[-1] + t[1]) / 3600.0:.3g} h")
    axes[-1, 0].set_xlabel(r"time $t$ (h)")
    fig.tight_layout(); fig.show()
    return fig, axes


def plot_time(log, groups=(("thermistor", "outloop"), ("ambient",)), colors=CH_COLORS):
    fig, axes = plt.subplots(len(groups), 1, figsize=(9, 2.5 * len(groups) + 1), sharex=True, squeeze=False)
    for ax, grp in zip(axes[:, 0], groups):
        for c in grp:
            f_cut = log.get("lowpass", {}).get(c)                  # Hz or None
            ax.plot(log["t"], log["y_lp"][c], color=colors[c], linewidth=0.5,
                    label=c if f_cut is None else f"{c} (low-pass {f_cut:.3g} Hz)")   # s, channel units
        ax.set_ylabel(f"$T$ ({log['units'].get(grp[0], 'K')})")
        ax.legend(loc="upper right", fontsize=8)
    axes[0, 0].set_title(f"measured time series ({log['path']})")
    axes[-1, 0].set_xlabel(r"time $t$ (s)")
    fig.tight_layout(); fig.show()
    return fig, axes





