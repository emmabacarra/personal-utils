from typing import List
from dataclasses import dataclass



@dataclass
class CavityGeometry:
    """Bow-tie cavity geometry parameters (SI meters throughout)"""
    W: float  # Width of bow-tie cavity in m
    H: float  # Height of bow-tie cavity in m
    fconcave: float  # Focal length of concave mirror in m
    R_mirrors: List[float]  # Reflectivities [R_input, R_output, R_flat1, R_flat2]
    wavelength: float  # Wavelength in m


@dataclass
class LaserBeam:
    """Laser beam parameters (SI meters throughout)"""
    w0: float  # Beam waist radius in m
    z0_location: float  # Distance from reference point to beam waist in m
    wavelength: float  # Wavelength in m


@dataclass
class Telescope:
    """Telescope lens parameters (SI meters throughout)"""
    f1: float  # Focal length of first lens in m
    f2: float  # Focal length of second lens in m


@dataclass
class PiezoActuator:
    """Piezo actuator parameters"""
    displacement_per_volt: float  # nm/V
    voltage_amplitude: float  # V (peak amplitude)
    frequency: float  # Hz
    offset_voltage: float  # V (DC offset)


@dataclass
class Photodetector:
    """Photodetector parameters"""
    responsivity: float  # A/W
    load_resistance: float  # Ohms
    gain: float  # Additional amplifier gain

@dataclass
class SPDC:
    """
    SPDC parameters for the quED entanglement demonstrator

    Parameters
    ----------
    lambda_pump    : pump wavelength [m], default 405 nm (Blu-Ray diode)
    spdc_efficiency: probability per pump photon of producing one pair, default 1e-11
    eta_1, eta_2   : end-to-end detection efficiencies for arm 1 and arm 2
    P_max          : pump power at operating current [W], default 18 mW
    I_threshold    : laser diode threshold current [mA], default 26 mA
    I_operating    : laser diode operating current [mA], default 41 mA (quED-3)
    """
    lambda_pump:     float = 405e-9
    spdc_efficiency: float = 1e-11
    eta_1:           float = 0.20
    eta_2:           float = 0.20
    P_max:           float = 18e-3
    I_threshold:     float = 26.0
    I_operating:     float = 41.0

@dataclass
class BellMeasurement:
    """
    One row of a CHSH coincidence dataset.

    Parameters
    ----------
    alpha : Alice polarizer angle [degrees]
    beta  : Bob polarizer angle   [degrees]
    N_A   : Alice singles counts  (over acquisition time T)
    N_B   : Bob singles counts    (over acquisition time T)
    N     : Raw coincidence counts
    N_ac  : Accidental coincidences = tau * N_A * N_B / T
            where tau is the coincidence time window and T is the run length.
    """
    alpha: float
    beta:  float
    N_A:   float
    N_B:   float
    N:     float
    N_ac:  float = 0.0

    @property
    def N_net(self) -> float:
        """Net coincidences after subtracting accidentals."""
        return self.N - self.N_ac


