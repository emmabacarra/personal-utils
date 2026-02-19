from .general import *

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
import pandas as pd

from qutip import *
from qutip import Qobj

from typing import Literal


def locate_peaks(data, frequencies, pad = 5):
    peaks, _ = signal.find_peaks(data.abs())
    sorted_peaks = sorted(peaks, key=lambda x: np.abs(data[x]), reverse=True)
    peak_list = [(data.iloc[p], frequencies.iloc[p]) for p in sorted_peaks]
    
    if len(sorted_peaks) > 1:
        peak1, freq1 = peak_list[0]
        peak2, freq2 = peak_list[1]
        freq_min = np.min([freq1 - pad, freq1 + pad, freq2 - pad, freq2 + pad])
        freq_max = np.max([freq1 - pad, freq1 + pad, freq2 - pad, freq2 + pad])
        
        return peak_list, freq_min, freq_max
    if len(sorted_peaks) == 1:
        peak, freq = peak_list[0]
        freq_min = np.min([freq - pad, freq + pad])
        freq_max = np.max([freq - pad, freq + pad])
    
        return peak_list, freq_min, freq_max
    else:
        niceprint('No peaks found.')
        return [], None, None

class NMRGates:
    
    def __init__(self, J=215.0):
        self.J = J
        
        self.identity = qeye(2) # identity operator
        
        self.Ix = sigmax()/2
        self.Iy = sigmay()/2
        self.Iz = sigmaz()/2
        self.Izz = tensor(self.Iz, self.Iz)
        
        self.Uzz = (-1j * np.pi/4 * tensor(sigmaz(), sigmaz())).expm()       
            
    '''
    ===============================================
    --------------Single Qubit Gates---------------
    ===============================================
    '''
    # ============== Rotation Gates ===============
    def pulse(self, R: Literal['X', 'Y', 'Z'],theta, target: Literal[0,1] = None, phase=1, expm=True):
        if target is None: # single qubit
            px = (-1j * theta * self.Ix) * phase
            py = (-1j * theta * self.Iy) * phase
            pz = (-1j * theta * self.Iz) * phase

        if target == 0: # two qubit, first qubit
            px = tensor((-1j * theta * self.Ix), self.identity) * phase
            py = tensor((-1j * theta * self.Iy), self.identity) * phase
            pz = tensor((-1j * theta * self.Iz), self.identity) * phase

        if target == 1: # two qubit, second qubit
            px = tensor(self.identity, (-1j * theta * self.Ix)) * phase
            py = tensor(self.identity, (-1j * theta * self.Iy)) * phase
            pz = tensor(self.identity, (-1j * theta * self.Iz)) * phase

        if expm:
            return px.expm() if R == 'X' else py.expm() if R == 'Y' else pz.expm()
        else:
            return px if R == 'X' else py if R == 'Y' else pz

    # =============== Hadamard Gate ===============
    def H(self, target: Literal[0,1] = None):
        if target is None:
            H = self.pulse('Y', np.pi/2) * self.pulse('Z', np.pi) 
            return H * 1j
        if target == 0:
            H = tensor(self.pulse('Y', np.pi/2, expm=False), self.identity).expm() * tensor(self.pulse('Z', np.pi, expm=False), self.identity).expm()
        if target == 1:
            H = tensor(self.identity, self.pulse('Y', np.pi/2, expm=False)).expm() * tensor(self.identity, self.pulse('Z', np.pi, expm=False)).expm()
        return H * 1j

    # =========== Pseudo-Hadamard Gate ============
    def H_pseudo(self, inverse=False, target: Literal[0,1] = None, expm=True):
        y_theta = -np.pi/2 if not inverse else np.pi/2 # y-bar: inverse=False
        if target is None:
            return self.pulse('Y', y_theta, expm=expm) 
        if target == 0:
            return tensor(self.pulse('Y', y_theta, expm=expm), self.identity)
        if target == 1:
            return tensor(self.identity, self.pulse('Y', y_theta, expm=expm))
    '''
    ===============================================
    ----------------Two Qubit Gates----------------
    ===============================================
    '''
    # ================= CZ Gate ===================
    def CZ(self, appx = False):
        if not appx:
            p1z = self.pulse('Z', -np.pi/2, target=0)
            p2z = self.pulse('Z', -np.pi/2, target=1)
        if appx:
            p1z = (self.pulse('Y', np.pi/2, target=0) * 
                   self.pulse('X', np.pi/2, target=0) * 
                   self.pulse('Y', -np.pi/2, target=0))
            
            p2z = (self.pulse('Y', np.pi/2, target=1) * 
                   self.pulse('X', np.pi/2, target=1) * 
                   self.pulse('Y', -np.pi/2, target=1))
            
        phase_correction = np.exp(-1j * np.pi/4)
        return p1z * p2z * self.Uzz * phase_correction

    # ================ CNOT Gate ==================
    def CNOT(self, target: Literal[0,1], appx=False, reduce=False):
        if not reduce:
            if not appx:
                H = self.H(target=target)
                gate = H * self.CZ() * H
            if appx:
                H_pseudo = self.H_pseudo(target=target)
                H_pseudo_inv = self.H_pseudo(inverse=True, target=target)
                gate =  H_pseudo * self.CZ(appx=True) * H_pseudo_inv
        if reduce:
            gate = (self.pulse('X', np.pi/2, target=target) * # acquire pulse
                    self.pulse('Y', -np.pi/2, target=target) *
                    self.Uzz *
                    self.pulse('Y', np.pi/2, target=target)
                    )
        return gate
    
    '''
    ===============================================
    -------------Convenience Functions-------------
    ===============================================
    '''    
    def simulate(self, H, rho, t_array, T2=None, freq_range=200, plot=True, title=None, 
                 view: Literal['H', 'H_t', 'H_f', 'C', 'C_t', 'C_f', 'all', 'freqs'] = 'all'):
        signalH = np.zeros(len(t_array), dtype=complex)
        signalC = np.zeros(len(t_array), dtype=complex)
        
        # time evolve
        for idx, t in enumerate(t_array):
            if plot:
                print(f'Simulating point {idx+1} / {len(t_array)}    ', end='\r')
            
            Uzz = (-1j * H * t).expm()
            decay = np.exp(-t/T2) if T2 is not None else 1 # for linewidth
            obsH = Uzz.dag() * tensor(sigmap(), self.identity) * Uzz
            obsC = Uzz.dag() * tensor(self.identity, sigmap()) * Uzz
            
            # observe signal
            '''
            S(t) = trace(rho_t * sigmap())
                = trace(Uzz * rho * Uzz.dag() * sigmap())
                = trace(rho * Uzz.dag() * sigmap() * Uzz)
                = trace(rho_t * sigmap().dag())
            '''
            p1y = self.pulse('Y', np.pi/2, target=0)
            p2y = self.pulse('Y', np.pi/2, target=1)
            signalH[idx] = (obsH * p1y * rho * p1y.dag()).tr() * decay
            signalC[idx] = (obsC * p2y * rho * p2y.dag()).tr() * decay
       
        
        # fourier transform
        dt = t_array[1] - t_array[0]
        spectrumH = np.fft.fftshift(np.fft.fft(signalH))
        spectrumC = np.fft.fftshift(np.fft.fft(signalC))
        freqH = np.fft.fftshift(np.fft.fftfreq(len(t_array), d=dt))
        freqC = np.fft.fftshift(np.fft.fftfreq(len(t_array), d=dt))
        
        
        H_peaks, _ = signal.find_peaks(np.abs(spectrumH))
        H_peaks = sorted(H_peaks, key=lambda x: np.abs(spectrumH[x]), reverse=True)
        H_peak_list = [np.real(spectrumH[p]) for p in H_peaks]
        
        C_peaks, _ = signal.find_peaks(np.abs(spectrumC))
        C_peaks = sorted(C_peaks, key=lambda x: np.abs(spectrumC[x]), reverse=True)
        C_peak_list = [np.real(spectrumC[p]) for p in C_peaks]
        
        if not plot:
            simdata = {
                'H': (signalH, freqH, spectrumH, H_peak_list),
                'C': (signalC, freqC, spectrumC, C_peak_list)
            }
            return simdata
        
        
        if view == 'all':
            plt.figure(figsize=(13,6))
            
            # plot proton
            plt.subplot(2,2,1)
            plt.plot(t_array, signalH);
            plt.title('Proton', weight='bold')
            
            plt.subplot(2,2,3)
            plt.plot(freqH, spectrumH);
            plt.xlim(-freq_range, freq_range);
            plt.gca().invert_xaxis()

            # plot carbon
            plt.subplot(2,2,2)
            plt.plot(t_array, signalC);
            plt.title('Carbon', weight='bold')

            plt.subplot(2,2,4)
            plt.plot(freqC, spectrumC);
            plt.xlim(-freq_range, freq_range);
            plt.gca().invert_xaxis()
            
            if len(H_peaks) > 1 and len(C_peaks) > 1:
                H_p1, H_p2 = H_peak_list[1], H_peak_list[0]
                C_p1, C_p2 = C_peak_list[1], C_peak_list[0]

                niceprint(r'**$^1$ H ratio:**  $\quad$ ' + f'{H_p1} / {H_p2} = {H_p1/H_p2:.3f} <br>' +
                        r'**$^{13}$ C ratio:**  $\quad$ ' + f'{C_p1} / {C_p2} = {C_p1/C_p2:.3f}')
            else:
                niceprint(r'**$^1$ H peak:**  $\quad$ ' + f'{H_peak_list[0]:.3f} <br>' +
                        r'**$^{13}$ C peak:**  $\quad$ ' + f'{C_peak_list[0]:.3f}')
        
        if view == 'freqs':
            plt.figure(figsize=(13,4))
            
            # plot proton
            plt.subplot(1,2,1)
            plt.title('Proton', weight='bold')
            plt.plot(freqH, spectrumH);
            plt.xlim(-freq_range, freq_range);
            plt.gca().invert_xaxis()

            # plot carbon
            plt.subplot(1,2,2)
            plt.title('Carbon', weight='bold')
            plt.plot(freqC, spectrumC);
            plt.xlim(-freq_range, freq_range);
            plt.gca().invert_xaxis()
            
            if len(H_peaks) > 1 and len(C_peaks) > 1:
                H_p1, H_p2 = H_peak_list[1], H_peak_list[0]
                C_p1, C_p2 = C_peak_list[1], C_peak_list[0]

                niceprint(r'**$^1$ H ratio:**  $\quad$ ' + f'{H_p1} / {H_p2} = {H_p1/H_p2:.3f} <br>' +
                        r'**$^{13}$ C ratio:**  $\quad$ ' + f'{C_p1} / {C_p2} = {C_p1/C_p2:.3f}')
            else:
                niceprint(r'**$^1$ H peak:**  $\quad$ ' + f'{H_peak_list[0]:.3f} <br>' +
                        r'**$^{13}$ C peak:**  $\quad$ ' + f'{C_peak_list[0]:.3f}')
        
        elif view == 'H' or view == 'C':
            plt.figure(figsize=(13,4))
            
            signal_data = signalH if view == 'H' else signalC
            spectrum = spectrumH if view == 'H' else spectrumC
            freqs = freqH if view == 'H' else freqC
            
            plt.subplot(1,2,1)
            plt.plot(t_array, signal_data);
            plt.title('Proton' if view == 'H' else 'Carbon', weight='bold')
            
            plt.subplot(1,2,2)
            plt.plot(freqs, spectrum);
            plt.xlim(-freq_range, freq_range);
            plt.gca().invert_xaxis()
            
            peaks = H_peaks if view == 'H' else C_peaks
            peak_list = H_peak_list if view == 'H' else C_peak_list
            name = r'$^1$ H' if view == 'H' else r'$^{13}$ C'
            if len(peaks) > 1:
                peak1, peak2 = peak_list[1], peak_list[0]
                niceprint(fr'**{name} ratio:**  $\quad$ ' + f'{peak1} / {peak2} = {peak1/peak2:.3f}')
            else:
                niceprint(rf'**{name} peak:**  $\quad$ ' + f'{H_peak_list[0]:.3f}')
        
        elif 'H_' in view or 'C_' in view:
            plt.figure(figsize=(10,4))
            
            signal_data = signalH if 'H_' in view else signalC
            spectrum = spectrumH if 'H_' in view else spectrumC
            freqs = freqH if 'H_' in view else freqC
            
            if 't' in view:
                plt.plot(t_array, signal_data);
                plt.title('Proton' if 'H_' in view else 'Carbon', weight='bold')
                dat = (t_array, signal_data)
                
            if 'f' in view:
                plt.plot(freqs, spectrum);
                plt.xlim(-freq_range, freq_range);
                plt.gca().invert_xaxis()
                dat = (freqs, spectrum)
            
            if title is not None:
                plt.suptitle(title, weight='bold')
            plt.tight_layout()
            plt.show()
            return dat 
        
        if title is not None:
            plt.suptitle(title, weight='bold')
        plt.tight_layout()
        plt.show()
        
        return [H_peak_list, C_peak_list]

    # matrix form of a gate
    def matrix(self, gate):
        return gate.full()
    
    # check if constructed gate matches expected gate
    def check_gate(self, constructed_gate, expected_gate, tolerance=1e-10):
        
        constructed_gate = constructed_gate.full() if isinstance(constructed_gate, Qobj) else constructed_gate
        
        expected_gate = expected_gate.full() if isinstance(expected_gate, Qobj) else expected_gate

        # fidelity (up to global phase)
        overlap = np.abs(np.trace(constructed_gate.conj().T @ expected_gate)) # @ is numpy matrix multiplication
        fidelity = overlap / expected_gate.shape[0]
        print(f'Fidelity: {fidelity}')
        
        matches = (np.abs(fidelity - 1.0) < tolerance)
        
        # return matches, fidelity

def shift_and_scale(rho_dict, center_freq):
    freqs, datas, peaks, sim_peaks = rho_dict
    
    scale = np.abs(sim_peaks[0] / peaks[0][0] if abs(peaks[0][0]) >= abs(peaks[1][0]) 
                   else sim_peaks[1] / peaks[1][0]) * 0.001
    
    data_normalized = datas / np.max(np.abs(datas))
    norm_then_scale = data_normalized * scale
    
    datas_scaled = datas * scale
    # scale_then_norm = datas_scaled / np.max(np.abs(datas_scaled))
    
    # data_new = norm_then_scale if norm_then_scale.mean() < scale_then_norm.mean() else scale_then_norm
    
    shift = center_freq - np.mean([peaks[0][1], peaks[1][1]])
    freqs_shifted = freqs + shift
    
    return freqs_shifted, datas_scaled

def combine_data(freqs_datas_list):
    freq_max, freq_min = max([f[0].max() for f in freqs_datas_list]), min([f[0].min() for f in freqs_datas_list])
    
    n_points = len(freqs_datas_list[0][0])
    freqs_common = np.linspace(freq_min, freq_max, n_points)
    
    f_sum = np.zeros_like(freqs_common)
    for freq, data in freqs_datas_list:
        interpolation = np.interp(freqs_common, freq, data)
        f_sum += interpolation
    
    return freqs_common, f_sum

class PseudoPure(NMRGates):
    
    def __init__(self, state: Literal['00', '01', '10', '11'], H_J, T2, t_array):
        super().__init__() # inherit parent init
        
        self.state = state
        self.H_J = H_J
        self.T2 = T2
        self.t_array = t_array
        
        self.identity = qeye(2) # identity operator
        self.Ix = sigmax()/2
        self.Iy = sigmay()/2
        self.Iz = sigmaz()/2
        self.Izz = tensor(self.Iz, self.Iz)
        self.Uzz = (-1j * np.pi * self.Izz).expm()
        
        self.flip_0 = self.pulse('X', np.pi, target=0) * 1j
        self.flip_1 = self.pulse('X', np.pi, target=1) * 1j
        self.flip_01 = tensor(self.pulse('X', np.pi), self.pulse('X', np.pi)) * 1j
        
        self.rho_init = - 2 * ( - tensor(self.identity, self.Iz) - 4 * tensor(self.Iz, self.identity) )
        
        self.P1 = (self.pulse('Y', -np.pi/2, target=1) * 
                   self.Uzz * 
                   self.pulse('X', np.pi/2, target=1) *
                   self.pulse('Y', -np.pi/2, target=0) * 
                   self.Uzz * 
                   self.pulse('X', np.pi/2, target=0))

        self.P2 = (self.pulse('Y', -np.pi/2, target=0) * 
            self.Uzz * 
            self.pulse('Y', -np.pi/2, target=1) * 
            self.pulse('X', np.pi/2, target=0) * 
            self.Uzz * 
            self.pulse('X', np.pi/2, target=1))
        # self.P2 = self.P1.dag()
        
        if state == '00':
            self.rho0 = self.rho_init
            
        if state == '01':
            self.rho0 = self.flip_1 * self.rho_init * self.flip_1.dag()
            self.P1 = self.flip_1 * self.P1 * self.flip_1.dag()
            self.P2 = self.flip_1 * self.P2 * self.flip_1.dag()
        
        if state == '10':
            self.rho0 = self.flip_0 * self.rho_init * self.flip_0.dag()
            self.P1 = self.flip_0 * self.P1 * self.flip_0.dag()
            self.P2 = self.flip_0 * self.P2 * self.flip_0.dag()
        
        if state == '11': 
            self.rho0 = self.flip_01 * self.rho_init * self.flip_01.dag()
            self.P1 = self.flip_01 * self.P1 * self.flip_01.dag()
            self.P2 = self.flip_01 * self.P2 * self.flip_01.dag()
        
        self.rho1 = self.P1 * self.rho0 * self.P1.dag()
        self.rho2 = self.P2 * self.rho0 * self.P2.dag()
        self.rho_avg = (self.rho0 + self.rho1 + self.rho2) / 3
        
        self.storable_keys = ['current', 'simdata', 'H_path', 'C_path', 'expH', 'expC']
        self.active_matrix = {
            'name': None,
            'initial': None,
            'current': None,
            'simdata': None,
            'expH': None,
            'expC': None
        }
        self.dm_dict = {
            'rho0': {},
            'rho1': {},
            'rho2': {},
            'rho_avg': {}
        }
    
    def truth_table(self, sequence, sequence_name='Gate Sequence'):
        P1_str = cleandisp(self.P1, return_str='Latex')
        P1dag_str = cleandisp(self.P1.dag(), return_str='Latex')

        P2_str = cleandisp(self.P2, return_str='Latex')
        P2dag_str = cleandisp(self.P2.dag(), return_str='Latex')
        
        seq_str = cleandisp(sequence, return_str='Latex')
        seqdag_str = cleandisp(sequence.dag(), return_str='Latex')

        density_matrices = {
            r'\rho_0': {'equ_str': None,
                        'permutation': None,
                        'equ_steps': [self.rho0 * sequence, sequence.dag() * self.rho0 * sequence],
                        'matrix': self.rho0,
                        'rho_str': cleandisp(self.rho0, return_str='Latex')},
            
            r'\rho_1': {'equ_str': r'\rho_1 \rho_0 \rho_1^{\dag}',
                        'permutation': [P1_str, P1dag_str],
                        'equ_steps': [self.rho0 * self.P1.dag(), self.P1 * (self.rho0 * self.P1.dag())],
                        'matrix': self.rho1,
                        'rho_str': cleandisp(self.rho1, return_str='Latex')},
            
            r'\rho_2': {'equ_str': r'\rho_2 \rho_0 \rho_2^{\dag}',
                        'permutation': [P2_str, P2dag_str],
                        'equ_steps': [self.rho0 * self.P2.dag(), self.P2 * (self.rho0 * self.P2.dag())],
                        'matrix': self.rho2,
                        'rho_str': cleandisp(self.rho2, return_str='Latex')},
            
            r'\rho_{avg}': {'equ_str': r'\frac{1}{3}(\rho_0 + \rho_1 + \rho_2)',
                            'permutation': None,
                            'equ_steps': None,
                            'matrix': self.rho_avg,
                            'rho_str': cleandisp(self.rho_avg, return_str='Latex')}
        }

        # region truth table 
        for i, (name, info_dict) in enumerate(density_matrices.items()):
            niceprint('---')
            niceprint(fr'<u>Applying {sequence_name} to state ${name}$</u>', 4)
            
            # region print initial matrix
            niceprint('Initial Density Matrix:', 5)
            if info_dict['permutation'] is not None:
                init_steps = [cleandisp(step, return_str='Latex') for step in info_dict['equ_steps']]
                niceprint(fr"""\begin{{equation*}}
                            \begin{{aligned}}
                                    {name} &= {info_dict['equ_str']} \\
                                        &= {info_dict['permutation'][1]} \, {info_dict['rho_str']} \, {info_dict['permutation'][0]} \\
                                        &= {info_dict['permutation'][1]} \, {init_steps[0]} \\
                                        &= {init_steps[1]}
                            \end{{aligned}}
                            \end{{equation*}}""", method='Latex')
            else:
                niceprint(rf"\begin{{equation*}} {name} = {info_dict['rho_str']} \end{{equation*}}", method='Latex')
            # endregion
                
            # region apply gate sequence to density matrix
            niceprint(f'Applying {sequence_name}:', 5)
            if i < 3:
                step_1 = info_dict['matrix'] * sequence
                step_1_str = cleandisp(step_1, return_str='Latex')
                
                step_2 = sequence.dag() * step_1
                step_2_str = cleandisp(step_2, return_str='Latex')
                
                niceprint(fr"""\begin{{equation*}}
                            \begin{{aligned}}
                                U_{{\text{{{sequence_name}}}}}\,{name}\,U_{{\text{{{sequence_name}}}}}^{{\dag}} &= {seqdag_str} \, {info_dict['rho_str']} \, {seq_str} \\
                                                                                    &= {seqdag_str} \, {step_1_str} \\
                                                                                    &= {step_2_str}
                            \end{{aligned}}
                            \end{{equation*}}""", method='Latex')
                info_dict['afterGateSeq'] = step_2
                info_dict['afterGateSeq_str'] = step_2_str
            elif i == 3:
                rhokeys = [r'\rho_0', r'\rho_1', r'\rho_2']
                
                GateSeq_sum = (density_matrices[rhokeys[0]]['afterGateSeq'] +
                            density_matrices[rhokeys[1]]['afterGateSeq'] +
                            density_matrices[rhokeys[2]]['afterGateSeq'])
                GateSeq_avg = GateSeq_sum / 3
                
                info_dict['afterGateSeq'] = GateSeq_avg
                GateSeq_sum_str = (f"{density_matrices[rhokeys[0]]['afterGateSeq_str']}" 
                            f" + {density_matrices[rhokeys[1]]['afterGateSeq_str']}" 
                            f" + {density_matrices[rhokeys[2]]['afterGateSeq_str']}")
                GateSeq_avg_str = cleandisp(GateSeq_avg, return_str='Latex')
                info_dict['afterGateSeq_str'] = GateSeq_avg_str
                niceprint(rf"""\begin{{equation*}}
                            \begin{{aligned}}
                                U_{{\text{{{sequence_name}}}}}\,{name}\,U_{{\text{{{sequence_name}}}}}^{{\dag}} &= {seqdag_str} \, {info_dict['equ_str']} \, {seq_str} \\
                                                                                    &= \frac{{1}}{{3}} \, \left( {GateSeq_sum_str} \right) \\
                                                                                    &= {GateSeq_avg_str}
                            \end{{aligned}}
                            \end{{equation*}}""", method='Latex')
            # endregion
            
            # simulate!
            self.simulate(self.H_J, info_dict['afterGateSeq'], self.t_array, self.T2, 
                          title=fr'Simulating ${name}$ in {self.state} state with {sequence_name}')

    def update_dm_dict(self):
        if self.active_matrix is None:
            raise ValueError("No active matrix. Run 'run_simulation' first.")
        
        current_dict = self.dm_dict[self.active_matrix['name']]
        
        for key in self.storable_keys:
            if key in self.active_matrix:
                if self.active_matrix[key] is not None:
                    current_dict[key] = self.active_matrix[key]
        
        self.dm_dict[self.active_matrix['name']] = current_dict
    
    def check_requirements(self, name: Literal['rho0', 'rho1', 'rho2', 'rho_avg', 'all'], requirements):
        for key in requirements:
            if key not in self.storable_keys:
                raise ValueError(f"Invalid requirement key: {key}.")
            
            if name != 'all':
                if self.dm_dict[name] is None:
                    raise ValueError(f"Density matrix data for '{name}' not found.")
                rho_info = self.dm_dict[name]
                if not (key in rho_info) and (rho_info[key] is not None):
                    raise ValueError(f"Missing data in dm_dict for '{name}': {requirements}.")
        
            else:
                for rho_name, rho_info in self.dm_dict.items():
                    if not (key in rho_info) and (rho_info[key] is not None):
                        raise ValueError(f"Missing data in '{rho_name}': {requirements}.")
    
    def run_simulation(self, density_matrix: Literal['rho0', 'rho1', 'rho2', 'rho_avg'], sequence, title):
        self.active_matrix['name'] = density_matrix
        
        self.active_matrix['initial'] = getattr(self, density_matrix)
        rho = self.active_matrix['initial']
        
        niceprint(f'<u> Simulation for {title} <u/>', 4)
        if not isinstance(sequence, list):
            self.active_matrix['current'] = sequence * rho * sequence.dag()
            self.active_matrix['simdata'] = self.simulate(self.H_J, self.active_matrix['current'], 
                                                          self.t_array, self.T2, 
                                                          title=title, view='freqs')
        else:
            self.active_matrix['current'] = []
            self.active_matrix['simdata'] = []
            for seq in sequence:
                current = seq * rho * seq.dag()
                self.active_matrix['current'].append(current)
                current_sim = self.simulate(self.H_J, current, self.t_array, self.T2, plot=False)
                self.active_matrix['simdata'].append(current_sim)
            
            targ1_H = self.active_matrix['simdata'][0]['H']
            targ1_C = self.active_matrix['simdata'][0]['C']
            
            targ2_H = self.active_matrix['simdata'][1]['H']
            targ2_C = self.active_matrix['simdata'][1]['C']
            
            plt.figure(figsize=(13,6))
            plt.subplot(2, 2, 1)
            t1H_peak1, t1H_peak2 = targ1_H[3][1], targ1_H[3][0]
            plt.plot(targ1_H[1], targ1_H[2]);
            plt.title('Proton, 1', weight='bold')
            plt.xlabel(f'Peak Ratio: {t1H_peak1:.4f} / {t1H_peak2:.4f} = {t1H_peak1/t1H_peak2:.4f}')
            plt.gca().invert_xaxis()
            
            plt.subplot(2, 2, 2)
            t1C_peak1, t1C_peak2 = targ1_C[3][1], targ1_C[3][0]
            plt.plot(targ1_C[1], targ1_C[2]);
            plt.title('Carbon, 1', weight='bold')
            plt.xlabel(f'Peak Ratio: {t1C_peak1:.4f} / {t1C_peak2:.4f} = {t1C_peak1/t1C_peak2:.4f}')
            plt.gca().invert_xaxis()
            
            plt.subplot(2,2,3)
            plt.plot(targ2_H[1], targ2_H[2]);
            t2H_peak1, t2H_peak2 = targ2_H[3][1], targ2_H[3][0]
            plt.title('Proton, 2', weight='bold')
            plt.xlabel(f'Peak Ratio: {t2H_peak1:.4f} / {t2H_peak2:.4f} = {t2H_peak1/t2H_peak2:.4f}')
            plt.gca().invert_xaxis()
            
            plt.subplot(2,2,4)
            t2C_peak1, t2C_peak2 = targ2_C[3][1], targ2_C[3][0]
            plt.plot(targ2_C[1], targ2_C[2]);
            plt.title('Carbon, 2', weight='bold')
            plt.xlabel(f'Peak Ratio: {t2C_peak1:.4f} / {t2C_peak2:.4f} = {t2C_peak1/t2C_peak2:.4f}')
            plt.gca().invert_xaxis()
            
            plt.tight_layout()
            plt.show()
        
        self.update_dm_dict()
    
    def view_data(self, H_path, C_path,
                  plot=True, title=None, pad=5,
                  invert_xaxis=True, invert_yaxis=False, 
                  reverse_x=False, reverse_y=False):
        self.check_requirements(self.active_matrix['name'], ['simdata'])
        
        niceprint(f'<u> Experimental Data for {title} <u/>', 4)
        self.active_matrix['H_path'] = H_path if self.active_matrix['name'] is not 'rho_avg' else None
        self.active_matrix['C_path'] = C_path if self.active_matrix['name'] is not 'rho_avg' else None
        H_csv = pd.read_csv(H_path, header=None, names=['freq', 'real', 'imag'])
        C_csv = pd.read_csv(C_path, header=None, names=['freq', 'real', 'imag'])
        H_data, H_freq = H_csv['real'], H_csv['freq']
        C_data, C_freq = C_csv['real'], C_csv['freq']
        if reverse_x:
            H_data = H_data[::-1].reset_index(drop=True)
            C_data = C_data[::-1].reset_index(drop=True)
        if reverse_y:
            H_data = -H_data
            C_data = -C_data
        H_peak_list, Hfreq_min, Hfreq_max = locate_peaks(H_data, H_freq, pad)
        C_peak_list, Cfreq_min, Cfreq_max = locate_peaks(C_data, C_freq, pad)
        H_peak1, H_peak2 = H_peak_list[0][0], H_peak_list[1][0]
        C_peak1, C_peak2 = C_peak_list[0][0], C_peak_list[1][0]
        
        simdata = self.active_matrix['simdata']
        H_sim_peaks, C_sim_peaks = simdata[0], simdata[1]
        self.active_matrix['expH'] = (H_freq, H_data, H_peak_list, H_sim_peaks)
        self.active_matrix['expC'] = (C_freq, C_data, C_peak_list, C_sim_peaks)
        self.update_dm_dict()

        niceprint(r'**$^1$ H ratio:**  $\quad$ ' + f'{H_peak2} / {H_peak1} = {H_peak2/H_peak1:.3f} <br>' +
                r'**$^{13}$ C ratio:**  $\quad$ ' + f'{C_peak2} / {C_peak1} = {C_peak2/C_peak1:.3f}')
        
        if plot:
            plt.subplots(1, 2, figsize=(15,4))
            if title is not None:
                plt.suptitle(title, weight='bold')
            
            print(H_path), plt.subplot(1,2,1)
            plt.plot(H_freq, H_data)
            plt.title(fr'$^1$H Spectrum')
            plt.xlabel('Frequency (Hz)'), plt.ylabel('Signal (Real)')
            plt.xlim(Hfreq_min, Hfreq_max)
            if invert_xaxis:
                plt.gca().invert_xaxis()
            if invert_yaxis:
                plt.gca().invert_yaxis()
            
            print(C_path), plt.subplot(1,2,2)
            plt.plot(C_freq, C_data)
            plt.title(fr'$^{{13}}$C Spectrum')
            plt.xlabel('Frequency (Hz)'), plt.ylabel('Signal (Real)')
            plt.xlim(Cfreq_min, Cfreq_max)
            if invert_xaxis:
                plt.gca().invert_xaxis()
            if invert_yaxis:
                plt.gca().invert_yaxis()
            
            plt.tight_layout()
            plt.show()

    def thermal_average(self, title=None, pad=5):
        self.check_requirements('all', ['simdata', 'expH', 'expC'])
        name_list = [name.replace('rho', r'\rho_') for name in list(self.dm_dict.keys())[:3]]
        dicts_list = list(self.dm_dict.values())[:3]
        
        fig, axes = plt.subplots(1, 2, figsize=(15,4))
        colors = ['red', 'green', 'blue']
        
        H_center_freq = 3.82
        C_center_freq = 77.56
        H_fd, C_fd = [], []
        for i, rho_dict in enumerate(dicts_list):
            H_freqs_shifted, H_data_normalized = shift_and_scale(rho_dict['expH'], H_center_freq)
            C_freqs_shifted, C_data_normalized = shift_and_scale(rho_dict['expC'], C_center_freq)
            
            axes[0].plot(H_freqs_shifted, H_data_normalized, label=fr'${name_list[i]}$', alpha=0.5, color = colors[i])
            axes[1].plot(C_freqs_shifted, C_data_normalized, label=fr'${name_list[i]}$', alpha=0.5, color = colors[i])

            H_fd.append((H_freqs_shifted, H_data_normalized))
            C_fd.append((C_freqs_shifted, C_data_normalized))
        
        axes[0].set_title(fr'$^1$H Spectrum'), axes[0].set_xlabel('Frequency (Hz)'), axes[0].set_ylabel('Signal (Real)')
        axes[0].set_xlim(0, 8), axes[0].invert_xaxis(), axes[0].legend()
        axes[1].set_title(fr'$^{{13}}$C Spectrum'), axes[1].set_xlabel('Frequency (Hz)'), axes[1].set_ylabel('Signal (Real)')
        axes[1].set_xlim(65, 90), axes[1].invert_xaxis(), axes[1].legend()
        plt.show()
        
        
        H_freqs_common, H_sum_data = combine_data(H_fd)
        H_datas, H_freqs = pd.Series(H_sum_data / len(dicts_list)), pd.Series(H_freqs_common)
        H_peaks, Hfreq_min, Hfreq_max = locate_peaks(H_datas, H_freqs, pad)
        self.active_matrix['expH'] = (H_freqs, H_datas, H_peaks)
        
        C_freqs_common, C_sum_data = combine_data(C_fd)
        C_datas, C_freqs = pd.Series(C_sum_data / len(dicts_list)), pd.Series(C_freqs_common)
        C_peaks, Cfreq_min, Cfreq_max = locate_peaks(C_datas, C_freqs, pad)
        self.active_matrix['expC'] = (C_freqs, C_datas, C_peaks)
        
        self.update_dm_dict()
        
        plt.subplots(1, 2, figsize=(13,4))
        if title is not None:
            plt.suptitle(title, weight='bold')
        
        plt.subplot(1,2,1)
        plt.plot(H_freqs, H_datas)
        plt.title(fr'$^1$H Spectrum'), plt.xlabel('Frequency (Hz)'), plt.ylabel('Signal (Real)')
        plt.xlim(0, 8), plt.gca().invert_xaxis()
        
        plt.subplot(1,2,2)
        plt.plot(C_freqs, C_datas)
        plt.title(fr'$^{{13}}$C Spectrum'), plt.xlabel('Frequency (Hz)'), plt.ylabel('Signal (Real)')
        plt.xlim(65, 90), plt.gca().invert_xaxis()
        
        plt.tight_layout()
        plt.show()
        
        if len(H_peaks) > 1 and len(C_peaks) > 1:
            H_peak1, H_peak2 = H_peaks[0][0], H_peaks[1][0]
            C_peak1, C_peak2 = C_peaks[0][0], C_peaks[1][0]
            niceprint(r'**$^1$ H ratio:**  $\quad$ ' + f'{H_peak2} / {H_peak1} = {H_peak2/H_peak1:.3f} <br>' +
                    r'**$^{13}$ C ratio:**  $\quad$ ' + f'{C_peak2} / {C_peak1} = {C_peak2/C_peak1:.3f}')
        else:
            H_peak, C_peak = H_peaks[0][0], C_peaks[0][0]
            niceprint(r'**$^1$ H peak:**  $\quad$ ' + f'{H_peak:.3f} <br>' +
                    r'**$^{13}$ C peak:**  $\quad$ ' + f'{C_peak:.3f}')


class Moe:
    
    def __init__(self):
        self.pi = np.pi
        self.sq2 = np.sqrt(2)
        self.j = complex(0,1)
        self.sqj = np.sqrt(self.j)
        self.E = qeye(2)
        self.Ix = sigmax()/2
        self.Iy = sigmay()/2
        self.Iz = sigmaz()/2
        self.p1x=(-1j*np.pi/4*tensor(sigmax(),qeye(2))).expm()
        self.p1y=(-1j*np.pi/4*tensor(sigmay(),qeye(2))).expm()
        self.p2x=(-1j*np.pi/4*tensor(qeye(2),sigmax())).expm()
        self.p2y=(-1j*np.pi/4*tensor(qeye(2),sigmay())).expm()
        self.p1z=(-1j*np.pi/4*tensor(sigmaz(),qeye(2))).expm()
        self.p2z=(-1j*np.pi/4*tensor(qeye(2),sigmaz())).expm()
        self.uzz=(-1j*np.pi/4*tensor(sigmaz(),sigmaz())).expm()

    def format_state_vector(self, state): # for markdown table display
        vec = state.full().flatten()
        basis_labels = ["|00>", "|01>", "|10>", "|11>"]
        
        terms = []
        for i, amp in enumerate(vec):
            if abs(amp) > 1e-10:
                mag = abs(amp)
                phase = np.angle(amp)
                
                # Determine coefficient
                if abs(phase) < 0.1:  # Real positive
                    if abs(mag - 1) < 0.01:
                        coeff = ""
                    else:
                        coeff = f"{mag:.3f}"
                elif abs(phase - np.pi) < 0.1:  # Real negative
                    if abs(mag - 1) < 0.01:
                        coeff = "-"
                    else:
                        coeff = f"-{mag:.3f}"
                else:  # Complex
                    real = np.real(amp)
                    imag = np.imag(amp)
                    coeff = f"({real:.3f}{'+' if imag >= 0 else ''}{imag:.3f}i)"
                
                term = f"{coeff}{basis_labels[i]}" if coeff else basis_labels[i]
                terms.append(term)
        
        return " + ".join(terms).replace(" + -", " - ") if terms else "|00>"

    def detection_signal(self, t_array, rho, T2=None):
        """
        Compute the time-domain signals for both proton (H) and carbon (C)
        using the evolving sigma⁺ detection operator.

        Parameters:
            t_array : array of time points for the evolution.
            rho     : initial two-spin density matrix (Qobj with dims [[2,2],[2,2]]).
            T2      : relaxation time for exponential decay (if None, no decay is applied).

        Returns:
            signalH : array of complex detection signals for proton (H).
            signalC : array of complex detection signals for carbon (C).
        """
        signalH = np.zeros(len(t_array), dtype=complex)
        signalC = np.zeros(len(t_array), dtype=complex)

        for idx, t in enumerate(t_array):
            # Example: Uzz(t) = exp(-i * (215*pi/4 * sigmaz ⊗ sigmaz) * t)
            Uzz = (-1j * 215*np.pi/2 * tensor(sigmaz(), sigmaz()) * t).expm()

            # Evolve the sigma⁺ operator:
            # Proton: sigma⁺ on the first spin
            obsO_H = Uzz * tensor(sigmap(), self.E) * Uzz.dag()
            # Carbon: sigma⁺ on the second spin
            obsO_C = Uzz * tensor(self.E, sigmap()) * Uzz.dag()

            # Apply exponential decay if T2 is provided
            decay = np.exp(-t/T2) if T2 is not None else 1

            # Compute the detection signals using QuTiP's .tr() method
            signalH[idx] = (obsO_H * self.p1y * rho * self.p1y.dag()).tr() * decay
            signalC[idx] = (obsO_C * self.p2y * rho * self.p2y.dag()).tr() * decay

        return signalH, signalC

    def compute_spectrum(self, signal, t_array):
        """
        Compute the Fourier Transform of a time-domain signal.

        Parameters:
            signal  : time-domain signal (numpy array).
            t_array : time points corresponding to the signal.

        Returns:
            freq     : frequency axis (numpy array).
            spectrum : Fourier-transformed signal (numpy array).
        """
        dt = t_array[1] - t_array[0]
        spectrum = np.fft.fftshift(np.fft.fft(signal))
        freq = np.fft.fftshift(np.fft.fftfreq(len(t_array), d=dt))
        return freq, spectrum

    def simulate_and_plot_spectra(self, rho, t_array, T2=None, freq_range=5000):
        """
        Run the detection simulation for a given density matrix (rho) and
        plot the real part of the resulting proton and carbon spectra.

        Parameters:
            rho        : two-spin density matrix (Qobj)
            t_array    : time points for the simulation
            T2         : relaxation time for exponential decay (None -> no decay)
            freq_range : +/- range (in Hz) around the center frequency to display
        """
        # 1. Simulate the time-domain signals for proton (H) and carbon (C).
        signalH, signalC = self.detection_signal(t_array, rho, T2=T2)

        # 2. Compute the frequency-domain spectra via FFT.
        freqH, spectrumH = self.compute_spectrum(signalH, t_array)
        freqC, spectrumC = self.compute_spectrum(signalC, t_array)
        # 3. Shift frequencies to the proper resonance offsets.
        #    Proton at 62 MHz, Carbon at 15 MHz.
        freqH += 62e6
        freqC += 15e6

        # 4. Plot the real part of each spectrum.
        plt.figure(figsize=(12, 5))

        # Proton subplot
        plt.subplot(1, 2, 1)
        plt.plot(freqH, np.real(spectrumH), label="Proton")
        plt.xlim(62e6 - freq_range, 62e6 + freq_range)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Real(Spectrum)")
        plt.title("Proton Spectrum")

        # Carbon subplot
        plt.subplot(1, 2, 2)
        plt.plot(freqC, np.real(spectrumC), label="Carbon")
        plt.xlim(15e6 - freq_range, 15e6 + freq_range)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Real(Spectrum)")
        plt.title("Carbon Spectrum")

        plt.tight_layout()
        plt.show()