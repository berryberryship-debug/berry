#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=======================================================================
TRANSMON RESEARCH ENGINE
Version 2.0 — Implantation Termux
=======================================================================
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from numpy.linalg import eigh

import numpy as np
def mini_solve_ivp(fun, t_span, y0, t_eval, method=None, rtol=None, atol=None):
    dt = t_eval[1] - t_eval[0]
    ts = t_eval
    ys = [np.array(y0, dtype=complex)]
    y = np.array(y0, dtype=complex)
    for i in range(1, len(ts)):
        t = ts[i-1]
        # Euler / RK2 light step for quantum state evolution
        dy = fun(t, y)
        y = y + dy * dt
        # Normalisation pour un état quantique
        y = y / np.linalg.norm(y)
        ys.append(y.copy())
    class Solution:
        pass
    sol = Solution()
    sol.t = ts
    sol.y = np.array(ys).T
    return sol

solve_ivp = mini_solve_ivp


TWO_PI = 2.0 * np.pi
PHI0 = 1.0

def dagger(A):
    return np.conjugate(A.T)

def commutator(A, B):
    return A @ B - B @ A

def gaussian(t, sigma, center):
    return np.exp(-0.5 * ((t - center) / sigma) ** 2)

@dataclass
class TransmonParameters:
    Ec_GHz: float = 0.25
    Ej_GHz: float = 50.0
    ng: float = 0.0
    flux: float = 0.0
    Ncharge: int = 15
    levels: int = 5
    asymmetry: float = 0.0

class Transmon:
    def __init__(self, params: TransmonParameters):
        self.p = params
        self.n_values = np.arange(-self.p.Ncharge, self.p.Ncharge + 1, dtype=float)
        self.dim_charge = len(self.n_values)

        if self.p.levels > self.dim_charge:
            raise ValueError("levels doit être <= dimension de la base de charge.")

        self.n_operator = np.diag(self.n_values)
        self.cos_phi = self._build_cos_phi()
        self.H_charge_GHz = self.build_hamiltonian_GHz()
        self.energies_GHz, self.eigenvectors = eigh(self.H_charge_GHz)

        self.U = self.eigenvectors[:, :self.p.levels]
        self.H_eigen_GHz = np.diag(self.energies_GHz[:self.p.levels])
        self.n_eigen = dagger(self.U) @ self.n_operator @ self.U

    def _build_shift_operators(self):
        d = self.dim_charge
        plus = np.zeros((d, d), dtype=complex)
        minus = np.zeros((d, d), dtype=complex)
        for i in range(d - 1):
            plus[i + 1, i] = 1.0
            minus[i, i + 1] = 1.0
        return plus, minus

    def _build_cos_phi(self):
        plus, minus = self._build_shift_operators()
        return 0.5 * (plus + minus)

    def Ej_flux_GHz(self, flux):
        f = np.asarray(flux)
        theta = np.pi * f / PHI0
        d = self.p.asymmetry
        if d == 0.0:
            return self.p.Ej_GHz * np.abs(np.cos(theta))
        return self.p.Ej_GHz * np.sqrt(np.cos(theta)**2 + d**2 * np.sin(theta)**2)

    def build_hamiltonian_GHz(self, ng=None, flux=None, Ej_GHz=None):
        if ng is None:
            ng = self.p.ng
        if flux is None:
            flux = self.p.flux
        if Ej_GHz is None:
            Ej_GHz = self.Ej_flux_GHz(flux)

        charging = 4.0 * self.p.Ec_GHz * np.diag((self.n_values - ng)**2)
        josephson = -Ej_GHz * self.cos_phi
        return charging + josephson

    def spectrum(self):
        E = self.energies_GHz[:self.p.levels]
        result = {"energies_GHz": E}
        if len(E) >= 2:
            result["f01_GHz"] = E[1] - E[0]
        if len(E) >= 3:
            result["f12_GHz"] = E[2] - E[1]
            result["anharmonicity_GHz"] = result["f12_GHz"] - result["f01_GHz"]
        return result

@dataclass
class PulseParameters:
    duration_ns: float = 40.0
    sigma_ns: float = 8.0
    amplitude_rad_ns: float = 0.02
    beta: float = 0.0
    center_ns: float | None = None

class Pulse:
    def __init__(self, params: PulseParameters):
        self.p = params
        self.center_ns = self.p.center_ns if self.p.center_ns is not None else self.p.duration_ns / 2.0

    def I_gaussian(self, t):
        return self.p.amplitude_rad_ns * gaussian(t, self.p.sigma_ns, self.center_ns)

    def dI_dt(self, t):
        I = self.I_gaussian(t)
        return -((t - self.center_ns) / self.p.sigma_ns**2) * I

    def envelope(self, t, drag=True):
        I = self.I_gaussian(t)
        if not drag:
            return I + 0.0j
        Q = self.p.beta * self.dI_dt(t)
        return I + 1j * Q

class DissipationModel:
    def __init__(self, transmon: Transmon, T1_ns=None, Tphi_ns=None):
        self.transmon = transmon
        self.dim = transmon.p.levels
        self.T1_ns = T1_ns
        self.Tphi_ns = Tphi_ns

    def collapse_operators(self):
        operators = []
        if self.T1_ns is not None:
            gamma = 1.0 / self.T1_ns
            nmat = self.transmon.n_eigen
            for j in range(1, self.dim):
                L = np.zeros((self.dim, self.dim), dtype=complex)
                L[j - 1, j] = np.sqrt(gamma) * nmat[j - 1, j]
                operators.append(L)
        if self.Tphi_ns is not None:
            gamma_phi = 1.0 / self.Tphi_ns
            D = np.zeros((self.dim, self.dim), dtype=complex)
            for k in range(self.dim):
                D[k, k] = float(k)
            D -= (np.trace(D) / self.dim) * np.eye(self.dim)
            operators.append(np.sqrt(gamma_phi) * D)
        return operators

class QuantumSimulator:
    def __init__(self, transmon: Transmon, t_ns, drive_frequency_GHz=None):
        self.transmon = transmon
        self.t_ns = np.asarray(t_ns, dtype=float)
        self.dim = transmon.p.levels
        spectrum = transmon.spectrum()
        self.drive_frequency_GHz = drive_frequency_GHz if drive_frequency_GHz is not None else spectrum["f01_GHz"]
        self.drive_omega = TWO_PI * self.drive_frequency_GHz

    def controlled_hamiltonian(self, t_ns, pulse, use_drag=True, phase=0.0):
        H0 = TWO_PI * self.transmon.H_eigen_GHz
        envelope = pulse.envelope(t_ns, drag=use_drag)
        I, Q = np.real(envelope), np.imag(envelope)
        theta = self.drive_omega * t_ns + phase
        drive = I * np.cos(theta) + Q * np.sin(theta)
        return H0 + drive * self.transmon.n_eigen

    def evolve_state(self, psi0, pulse, use_drag=True):
        psi0 = np.asarray(psi0, dtype=complex)
        psi0 /= np.linalg.norm(psi0)

        def rhs(t, psi):
            H = self.controlled_hamiltonian(t, pulse, use_drag=use_drag)
            return -1j * H @ psi

        solution = solve_ivp(rhs, (self.t_ns[0], self.t_ns[-1]), psi0, t_eval=self.t_ns, method="DOP853", rtol=1e-8, atol=1e-10)
        return solution.y.T

    def evolve_density_matrix(self, rho0, pulse, T1_ns=None, Tphi_ns=None, use_drag=True):
        rho0 = np.asarray(rho0, dtype=complex)
        collapse = DissipationModel(self.transmon, T1_ns, Tphi_ns).collapse_operators()

        def unpack(y): return y.reshape(self.dim, self.dim)
        def pack(rho): return rho.reshape(self.dim * self.dim)

        def rhs(t, y):
            rho = unpack(y)
            H = self.controlled_hamiltonian(t, pulse, use_drag=use_drag)
            drho = -1j * commutator(H, rho)
            for L in collapse:
                drho += L @ rho @ dagger(L) - 0.5 * (dagger(L) @ L @ rho + rho @ dagger(L) @ L)
            return pack(drho)

        solution = solve_ivp(rhs, (self.t_ns[0], self.t_ns[-1]), pack(rho0), t_eval=self.t_ns, method="DOP853", rtol=1e-7, atol=1e-9)
        return np.array([unpack(y) for y in solution.y.T])

    @staticmethod
    def populations_state(psi): return np.abs(psi) ** 2

    @staticmethod
    def populations_density(rho): return np.real(np.diagonal(rho, axis1=1, axis2=2))

    @staticmethod
    def leakage(populations, logical_levels=2):
        return np.maximum(0.0, 1.0 - np.sum(populations[:, :logical_levels], axis=1))

def main():
    print("[*] Initialisation des paramètres...")
    params = TransmonParameters(Ec_GHz=0.25, Ej_GHz=50.0, Ncharge=15, levels=5)
    transmon = Transmon(params)
    s = transmon.spectrum()
    print(f"[*] f01 = {s['f01_GHz']:.6f} GHz | Anharmonicité = {s['anharmonicity_GHz']:.6f} GHz")

    duration = 40.0
    t = np.linspace(0.0, duration, 500)
    pulse = Pulse(PulseParameters(duration_ns=duration, amplitude_rad_ns=0.02, beta=0.15))
    simulator = QuantumSimulator(transmon, t)

    psi0 = np.zeros(params.levels, dtype=complex)
    psi0[0] = 1.0

    print("[*] Exécution de la simulation Schrödinger (DRAG)...")
    psi = simulator.evolve_state(psi0, pulse, use_drag=True)
    pop = simulator.populations_state(psi)
    leak = simulator.leakage(pop)

    print(f"[✓] Simulation terminée. Population finale P1 = {pop[-1, 1]:.6f} | Leakage final = {leak[-1]:.3e}")

if __name__ == "__main__":
    main()





