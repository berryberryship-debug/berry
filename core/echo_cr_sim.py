import numpy as np
from qutip import *
import matplotlib.pyplot as plt

# ============================================================
# 1. Opérateurs de Pauli à deux qubits
# ============================================================
I = qeye(2)
X = sigmax()
Y = sigmay()
Z = sigmaz()

IX = tensor(I, X)
IY = tensor(I, Y)
IZ = tensor(I, Z)
ZI = tensor(Z, I)
ZX = tensor(Z, X)
ZY = tensor(Z, Y)
ZZ = tensor(Z, Z)
XI = tensor(X, I)

# ============================================================
# 2. Paramètres physiques (en unités de 2π·MHz)
# ============================================================
Omega_ZX = 2 * np.pi * 1.5      # force utile ZX  (MHz)
Omega_ZZ = 2 * np.pi * 0.15     # ZZ parasite
Omega_ZI = 2 * np.pi * 0.25     # ZI parasite
Omega_IX = 2 * np.pi * 0.40     # IX parasite (crosstalk)
Omega_IY = 2 * np.pi * 0.05     # IY parasite
Omega_IZ = 2 * np.pi * 0.10     # IZ parasite

T = 300e-3                      # durée totale de la porte (µs)
dt = 0.5e-3                     # pas de temps

# ============================================================
# 3. Construction de l'Hamiltonien CR
# ============================================================
def H_CR(sign=1.0, cancellation=0.0):
    """
    Hamiltonien Cross-Resonance.
    sign = +1 ou -1
    cancellation : amplitude du tone de compensation sur la cible
    """
    H = (
        sign * Omega_ZX / 2 * ZX
        + Omega_ZZ / 2 * ZZ
        + Omega_ZI / 2 * ZI
        + (Omega_IX / 2 + cancellation) * IX
        + Omega_IY / 2 * IY
        + Omega_IZ / 2 * IZ
    )
    return H

# ============================================================
# 4. Propagation unitaire
# ============================================================
def evolve(H, t):
    """Retourne l'opérateur d'évolution exp(-i H t)"""
    return (-1j * H * t).expm()

# ============================================================
# 5. Séquence Echo-CR complète
# ============================================================
def echo_cr_unitary(use_cancellation=True):
    t_half = T / 2
    canc = -Omega_IX / 2 if use_cancellation else 0.0

    # Première moitié
    U1 = evolve(H_CR(sign=+1, cancellation=canc), t_half)

    # Impulsion Xπ sur le contrôle
    X_pi = tensor(X, I)

    # Deuxième moitié (signe inversé)
    U2 = evolve(H_CR(sign=-1, cancellation=canc), t_half)

    # Séquence complète
    U = U2 * X_pi * U1
    return U

# ============================================================
# 6. Analyse de la porte
# ============================================================
def analyze_gate(U, title="Echo-CR"):
    """Affiche les coefficients de Pauli de la porte unitaire"""
    pauli_basis = {
        "IX": IX, "IY": IY, "IZ": IZ,
        "ZI": ZI, "ZX": ZX, "ZY": ZY, "ZZ": ZZ,
        "XI": XI
    }

    print(f"\n=== Analyse de la porte : {title} ===")
    print(f"{'Terme':<6} {'Coefficient':>12} {'Phase (rad)':>12}")
    print("-" * 34)

    for name, op in pauli_basis.items():
        # Projection : (1/4) Tr(U† · op)
        coeff = (U.dag() * op).tr() / 4
        if abs(coeff) > 1e-3:
            phase = np.angle(coeff)
            print(f"{name:<6} {np.abs(coeff):>12.4f} {phase:>12.4f}")

# ============================================================
# 7. Exécution
# ============================================================
if __name__ == "__main__":
    # Porte sans cancellation
    U_raw = echo_cr_unitary(use_cancellation=False)
    analyze_gate(U_raw, title="Echo-CR sans cancellation")

    # Porte avec cancellation
    U_clean = echo_cr_unitary(use_cancellation=True)
    analyze_gate(U_clean, title="Echo-CR avec cancellation")

    # Fidélité par rapport à une porte ZX idéale
    theta = Omega_ZX * T
    U_ideal = (-1j * theta / 2 * ZX).expm()

    fid = np.abs((U_ideal.dag() * U_clean).tr()) / 4
    print(f"\nFidélité avec la porte ZX idéale : {fid.real:.6f}")
