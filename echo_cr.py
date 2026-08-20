import numpy as np

# ============================================================
# 1. Opérateurs de Pauli à deux qubits (NumPy pur)
# ============================================================
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

IX = np.kron(I, X)
IY = np.kron(I, Y)
IZ = np.kron(I, Z)
XI = np.kron(X, I)
ZI = np.kron(Z, I)
ZX = np.kron(Z, X)
ZY = np.kron(Z, Y)
ZZ = np.kron(Z, Z)

# ============================================================
# 2. Exponentielle matricielle via diagonalisation (100% pur NumPy)
# ============================================================
def expm_numpy(A):
    vals, vecs = np.linalg.eigh(A)
    return vecs @ np.diag(np.exp(vals)) @ vecs.conj().T

# ============================================================
# 3. Paramètres physiques optimisés
# ============================================================
Omega_ZX = 2 * np.pi * 3.0      # force utile ZX augmentée (MHz)
Omega_ZZ = 2 * np.pi * 0.08     # ZZ parasite
Omega_ZI = 2 * np.pi * 0.12     # ZI parasite
Omega_IX = 2 * np.pi * 0.25     # IX parasite (crosstalk)
Omega_IY = 2 * np.pi * 0.03     # IY parasite
Omega_IZ = 2 * np.pi * 0.05     # IZ parasite

T = 0.180                       # durée totale de la porte (µs)

# ============================================================
# 4. Construction de l'Hamiltonien CR
# ============================================================
def H_CR(sign=1.0, cancellation=0.0):
    H = (
        sign * (Omega_ZX / 2) * ZX
        + (Omega_ZZ / 2) * ZZ
        + (Omega_ZI / 2) * ZI
        + (Omega_IX / 2 + cancellation) * IX
        + (Omega_IY / 2) * IY
        + (Omega_IZ / 2) * IZ
    )
    return H

# ============================================================
# 5. Propagation unitaire
# ============================================================
def evolve(H, t):
    # -1j * H * t est hermitique si H l'est, mais attention aux phases complexes.
    # Pour eigh, on passe par un traitement Hermitian standard :
    # exp(-i H t) -> on diagonalise H directement.
    vals, vecs = np.linalg.eigh(H)
    return vecs @ np.diag(np.exp(-1j * vals * t)) @ vecs.conj().T

# ============================================================
# 6. Séquence Echo-CR complète
# ============================================================
def echo_cr_unitary(use_cancellation=True):
    t_half = T / 2
    canc = -Omega_IX / 2 if use_cancellation else 0.0

    U1 = evolve(H_CR(sign=+1, cancellation=canc), t_half)
    X_pi = XI  # Impulsion X sur le contrôle (X ⊗ I)
    U2 = evolve(H_CR(sign=-1, cancellation=canc), t_half)

    U = U2 @ X_pi @ U1
    return U

# ============================================================
# 7. Analyse de la porte
# ============================================================
def analyze_gate(U, title="Echo-CR"):
    pauli_basis = {
        "IX": IX, "IY": IY, "IZ": IZ,
        "ZI": ZI, "ZX": ZX, "ZY": ZY, "ZZ": ZZ,
        "XI": XI
    }

    print(f"\n=== Analyse de la porte : {title} ===")
    print(f"{'Terme':<6} {'Amplitude':>12}")
    print("-" * 20)

    for name, op in pauli_basis.items():
        coeff = np.trace(U.conj().T @ op) / 4
        if abs(coeff) > 1e-3:
            print(f"{name:<6} {np.abs(coeff):>12.4f}")

# ============================================================
# 8. Exécution et Fidélité corrigée
# ============================================================
if __name__ == "__main__":
    print("Simulation Echo-CR (NumPy pur - sans dépendance)")

    U_raw = echo_cr_unitary(use_cancellation=False)
    analyze_gate(U_raw, title="Sans cancellation")

    U_clean = echo_cr_unitary(use_cancellation=True)
    analyze_gate(U_clean, title="Avec cancellation")

    # Fidélité avec intégration de l'impulsion X_pi
    X_pi = XI
    theta = Omega_ZX * T
    # Idéal calculé via la même fonction d'évolution propre
    U_ideal = X_pi @ evolve(ZX * (Omega_ZX / 2), theta / (Omega_ZX * T) * theta ) # simplifié
    # Cible idéale pure ZX sur durée T :
    vals_id, vecs_id = np.linalg.eigh(ZX)
    U_ideal = X_pi @ (vecs_id @ np.diag(np.exp(-1j * vals_id * (Omega_ZX * T / 2))) @ vecs_id.conj().T) # ajusté

    fid = np.abs(np.trace(U_ideal.conj().T @ U_clean)) / 4
    print(f"\nFidélité avec ZX idéale : {fid.real:.6f}")
