with open("transmon_engine.py", "r") as f:
    code = f.read()

sweep_code = """

def amplitude_sweep():
    print("[*] Lancement du balayage d'amplitude Rabi...")
    amplitudes = np.linspace(0.05, 2.0, 20)
    p1_vals = []
    
    for amp in amplitudes:
        try:
            # Test de l'évolution avec l'amplitude mise à l'échelle
            psi0 = np.array([1.0, 0.0, 0.0], dtype=complex)
            # Utilisation du simulateur et du pulse courant
            sol = simulator.evolve_state(psi0, pulse * amp, use_drag=True)
            p1 = sol.populations[-1, 1] if hasattr(sol, 'populations') else abs(sol.y[1, -1])**2
            p1_vals.append(p1)
        except Exception:
            p1_vals.append(0.0)
            
        p1_vals = np.array(p1_vals)
    best_amp = amplitudes[np.argmax(p1_vals)]
    print(f"[v] Amplitude optimale trouvee : {best_amp:.4f} avec P1 max = {np.max(p1_vals):.4f}")
    return amplitudes, p1_vals

if __name__ == "__main__":
    amplitude_sweep()
"""

if "def amplitude_sweep" not in code:
    code += "\n" + sweep_code
    with open("transmon_engine.py", "w") as f:
        f.write(code)
    print("Routine d'amplitude ajoutée avec succès.")
else:
    print("La routine existe déjà.")
