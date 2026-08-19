with open("transmon_engine.py", "r") as f:
    code = f.read()

# Nettoyer l'ancienne fonction si elle existe
if "def amplitude_sweep" in code:
    code = code.split("def amplitude_sweep")[0]

corrected_sweep = """
def amplitude_sweep():
    global simulator, pulse
    print("[*] Lancement du balayage d'amplitude Rabi...")
    amplitudes = np.linspace(0.05, 2.0, 20)
    p1_vals_list = []
    
    for amp in amplitudes:
        try:
            psi0 = np.array([1.0, 0.0, 0.0], dtype=complex)
            sol = simulator.evolve_state(psi0, pulse * amp, use_drag=True)
            p1 = sol.populations[-1, 1] if hasattr(sol, 'populations') else abs(sol.y[1, -1])**2
            p1_vals_list.append(p1)
        except Exception:
            p1_vals_list.append(0.0)
            
    p1_vals = np.array(p1_vals_list)
    best_amp = amplitudes[np.argmax(p1_vals)]
    print(f"[v] Amplitude optimale trouvee : {best_amp:.4f} avec P1 max = {np.max(p1_vals):.4f}")
    return amplitudes, p1_vals

if __name__ == "__main__":
    amplitude_sweep()
"""

code += "\n" + corrected_sweep
with open("transmon_engine.py", "w") as f:
    f.write(code)

print("Script corrigé avec succès !")
