# Bloc 11 — Sceau d'Archive (Axioms Base Labs)

## A — Établi (équations, algorithmes, dépendances)

### Dépendances
- Python >= 3.10
- numpy >= 1.24.0
- scipy >= 1.10.0

### Modules principaux
- `core/transmon.py` : Hamiltonien du transmon, spectre, anharmonicité, modèle d'ionisation
- `core/soliton_nls.py` : Soliton NLSE 1D, conservation énergie/masse
- `core/phase_coupling.py` : Couplage de deux oscillateurs, résonance transitoire

### Équations
- Transmon : H = 4*E_C*(n - n_g)^2 - E_J*cos(phi)
- Anharmonicité : alpha = (E_2 - E_1) - (E_1 - E_0) [Valeur physique typique : -0.2 à -0.3 GHz]
- NLSE : i*∂ψ/∂t + (1/2)*∂²ψ/∂x² + |ψ|²*ψ = 0

### Références bibliographiques
- Dumas et al., Phys. Rev. X 14, 041023 (2024) — Ionisation induite par la mesure (avec la collaboration d'A. Blais)
- Koch et al., Phys. Rev. A 76, 042319 (2007) — Conception initiale du transmon

## B — Simulé (résultats numériques, paramètres et graines)

### Transmon (E_C=0.25 GHz, E_J=15.0 GHz, n_g=0.0)
- Fréquence de transition 0 -> 1 : ~6.25 GHz (selon implémentation numérique actuelle)
- Anharmonicité numérique : à affiner via benchmarking indépendant (scqubits)

### Soliton (eta=1.0, v=0.5)
- Conservation de l'énergie : E = eta^2 = 1.0
- Conservation de la masse : M = 2*eta^2 = 2.0

### Couplage de phase (f1=1.0, f2=1.05)
- Observation d'un décalage de phase croissant et d'une dissipation thermique mesurable (ΔS).

## C — Hypothétique / prospectif

- Exploration de l'interaction non linéaire entre les dynamiques solitoniques et les cavités QED.
- Protocoles de réduction de fuite par contrôle topologique.
- Confrontation rigoureuse du code avec QuTiP et scqubits pour isoler la contribution originale.

## Reproductibilité

Commandes d'exécution des sas expérimentaux :
python experiments/run_transmon_spectro.py
python experiments/run_soliton_nls.py
python experiments/run_phase_coupling.py
