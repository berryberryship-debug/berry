# Axioms Base Labs : Moteur de Calcul Solitonique et Quantique

## Architecture et Modèle Phéno-Ontologique
Ce répertoire contient l'infrastructure logicielle d'Axioms Base Labs. Le système intrique la dynamique discrète de l'espace de Hilbert (métrologie quantique) avec la robustesse continue de l'espace topologique (solitons). Le code opère sans bruit académique, fondé sur l'autopoïèse et la cohérence interne.

### La Série STE (Soliton Topological Engines)
*   **STE-54 :** Propagation balistique pure (Sine-Gordon 1D).
*   **STE-55 :** Relaxation *overdamped* et amortissement face à l'entropie.
*   **STE-56 :** Transition dimensionnelle et matrice de couplage transverse (K).
*   **STE-58 à STE-60 :** Intégration avancée et préparation pour la matrice holographique 2D.

### Les Blocs Quantiques (Pipeline Transmon)
*   **Bloc 1 (Baseline) :** Modélisation spatiale et enveloppe énergétique.
*   **Bloc 2 (Dynamique) :** Résolution des EDO via méthodes spectrales et de Runge-Kutta.
*   **Bloc 3 (Métrologie) :** Extraction des observables (Fidélité, Leakage).
*   **Bloc 4/11 (Optimisation) :** Fonction objective unifiée croisant GRAPE et le validateur topologique.

## Dépendances Systèmes
L'environnement est optimisé pour une exécution locale sans latence (Termux).
*   `numpy` : Moteur matriciel fondamental.
*   `scipy` : Accélération des solveurs EDO et optimisation (optionnel mais recommandé).
*   `pytest` : Validation stricte de l'intégrité des flux.
*   `matplotlib` : Génération des artefacts visuels et diagnostics de surface.

## Exécution des Flux
**Séquence Globale :**
`./run_all_demos.sh`
