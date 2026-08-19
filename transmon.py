import numpy as np

class TransmonQubit:
    def __init__(self, ej: float = 20.0, ec: float = 0.3, n_levels: int = 5) -> None:
        self.ej = ej
        self.ec = ec
        self.n_levels = n_levels
        self.hamiltonian = self._build_hamiltonian()

    def _build_hamiltonian(self) -> np.ndarray:
        H = np.zeros((self.n_levels, self.n_levels))
        for n in range(self.n_levels):
            H[n, n] = 4.0 * self.ec * (n**2)
        for n in range(self.n_levels - 1):
            H[n, n + 1] = -0.5 * self.ej
            H[n + 1, n] = -0.5 * self.ej
        return H

    def get_eigenenergies(self) -> np.ndarray:
        eigenvals, _ = np.eigh(self.hamiltonian)
        return eigenvals

    def get_anharmonicity(self) -> float:
        evals = self.get_eigenenergies()
        if len(evals) >= 3:
            e01 = evals[1] - evals[0]
            e12 = evals[2] - evals[1]
            return float(e12 - e01)
        return 0.0
