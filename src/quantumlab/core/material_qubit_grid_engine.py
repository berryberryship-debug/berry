#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MATERIAL-QUBIT GRID ENGINE — MQGE 1.1
===============================================================================

MATERIAL GRID
    ├── géométrie
    ├── matériau
    └── paramètres effectifs
            ↓
        TRANSMON
        (EJ / EC / ng)
            ↓
     Hamiltonien local
            ↓
       QUBIT NETWORK
       (J_ij)
            ↓
        spectre / modes
            ↓
     Monte-Carlo / HPC

Important :
- EJ, C, couplages et pertes sont des paramètres de modèle effectifs.
- Ils ne sont pas des propriétés expérimentales universelles des matériaux.
- Une application expérimentale exige une calibration par géométrie,
  simulation EM et mesures.

Unités :
- Énergie / fréquence : GHz
- Capacité : fF

Dépendances :
- numpy
- scipy
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


H_PLANCK = 6.626_070_15e-34
E_CHARGE = 1.602_176_634e-19
GHZ = 1.0e9


class Material(str, Enum):
    ALUMINUM = "Al"
    TANTALUM = "Ta"
    NIOBIUM = "Nb"
    TITANIUM_NITRIDE = "TiN"
    NIOBIUM_NITRIDE = "NbN"
    SAPPHIRE = "sapphire"
    SILICON = "high_resistivity_Si"
    VACUUM = "vacuum"


@dataclass(frozen=True, slots=True)
class MaterialModel:
    name: Material
    josephson_factor: float = 1.0
    coupling_factor: float = 1.0
    loss_factor: float = 1.0


MATERIALS: dict[Material, MaterialModel] = {
    material: MaterialModel(material)
    for material in Material
}


@dataclass(slots=True)
class MaterialCell:
    x: int
    y: int
    material: Material
    capacitance_fF: float = 70.0
    josephson_energy_GHz: float = 15.0
    ng: float = 0.0
    active: bool = True
    metadata: dict[str, float | str] = field(default_factory=dict)

    def effective_EJ_GHz(self) -> float:
        value = self.josephson_energy_GHz * MATERIALS[self.material].josephson_factor
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"EJ invalide pour la cellule ({self.x}, {self.y}).")
        return float(value)


@dataclass(frozen=True, slots=True)
class TransmonParameters:
    EJ_GHz: float
    EC_GHz: float
    ng: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.EJ_GHz) or self.EJ_GHz <= 0.0:
            raise ValueError("EJ_GHz doit être positif et fini.")
        if not np.isfinite(self.EC_GHz) or self.EC_GHz <= 0.0:
            raise ValueError("EC_GHz doit être positif et fini.")
        if not np.isfinite(self.ng):
            raise ValueError("ng doit être fini.")

    @property
    def ratio_EJ_EC(self) -> float:
        return self.EJ_GHz / self.EC_GHz

    @property
    def plasma_frequency_GHz(self) -> float:
        return math.sqrt(8.0 * self.EJ_GHz * self.EC_GHz)

    @property
    def approximate_f01_GHz(self) -> float:
        return self.plasma_frequency_GHz - self.EC_GHz

    @property
    def anharmonicity_GHz(self) -> float:
        return -self.EC_GHz


class MaterialToTransmon:
    @staticmethod
    def capacitance_to_EC_GHz(capacitance_fF: float) -> float:
        if not np.isfinite(capacitance_fF) or capacitance_fF <= 0.0:
            raise ValueError("capacitance_fF doit être positive et finie.")
        capacitance_F = capacitance_fF * 1e-15
        EC_joule = E_CHARGE**2 / (2.0 * capacitance_F)
        return float(EC_joule / H_PLANCK / GHZ)

    @classmethod
    def cell_to_parameters(cls, cell: MaterialCell) -> TransmonParameters:
        return TransmonParameters(
            EJ_GHz=cell.effective_EJ_GHz(),
            EC_GHz=cls.capacitance_to_EC_GHz(cell.capacitance_fF),
            ng=cell.ng,
        )


class TransmonHamiltonian:
    """
    Hamiltonien du transmon en base de charge :
        H = 4 EC (n - ng)^2 - EJ/2 (|n><n+1| + h.c.)
    """

    @staticmethod
    def matrix(params: TransmonParameters, ncut: int = 15) -> sparse.csr_matrix:
        if ncut < 1:
            raise ValueError("ncut doit être >= 1.")

        n = np.arange(-ncut, ncut + 1, dtype=float)
        diagonal = 4.0 * params.EC_GHz * (n - params.ng) ** 2
        offdiagonal = np.full(len(n) - 1, -0.5 * params.EJ_GHz)

        return sparse.diags(
            diagonals=[diagonal, offdiagonal, offdiagonal],
            offsets=[0, 1, -1],
            format="csr",
        )

    @staticmethod
    def spectrum(
        params: TransmonParameters,
        ncut: int = 15,
        levels: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        if levels < 1:
            raise ValueError("levels doit être >= 1.")

        H = TransmonHamiltonian.matrix(params, ncut)
        dimension = H.shape[0]

        if dimension <= 32 or levels >= dimension - 1:
            values, vectors = np.linalg.eigh(H.toarray())
            return values[:levels], vectors[:, :levels]

        values, vectors = spla.eigsh(H, k=levels, which="SA")
        order = np.argsort(values)
        return values[order], vectors[:, order]


@dataclass
class MaterialGrid:
    width: int
    height: int
    cells: dict[tuple[int, int], MaterialCell] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("La grille doit mesurer au minimum 1x1.")

    def _check_coordinates(self, x: int, y: int) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"Coordonnées hors grille : ({x}, {y}).")

    def set_cell(self, x: int, y: int, material: Material, **kwargs) -> None:
        self._check_coordinates(x, y)
        self.cells[(x, y)] = MaterialCell(x, y, material, **kwargs)

    def get_cell(self, x: int, y: int) -> MaterialCell:
        self._check_coordinates(x, y)
        return self.cells.get(
            (x, y),
            MaterialCell(x, y, Material.VACUUM, active=False),
        )

    def active_cells(self) -> list[MaterialCell]:
        return [cell for cell in self.cells.values() if cell.active]


@dataclass(frozen=True, slots=True)
class Coupling:
    a: tuple[int, int]
    b: tuple[int, int]
    J_GHz: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.J_GHz):
            raise ValueError("J_GHz doit être fini.")


class CouplingGraph:
    def __init__(self) -> None:
        self.edges: list[Coupling] = []

    def connect(self, a: tuple[int, int], b: tuple[int, int], J_GHz: float) -> None:
        self.edges.append(Coupling(a, b, float(J_GHz)))

    def nearest_neighbors(self, grid: MaterialGrid, J_GHz: float) -> None:
        for y in range(grid.height):
            for x in range(grid.width):
                if not grid.get_cell(x, y).active:
                    continue
                if x + 1 < grid.width and grid.get_cell(x + 1, y).active:
                    self.connect((x, y), (x + 1, y), J_GHz)
                if y + 1 < grid.height and grid.get_cell(x, y + 1).active:
                    self.connect((x, y), (x, y + 1), J_GHz)


@dataclass(frozen=True, slots=True)
class EffectiveQubit:
    index: int
    position: tuple[int, int]
    params: TransmonParameters
    material: Material

    @property
    def frequency_GHz(self) -> float:
        return self.params.approximate_f01_GHz

    @property
    def anharmonicity_GHz(self) -> float:
        return self.params.anharmonicity_GHz


class QubitNetwork:
    def __init__(self, grid: MaterialGrid, ncut: int = 15) -> None:
        self.grid = grid
        self.ncut = ncut
        self.qubits: list[EffectiveQubit] = []
        self.position_to_index: dict[tuple[int, int], int] = {}
        self.couplings: list[Coupling] = []
        self._build()

    def _build(self) -> None:
        for index, cell in enumerate(self.grid.active_cells()):
            params = MaterialToTransmon.cell_to_parameters(cell)
            self.qubits.append(
                EffectiveQubit(
                    index=index,
                    position=(cell.x, cell.y),
                    params=params,
                    material=cell.material,
                )
            )
            self.position_to_index[(cell.x, cell.y)] = index

    @property
    def size(self) -> int:
        return len(self.qubits)

    def add_couplings(self, graph: CouplingGraph) -> None:
        self.couplings = list(graph.edges)

    def frequency_vector(self) -> np.ndarray:
        return np.array([q.frequency_GHz for q in self.qubits], dtype=float)


class EffectiveHamiltonian:
    @staticmethod
    def single_excitation(
        network: QubitNetwork,
        frequencies_GHz: np.ndarray | None = None,
    ) -> sparse.csr_matrix:
        size = network.size
        if size == 0:
            return sparse.csr_matrix((0, 0), dtype=np.complex128)

        frequencies = (
            network.frequency_vector()
            if frequencies_GHz is None
            else np.asarray(frequencies_GHz, dtype=float)
        )

        if frequencies.shape != (size,):
            raise ValueError("Le vecteur de fréquences a une dimension invalide.")

        rows = list(range(size))
        cols = list(range(size))
        data: list[complex] = [complex(value) for value in frequencies]

        for edge in network.couplings:
            if edge.a not in network.position_to_index or edge.b not in network.position_to_index:
                continue
            ia = network.position_to_index[edge.a]
            ib = network.position_to_index[edge.b]
            coupling = complex(edge.J_GHz)
            rows.extend([ia, ib])
            cols.extend([ib, ia])
            data.extend([coupling, np.conjugate(coupling)])

        return sparse.coo_matrix(
            (np.asarray(data, dtype=np.complex128), (rows, cols)),
            shape=(size, size),
        ).tocsr()


@dataclass(slots=True)
class Spectrum:
    energies_GHz: np.ndarray
    states: np.ndarray | None = None

    def transitions_GHz(self) -> np.ndarray:
        return np.diff(self.energies_GHz) if len(self.energies_GHz) > 1 else np.empty(0)


class SpectralSolver:
    @staticmethod
    def solve(
        H: sparse.spmatrix,
        levels: int = 8,
        vectors: bool = True,
    ) -> Spectrum:
        size = H.shape[0]
        if size == 0:
            return Spectrum(np.empty(0), None)

        if size <= 4 or levels >= size:
            energies, states = np.linalg.eigh(H.toarray())
            return Spectrum(
                energies[:levels],
                states[:, :levels] if vectors else None,
            )

        k = max(1, min(levels, size - 1))
        energies, states = spla.eigsh(H.tocsr(), k=k, which="SA")
        order = np.argsort(energies)

        return Spectrum(
            energies[order],
            states[:, order] if vectors else None,
        )


class ModeAnalysis:
    @staticmethod
    def participation_ratio(states: np.ndarray) -> np.ndarray:
        if states.ndim != 2:
            raise ValueError("states doit être une matrice 2D.")

        probabilities = np.abs(states) ** 2
        probabilities /= np.sum(probabilities, axis=0, keepdims=True)

        inverse_participation = np.sum(probabilities**2, axis=0)
        return 1.0 / inverse_participation


class DisorderEngine:
    @staticmethod
    def frequency_disorder(
        network: QubitNetwork,
        sigma_MHz: float,
        seed: int | None = None,
    ) -> np.ndarray:
        if not np.isfinite(sigma_MHz) or sigma_MHz < 0.0:
            raise ValueError("sigma_MHz doit être positif ou nul.")

        rng = np.random.default_rng(seed)
        delta_GHz = rng.normal(0.0, sigma_MHz / 1000.0, size=network.size)
        return network.frequency_vector() + delta_GHz


class Reproducibility:
    @staticmethod
    def configuration_hash(network: QubitNetwork) -> str:
        payload = {
            "ncut": network.ncut,
            "qubits": [
                {
                    "position": q.position,
                    "material": q.material.value,
                    "EJ_GHz": q.params.EJ_GHz,
                    "EC_GHz": q.params.EC_GHz,
                    "ng": q.params.ng,
                }
                for q in network.qubits
            ],
            "couplings": [
                {"a": edge.a, "b": edge.b, "J_GHz": edge.J_GHz}
                for edge in network.couplings
            ],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class MaterialQuantumEngine:
    VERSION = "MQGE-1.1"

    def __init__(self, grid: MaterialGrid, ncut: int = 15) -> None:
        self.grid = grid
        self.network = QubitNetwork(grid, ncut=ncut)
        self.graph = CouplingGraph()

    def build_nearest_neighbor_grid(self, J_GHz: float = 0.020) -> None:
        self.graph = CouplingGraph()
        self.graph.nearest_neighbors(self.grid, J_GHz)
        self.network.add_couplings(self.graph)

    def hamiltonian(self, frequencies_GHz: np.ndarray | None = None) -> sparse.csr_matrix:
        return EffectiveHamiltonian.single_excitation(self.network, frequencies_GHz)

    def solve(self, levels: int = 8) -> Spectrum:
        return SpectralSolver.solve(self.hamiltonian(), levels=levels)

    def analyze(self, levels: int = 8) -> dict:
        spectrum = self.solve(levels=levels)
        result: dict[str, object] = {
            "version": self.VERSION,
            "qubits": self.network.size,
            "energies_GHz": spectrum.energies_GHz,
            "transitions_GHz": spectrum.transitions_GHz(),
            "configuration_hash": Reproducibility.configuration_hash(self.network),
        }
        if spectrum.states is not None:
            result["participation_ratio"] = ModeAnalysis.participation_ratio(spectrum.states)
        return result

    def export_json(self, path: str | Path, levels: int = 8) -> None:
        result = self.analyze(levels=levels)
        serializable = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in result.items()
        }
        Path(path).write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def example_engine() -> MaterialQuantumEngine:
    grid = MaterialGrid(width=4, height=4)

    for y in range(grid.height):
        for x in range(grid.width):
            grid.set_cell(
                x,
                y,
                Material.TANTALUM,
                capacitance_fF=70.0,
                josephson_energy_GHz=15.0,
            )

    grid.set_cell(
        1,
        1,
        Material.ALUMINUM,
        capacitance_fF=72.0,
        josephson_energy_GHz=15.5,
    )

    engine = MaterialQuantumEngine(grid, ncut=15)
    engine.build_nearest_neighbor_grid(J_GHz=0.020)
    return engine


def self_test() -> None:
    empty = MaterialQuantumEngine(MaterialGrid(2, 2))
    assert empty.network.size == 0

    grid = MaterialGrid(1, 1)
    grid.set_cell(0, 0, Material.ALUMINUM)
    network = QubitNetwork(grid)
    assert network.size == 1
    assert network.qubits[0].anharmonicity_GHz < 0.0

    engine = example_engine()
    H_eff = engine.hamiltonian()
    assert H_eff.shape == (16, 16)
    assert np.allclose(H_eff.toarray(), H_eff.toarray().conj().T)
    assert len(Reproducibility.configuration_hash(engine.network)) == 64


def main() -> None:
    self_test()

    engine = example_engine()
    result = engine.analyze(levels=8)

    print("=" * 72)
    print("MATERIAL-QUBIT GRID ENGINE — MQGE 1.1")
    print("=" * 72)
    print(f"Qubits actifs : {result['qubits']}")
    print(f"SHA256        : {result['configuration_hash']}")
    print("\nSpectre [GHz]")
    for index, energy in enumerate(result["energies_GHz"]):
        print(f"  mode {index:02d} : {energy:.9f}")

    print("\nParticipation ratio")
    for index, value in enumerate(result["participation_ratio"]):
        print(f"  mode {index:02d} : {value:.4f}")

    engine.export_json("mqge_example_result.json", levels=8)
    print("\nExport : mqge_example_result.json")
    print("Tests internes : PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
