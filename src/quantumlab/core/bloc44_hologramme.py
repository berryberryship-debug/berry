from __future__ import annotations
from dataclasses import dataclass, field
import cmath
import math
from typing import Iterator, Tuple

EPSILON = 1e-12

@dataclass(slots=True)
class EtatHilbert:
    amplitude_materiau: float = 1.0
    phase_champ: float = 0.0
    densite_soliton: float = 1.0

    @property
    def complex_vector(self) -> complex:
        return cmath.rect(self.amplitude_materiau * self.densite_soliton, self.phase_champ)

@dataclass(slots=True)
class SolitonPixel:
    position: float = 0.0
    vitesse: float = 1.0
    largeur: float = 0.5
    amplitude: float = 1.0

    def propager(self, dt: float) -> None:
        self.position += self.vitesse * dt

    def evaluer_densite(self, x: float) -> float:
        sec_h = 1.0 / math.cosh((x - self.position) / self.largeur)
        return self.amplitude**2 * (sec_h**2)

@dataclass(slots=True)
class MiroirPhase:
    def annuler_signal(self, etat_mat: EtatHilbert) -> Tuple[complex, complex, float]:
        z_mat = etat_mat.complex_vector
        z_miroir = cmath.rect(abs(z_mat), etat_mat.phase_champ + math.pi)
        residu = z_mat + z_miroir
        return z_mat, z_miroir, abs(residu) ** 2

@dataclass(slots=True)
class MatriceHolographique:
    etat: EtatHilbert = field(default_factory=EtatHilbert)
    soliton: SolitonPixel = field(default_factory=SolitonPixel)
    miroir: MiroirPhase = field(default_factory=MiroirPhase)
    temps: float = 0.0

    def cycle_flux(self, dt: float = 0.1) -> dict:
        self.soliton.propager(dt)
        self.temps += dt
        self.etat.densite_soliton = self.soliton.evaluer_densite(self.soliton.position)
        z_mat, z_miroir, intensite_obs = self.miroir.annuler_signal(self.etat)
        return {
            "temps": round(self.temps, 3),
            "soliton_pos": round(self.soliton.position, 3),
            "residu_surface": round(intensite_obs, 8),
        }

    def flux_conscience(self) -> Iterator[dict]:
        while True:
            yield self.cycle_flux()

if __name__ == "__main__":
    matrice = MatriceHolographique()
    flux = matrice.flux_conscience()
    print("[*] Matrice 44 Initialisée...")
    for _ in range(12):
        print(next(flux))
