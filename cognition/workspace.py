from dataclasses import dataclass, field


@dataclass
class Idee:
    nom: str
    activation: float
    source: str
    contenu: dict = field(default_factory=dict)


@dataclass
class GlobalWorkspace:
    seuil_broadcast: float = 0.35
    idee_active: Idee | None = None
    historique: list[Idee] = field(default_factory=list)

    def soumettre(
        self,
        nom: str,
        activation: float,
        source: str,
        contenu: dict | None = None
    ) -> None:
        idee = Idee(
            nom=nom,
            activation=max(0.0, min(1.0, activation)),
            source=source,
            contenu=contenu or {}
        )

        if (
            self.idee_active is None
            or idee.activation
            > self.idee_active.activation
        ):
            self.idee_active = idee

    def diffuser(self) -> Idee | None:
        if self.idee_active is None:
            return None

        if (
            self.idee_active.activation
            < self.seuil_broadcast
        ):
            return None

        self.historique.append(self.idee_active)

        if len(self.historique) > 100:
            self.historique = self.historique[-100:]

        idee = self.idee_active
        self.idee_active = None

        return idee

    def dernieres_idees(self, n: int = 5) -> list[Idee]:
        return self.historique[-n:]
