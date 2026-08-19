from dataclasses import dataclass, field


@dataclass
class Episode:
    time: int
    observation: dict
    action: str
    consequence: dict
    surprise: float
    valence: float
    causes: list[str] = field(default_factory=list)


@dataclass
class EpisodicMemory:
    episodes: list[Episode] = field(
        default_factory=list
    )
    maximum: int = 500

    def remember(self, episode: Episode) -> None:
        self.episodes.append(episode)

        if len(self.episodes) > self.maximum:
            self.episodes = self.episodes[-self.maximum:]

    def recent(self, n: int = 5) -> list[Episode]:
        return self.episodes[-n:]

    def similar(self, action: str) -> list[Episode]:
        return [
            episode
            for episode in self.episodes
            if episode.action == action
        ]
