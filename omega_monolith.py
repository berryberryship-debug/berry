# -*- coding: utf-8 -*-

"""
[Système 18.1 | Oméga] Monolithe Unifié : 
- Gestionnaire de Flux Mémoire (NanoIPC / mmap)
- Substrat Bio-Cognitif & Pharmacopée
- Processeur de Flux Langagier (TurnTalkProcessor)
- Laboratoire d'Expérimentation Zener & Modélisation Prédictive (AIModel)
"""

import os
import mmap
import ctypes
import re
import random
import statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from sklearn.linear_model import LinearRegression


# =====================================================================
# 1. NOYAU DE FLUX MÉMOIRE (NanoIPC & mmap)
# =====================================================================

SLOT_COUNT = 4
SLOT_DATA = 32

class CompactSlot(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("seq", ctypes.c_uint32),
        ("payload", ctypes.c_uint8 * SLOT_DATA),
    ]

class TermuxRing(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("slots", CompactSlot * SLOT_COUNT),
    ]

class NanoIPC:
    def __init__(self, filename: str = ".om_flux"):
        self.path = os.path.expanduser(f"~/{filename}")
        self.size = ctypes.sizeof(TermuxRing)
        self.fd = None
        self.mem = None
        self.ring = None

    def init_flux(self):
        exists = os.path.exists(self.path)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        os.ftruncate(self.fd, self.size)
        
        self.mem = mmap.mmap(self.fd, self.size, access=mmap.ACCESS_WRITE)
        self.ring = TermuxRing.from_buffer(self.mem)
        
        if not exists:
            ctypes.memset(ctypes.addressof(self.ring), 0, self.size)

    def write(self, slot_idx: int, data_bytes: bytes):
        if not self.ring:
            return
        slot = self.ring.slots[slot_idx % SLOT_COUNT]
        slot.seq += 1
        payload_len = min(len(data_bytes), SLOT_DATA)
        slot.payload[:payload_len] = data_bytes[:payload_len]
        if payload_len < SLOT_DATA:
            ctypes.memset(ctypes.addressof(slot.payload) + payload_len, 0, SLOT_DATA - payload_len)
        slot.seq += 1

    def read(self, slot_idx: int) -> bytes:
        if not self.ring:
            return b""
        slot = self.ring.slots[slot_idx % SLOT_COUNT]
        while True:
            s1 = slot.seq
            if s1 & 1:
                continue
            data = bytes(slot.payload)
            s2 = slot.seq
            if s1 == s2:
                return data

    def close(self):
        if self.mem:
            self.mem.close()
        if self.fd:
            os.close(self.fd)


# =====================================================================
# 2. SUBSTRAT BIOCHIMIQUE & CARTOGRAPHIE (Pharmacopée)
# =====================================================================

PHARMACOPEE = {
    "terpenes": {
        "myrcene": {"arome": "terre, musc, mangue", "effets_etudies": ["relaxation potentielle", "effets anti-inflammatoires étudiés"]},
        "limonene": {"arome": "citron, agrumes", "effets_etudies": ["humeur", "anxiété étudiée", "activité anti-inflammatoire"]},
        "alpha_pinene": {"arome": "pin, résine", "effets_etudies": ["vigilance", "activité anti-inflammatoire étudiée"]},
        "beta_pinene": {"arome": "pin, herbacé", "effets_etudies": ["activité antimicrobienne étudiée"]},
        "beta_caryophyllene": {"arome": "poivre, épices", "particularite": "interaction avec CB2", "effets_etudies": ["anti-inflammatoire", "analgésique potentiel"]},
        "linalol": {"arome": "lavande, floral", "effets_etudies": ["relaxation", "anxiété étudiée", "sommeil étudié"]},
        "terpinolene": {"arome": "floral, pin", "effets_etudies": ["activité antioxydante étudiée"]},
        "ocimene": {"arome": "sucré, herbacé", "effets_etudies": ["activité antimicrobienne étudiée"]},
        "humulene": {"arome": "boisé, houblon, terreux", "effets_etudies": ["anti-inflammatoire potentiel"]},
        "farnesene": {"arome": "pomme verte, floral", "effets_etudies": ["activité antioxydante étudiée"]},
        "germacrene": {"arome": "boisé, épicé", "effets_etudies": ["activité antimicrobienne étudiée"]},
        "eudesmol": {"arome": "boisé, doux", "famille": "sesquiterpène alcool", "effets_etudies": ["anti-inflammatoire étudié", "effets neurobiologiques explorés"]}
    },
    "cannabinoides": ["THC", "CBD", "CBG", "CBC", "CBN", "THCV", "CBDV"],
    "neurotransmetteurs": ["dopamine", "serotonine", "GABA", "glutamate", "acetylcholine", "noradrenaline"],
    "hormones_endocriniennes": ["cortisol", "melatonine", "insuline", "testosterone", "oestrogenes", "hormone_de_croissance"],
    "neuropeptides": ["endorphines", "encephalines", "dynorphines", "oxytocine", "vasopressine", "orexines"],
    "systemes": ["immunitaire", "endocannabinoide", "axe_stress_HPA", "rythmes_circadiens"]
}

class DemeleurPharmacopee:
    def __init__(self, data: dict):
        self.data = data
        self.index_effets = self._indexer_par_effet()

    def _indexer_par_effet(self) -> Dict[str, List[str]]:
        index = {}
        terpenes = self.data.get("terpenes", {})
        for mol, details in terpenes.items():
            for effet in details.get("effets_etudies", []):
                effet_cle = effet.lower()
                if effet_cle not in index:
                    index[effet_cle] = []
                index[effet_cle].append(mol)
        return index

    def chercher_par_effet(self, effet_recherche: str) -> List[Tuple[str, List[str]]]:
        resultats = []
        effet_recherche = effet_recherche.lower()
        for effet, mols in self.index_effets.items():
            if effet_recherche in effet:
                resultats.append((effet, mols))
        return resultats


# =====================================================================
# 3. PROCESSEUR DE FLUX LANGAGIER (TurnTalkProcessor)
# =====================================================================

FILLERS = {"um", "uh", "euh", "ben", "like", "genre"}
PAUSE_RE = re.compile(r"\[pause(?:=(\d+(?:\.\d+)?))?\]", re.I)
WORD_RE = re.compile(r"[\wÀ-ÿ']+|[.,!?;:()\[\]]")

@dataclass
class TurnResult:
    cleaned_text: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

class TurnTalkProcessor:
    def __init__(self, pause_threshold: float = 0.45, long_pause_threshold: float = 1.1, max_buffer_chars: int = 260):
        self.pause_threshold = pause_threshold
        self.long_pause_threshold = long_pause_threshold
        self.max_buffer_chars = max_buffer_chars

    def tokenize(self, text: str) -> List[str]:
        return WORD_RE.findall(text)

    def process(self, text: str) -> TurnResult:
        events = []
        tokens = self.tokenize(text)
        out = []
        pause_acc = 0.0
        pause_count = 0
        filler_count = 0
        technical_mode = False
        buffer_chars = 0
        score = 0.0

        for tok in tokens:
            low = tok.lower()
            m = PAUSE_RE.fullmatch(tok)
            if m:
                dur = float(m.group(1) or 0.3)
                pause_acc += dur
                pause_count += 1
                events.append({"event": "pause", "duration": dur})
                if dur >= self.long_pause_threshold:
                    out.append(" <pause>")
                continue

            if low in FILLERS:
                filler_count += 1
                events.append({"event": "filler", "token": tok})
                if technical_mode:
                    continue
                out.append(tok)
                continue

            if re.fullmatch(r"\d+(?:\.\d+)?", tok) or re.fullmatch(r"[A-Z]{2,}", tok):
                technical_mode = True

            if tok in {".", "?", "!"}:
                technical_mode = False

            if not out:
                out.append(tok)
            elif tok in {".", ",", "?", "!", ";", ":"}:
                out[-1] = out[-1].rstrip() + tok
            elif out[-1].endswith("("):
                out.append(tok)
            else:
                out.append(" " + tok)

            buffer_chars += len(tok)
            if buffer_chars > self.max_buffer_chars:
                events.append({"event": "buffer_flush", "chars": buffer_chars})
                buffer_chars = 0

        cleaned = "".join(out).replace("  ", " ").strip()
        if pause_count:
            score += min(1.0, pause_count / 4)
        if filler_count:
            score += min(1.0, filler_count / 3)
        if pause_acc >= self.long_pause_threshold:
            score += 0.5
        if technical_mode:
            score += 0.25
        if len(cleaned) > self.max_buffer_chars:
            score += 0.25

        metrics = {
            "pause_count": pause_count,
            "pause_seconds": round(pause_acc, 2),
            "filler_count": filler_count,
            "robustness_risk": round(min(1.0, score / 3), 3),
        }
        return TurnResult(cleaned_text=cleaned, events=events, metrics=metrics)


# =====================================================================
# 4. CONFIGURATION, SESSION & LABORATOIRE ZENER COGNITIF
# =====================================================================

@dataclass
class Config:
    trials: int = 25
    symbols: int = 5
    sessions: int = 3

@dataclass
class Session:
    predicted: List[int]
    actual: List[int]
    score: int
    fatigue: int
    focus: int
    mood: int

def generate_sequence(length: int, symbols: int) -> List[int]:
    return [random.randint(0, symbols - 1) for _ in range(length)]

def score_sequence(pred: List[int], actual: List[int]) -> int:
    return sum(p == a for p, a in zip(pred, actual))

class Analyzer:
    @staticmethod
    def summary(scores: List[int]) -> dict:
        return {
            "mean": statistics.mean(scores),
            "max": max(scores),
            "min": min(scores),
            "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0
        }

    @staticmethod
    def correlation(sessions: List[Session]):
        if len(sessions) < 2:
            return
        scores = np.array([s.score for s in sessions])
        fatigue = np.array([s.fatigue for s in sessions])
        focus = np.array([s.focus for s in sessions])
        mood = np.array([s.mood for s in sessions])

        def safe_corr(x, y):
            if np.std(x) == 0 or np.std(y) == 0:
                return 0.0
            return np.corrcoef(x, y)[0, 1]

        print("\n📊 Corrélations (État interne vs Score) :")
        print(f" - Score vs Fatigue : {safe_corr(scores, fatigue):.2f}")
        print(f" - Score vs Focus   : {safe_corr(scores, focus):.2f}")
        print(f" - Score vs Mood    : {safe_corr(scores, mood):.2f}")

class Heatmap:
    @staticmethod
    def plot_session(predicted, actual, symbols=5, title="Session"):
        matrix = np.zeros((symbols, symbols))
        for p, a in zip(predicted, actual):
            matrix[a][p] += 1
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues")
        plt.title(title)
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.tight_layout()
        plt.show()

class Performance:
    @staticmethod
    def detect_peaks(sessions: List[Session], threshold: int = 8) -> List[tuple]:
        return [(i, s.score) for i, s in enumerate(sessions) if s.score >= threshold]

    @staticmethod
    def print_peaks(peaks: List[tuple]):
        if not peaks:
            return
        print("\n🔥 Sessions de haute performance :")
        for idx, score in peaks:
            print(f"Session {idx} → Score {score}")

class AIModel:
    def __init__(self):
        self.model = LinearRegression()
        self.trained = False

    def train(self, sessions: List[Session]):
        if len(sessions) < 3:
            return
        X = np.array([[s.fatigue, s.focus, s.mood] for s in sessions])
        y = np.array([s.score for s in sessions])
        self.model.fit(X, y)
        self.trained = True

    def predict_score(self, fatigue: int, focus: int, mood: int) -> float:
        if not self.trained:
            return 0.0
        return float(self.model.predict(np.array([[fatigue, focus, mood]]))[0])

    def find_optimal_state(self):
        if not self.trained:
            return None
        best_score = -float('inf')
        optimal_state = None
        for f in range(1, 11):
            for fo in range(1, 11):
                for m in range(1, 11):
                    pred = self.predict_score(f, fo, m)
                    if pred > best_score:
                        best_score = pred
                        optimal_state = (f, fo, m)
        return optimal_state, best_score


# =====================================================================
# 5. EXÉCUTION MONOLITHIQUE UNIFIÉE
# =====================================================================

def run_omega_monolith():
    print("--- [Système 18.1 | Oméga] Initialisation du Monolithe ---")
    
    # 1. Test rapide du processeur de flux textuel
    processor = TurnTalkProcessor()
    sample_text = "Euh [pause=1.2] je veux tester ce test 42 avec Python."
    res = processor.process(sample_text)
    print(f"\n[Test Texte Nettoyé] -> {res.cleaned_text}")
    print(f"[Métriques Texte] -> {res.metrics}")

    # 2. Test rapide du démêlage pharmacologique
    demeleur = DemeleurPharmacopee(PHARMACOPEE)
    relax = demeleur.chercher_par_effet("relaxation")
    print(f"\n[Test Pharmacopée] Molécules relaxantes : {relax}")

    # 3. Test du gestionnaire IPC mémoire
    ipc = NanoIPC()
    ipc.init_flux()
    ipc.write(0, b"OMEGA_SYNC_OK")
    val = ipc.read(0)
    print(f"\n[Test IPC mmap] Lecture du flux mémoire : {val.rstrip(b'\\x00').decode('utf-8')}")
    ipc.close()

    # 4. Simulation rapide du laboratoire Zener & IA
    config = Config()
    analyzer = Analyzer()
    ai_model = AIModel()
    sessions = []

    print("\n--- [Simulation Laboratoire Zener] ---")
    for i in range(config.sessions):
        fatigue, focus, mood = random.randint(2, 8), random.randint(3, 9), random.randint(4, 8)
        predicted = generate_sequence(config.trials, config.symbols)
        actual = generate_sequence(config.trials, config.symbols)
        s = score_sequence(predicted, actual)

        sessions.append(Session(predicted, actual, s, fatigue, focus, mood))
        print(f"Session {i+1} simulée | Score : {s}/{config.trials} (Fatigue:{fatigue}, Focus:{focus}, Mood:{mood})")

    stats = analyzer.summary([s.score for s in sessions])
    print(f"\nRésultats globaux -> Moyenne : {stats['mean']:.2f} | Max : {stats['max']}")
    
    ai_model.train(sessions)
    optimal = ai_model.find_optimal_state()
    if optimal:
        state, est = optimal
        print(f"🧠 État optimal prédit par l'IA : Fatigue={state[0]}, Focus={state[1]}, Mood={state[2]} (Score estimé : {est:.2f})")

    print("\n--- [Système 18.1 | Oméga] Intégrité 100%. Blindage déployé. ---")

if __name__ == "__main__":
    run_omega_monolith()

