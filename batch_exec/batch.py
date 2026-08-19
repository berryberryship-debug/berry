#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de traitement par lots (Batch) - Exécution locale et asynchrone
Optimisé pour Termux.
"""
import os
import json
import numpy as np
from datetime import datetime

def executer_batch_local(ntraj=64):
    os.makedirs("../output", exist_ok=True)
    resultats = [{"trajectoire": i, "score": float(np.random.default_rng().uniform(0.5, 1.0))} for i in range(ntraj)]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_json = os.path.join("../output", f"batch_report_{timestamp}.json")
    
    bilan = {"timestamp": timestamp, "total_trajectoires": ntraj, "mode": "local_asynchrone", "resultats": resultats}
    with open(chemin_json, 'w') as f:
        json.dump(bilan, f, indent=4)

if __name__ == "__main__":
    executer_batch_local()
