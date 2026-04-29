import numpy as np

# 28 créneaux de 30 min : 8h00 → 21h30
N_SLOTS   = 28
N_OBS     = 67
N_ACTIONS = 28

SLOT_LABELS = [f"{8 + i//2}h{'30' if i % 2 else '00'}" for i in range(N_SLOTS)]

COGNITIF, PHYSIQUE, SOCIAL, ADMIN = 0, 1, 2, 3

TYPE_NOMS = {0: "Cognitif", 1: "Physique", 2: "Social", 3: "Admin"}
TYPE_COULEURS = {0: "#e74c3c", 1: "#2ecc71", 2: "#3498db", 3: "#f39c12"}

FATIGUE_PAR_TYPE = {0: 0.04, 1: 0.08, 2: 0.02, 3: 0.01}

RECUPERATION = {
    PHYSIQUE: {COGNITIF: 2},
    COGNITIF: {COGNITIF: 1},
}

PRIORITE_TYPE = {-1: 0, 0: 2, 1: 3, 2: 4, 3: 5}

ENERGIE_IDEALE = {0: 0.85, 1: 0.70, 2: 0.55, 3: 0.30}

W_ENERGIE             = 1.20
W_ETALEMENT           = 0.40
W_RECUPERATION        = 0.60
W_MATIN               = 0.30
W_FIN_JOURNEE         = 0.30
W_COMPLETION          = 2.00
W_FATIGUE             = 0.70
W_FATIGUE_PROGRESSIVE = 0.20
W_COMPLETION_FINALE   = 3.00
SEUIL_FATIGUE_JOUR    = 0.75

NOTES_RECOMPENSES = {1: -2.0, 2: -0.5, 3: 0.5, 4: 1.5, 5: 3.0}

PROFIL_BASE = np.array([
    0.4, 0.5, 0.65, 0.85, 1.0, 0.95,
    0.90, 0.80, 0.70, 0.60, 0.50, 0.40,
    0.30, 0.25, 0.20, 0.30, 0.40, 0.55,
    0.65, 0.70, 0.65, 0.55, 0.45, 0.35,
    0.25, 0.20, 0.15, 0.10,
], dtype="float32")
