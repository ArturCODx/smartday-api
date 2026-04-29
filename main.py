"""
SmartDay API — FastAPI server
Expose le DQN + PlanificateurMensuel via HTTP/JSON.

Routes :
  GET  /                        → statut du serveur
  GET  /etat                    → état du mois en cours
  POST /nouvelle-journee        → génère le planning du jour suivant
  GET  /planning/{jour}         → récupère le planning d'un jour donné
  POST /feedback-soir           → enregistre les tâches faites + note RLHF
  GET  /dashboard               → stats complètes du mois
  POST /reset                   → repart de zéro
"""

import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from smartday import (
    DQNAgent, SmartDayEnv, PlanificateurMensuel,
    SLOT_LABELS, TYPE_NOMS, TYPE_COULEURS,
    N_SLOTS, COGNITIF, PHYSIQUE, SOCIAL, ADMIN,
    PROFIL_BASE,
)

# ──────────────────────────────────────────────────────────────────
# Configuration du mois (modifiable selon l'utilisateur)
# ──────────────────────────────────────────────────────────────────

PROJETS_DU_MOIS = [
    {"nom": "Rapport IA",     "type": COGNITIF, "charge_totale": 20, "deadline": 20},
    {"nom": "Projet Math",    "type": COGNITIF, "charge_totale": 12, "deadline": 12},
    {"nom": "Presentation",   "type": SOCIAL,   "charge_totale":  8, "deadline": 18},
    {"nom": "Emails & Admin", "type": ADMIN,    "charge_totale": 20, "deadline": 30},
    {"nom": "Projet Perso",   "type": COGNITIF, "charge_totale": 10, "deadline": 28},
]

# Travail fixe : lundi-vendredi 8h00-10h00 (slots 0-3)
PLANNING_FIXE = {
    0: [(0, 4, COGNITIF)],
    1: [(0, 4, COGNITIF)],
    2: [(0, 4, COGNITIF)],
    3: [(0, 4, COGNITIF)],
    4: [(0, 4, COGNITIF)],
}

OBJECTIF_SPORT = 20  # créneaux/semaine = 10h

# ──────────────────────────────────────────────────────────────────
# Initialisation globale
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SmartDay API",
    description="Planificateur journalier intelligent par Double DQN",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle DQN
agent = DQNAgent()
MODEL_PATH = os.getenv("MODEL_PATH", "smartday_v5_best.pth")
if os.path.exists(MODEL_PATH):
    agent.charger(MODEL_PATH)
    print(f"Modèle chargé : {MODEL_PATH}")
else:
    print(f"ATTENTION : modèle {MODEL_PATH} introuvable — l'agent jouera de façon non entraînée")

# Planificateur et stockage des plannings
planificateur: PlanificateurMensuel = None
plannings_du_mois: dict = {}   # { jour: liste de tâches placées }


def init_planificateur():
    global planificateur, plannings_du_mois
    planificateur   = PlanificateurMensuel(PROJETS_DU_MOIS, PLANNING_FIXE, OBJECTIF_SPORT)
    plannings_du_mois = {}


init_planificateur()

# ──────────────────────────────────────────────────────────────────
# Modèles Pydantic (schémas JSON)
# ──────────────────────────────────────────────────────────────────

class TacheRealisee(BaseModel):
    projet_id: str
    duree:     int
    type_t:    int
    debut:     int

class FeedbackSoir(BaseModel):
    taches_realisees: List[TacheRealisee]
    note_rlhf:        Optional[int] = None   # 1-5, optionnel
    energie_ressentie: Optional[int] = 3     # 1-5

# ──────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────

@app.get("/")
def racine():
    return {
        "service":  "SmartDay API",
        "version":  "1.0.0",
        "jour":     planificateur.jour_actuel,
        "modele":   MODEL_PATH if os.path.exists(MODEL_PATH) else "non charge",
    }


@app.get("/etat")
def get_etat():
    """Retourne l'état complet du mois : projets, sport, jour actuel."""
    return planificateur.get_etat()


@app.post("/nouvelle-journee")
def nouvelle_journee():
    """
    Génère et retourne le planning du jour suivant.
    Le DQN place les tâches selon le profil énergie courant.
    """
    if planificateur.jour_actuel >= 30:
        raise HTTPException(status_code=400, detail="Le mois est terminé. Appelez /reset pour recommencer.")

    jour = planificateur.jour_actuel
    taches_jour = planificateur.calculer_taches_du_jour()

    if not taches_jour:
        return {
            "jour":     jour + 1,
            "message":  "Tous les projets sont terminés !",
            "planning": [],
            "etat":     planificateur.get_etat(),
        }

  # Variation quotidienne du profil energie
    profil_du_jour = planificateur.profil_energie.copy()
    bruit = np.random.normal(0, 0.05, len(profil_du_jour))
    profil_du_jour = np.clip(profil_du_jour + bruit, 0.1, 1.0).astype("float32")

    env = SmartDayEnv(tasks=taches_jour, energy_profile=profil_du_jour)
    obs, _ = env.reset()
    termine = False

    while not termine:
        valides = env.get_valid_actions()
        if not valides:
            break
        action = agent.choisir_action(obs, env=env)
        obs, _, termine, _, _ = env.step(action)

    # Sérialisation du planning
    planning_json = []
    for debut, duree, type_t, reward, projet_id in env.taches_placees:
        planning_json.append({
            "debut":        debut,
            "heure":        SLOT_LABELS[debut],
            "fin":          SLOT_LABELS[min(debut + duree - 1, N_SLOTS - 1)],
            "duree":        duree,
            "duree_min":    duree * 30,
            "type":         type_t,
            "type_nom":     TYPE_NOMS[type_t],
            "couleur":      TYPE_COULEURS[type_t],
            "projet_id":    projet_id,
            "reward":       round(reward, 3),
        })

    # Stockage pour /planning/{jour}
    plannings_du_mois[jour] = {
        "taches_placees": env.taches_placees,
        "env":            env,
    }

    return {
        "jour":          jour + 1,
        "planning":      planning_json,
        "nb_taches":     len(planning_json),
        "taux_placement": round(len(planning_json) / len(taches_jour), 2),
        "profil_energie": planificateur.profil_energie.tolist(),
        "etat":           planificateur.get_etat(),
    }


@app.get("/planning/{jour}")
def get_planning(jour: int):
    """Récupère le planning d'un jour déjà calculé (1-indexé)."""
    jour_0 = jour - 1
    if jour_0 not in plannings_du_mois:
        raise HTTPException(status_code=404, detail=f"Aucun planning trouvé pour le jour {jour}.")

    data = plannings_du_mois[jour_0]
    planning_json = []
    for debut, duree, type_t, reward, projet_id in data["taches_placees"]:
        planning_json.append({
            "debut":     debut,
            "heure":     SLOT_LABELS[debut],
            "fin":       SLOT_LABELS[min(debut + duree - 1, N_SLOTS - 1)],
            "duree":     duree,
            "duree_min": duree * 30,
            "type":      type_t,
            "type_nom":  TYPE_NOMS[type_t],
            "couleur":   TYPE_COULEURS[type_t],
            "projet_id": projet_id,
            "reward":    round(reward, 3),
        })

    return {"jour": jour, "planning": planning_json}


@app.post("/feedback-soir")
def feedback_soir(feedback: FeedbackSoir):
    """
    Enregistre les tâches réellement faites dans la journée.
    Si note_rlhf est fournie (1-5), réentraîne l'agent avec RLHF.
    """
    jour = planificateur.jour_actuel

    # Conversion pour le planificateur
    taches_realisees = [
        (t.debut, t.duree, t.type_t, 0.0, t.projet_id)
        for t in feedback.taches_realisees
    ]

    # Mise à jour du planificateur
    planificateur.feedback_soir(taches_realisees, feedback.energie_ressentie)

    # RLHF si note fournie
    rlhf_applique = False
    if feedback.note_rlhf and feedback.note_rlhf in range(1, 6):
        jour_0 = jour
        if jour_0 in plannings_du_mois:
            env = plannings_du_mois[jour_0]["env"]
            agent.rlhf(env, feedback.note_rlhf)
            rlhf_applique = True

    return {
        "jour":          jour + 1,
        "rlhf_applique": rlhf_applique,
        "message":       f"Feedback enregistré pour le jour {jour + 1}",
        "etat":          planificateur.get_etat(),
    }


@app.get("/dashboard")
def dashboard():
    """Stats complètes du mois en cours."""
    historique = planificateur.historique

    # Sport par semaine
    sport_par_semaine = []
    semaine_courante  = 0
    for h in historique:
        s = h["jour"] // 7
        while len(sport_par_semaine) <= s:
            sport_par_semaine.append(0)
        sport_par_semaine[s] += h["sport"]

    # Scores journaliers
    scores_journaliers = [
        plannings_du_mois[j]["taches_placees"]
        for j in sorted(plannings_du_mois)
        if j in plannings_du_mois
    ]
    scores = [
        round(sum(r for _, _, _, r, _ in tp), 2)
        for tp in scores_journaliers
    ]

    return {
        "jour_actuel":      planificateur.jour_actuel,
        "jours_restants":   30 - planificateur.jour_actuel,
        "projets":          planificateur.get_etat()["projets"],
        "sport_par_semaine": [
            {"semaine": i + 1, "creneaux": s, "heures": round(s * 30 / 60, 1),
             "objectif_atteint": s >= OBJECTIF_SPORT}
            for i, s in enumerate(sport_par_semaine)
        ],
        "scores_journaliers": scores,
        "score_moyen":        round(sum(scores) / len(scores), 2) if scores else 0,
        "historique_energie": [h["energie"] for h in historique],
    }


@app.post("/reset")
def reset():
    """Remet le mois à zéro — nouveau mois, nouveaux projets."""
    init_planificateur()
    return {"message": "Nouveau mois démarré !", "etat": planificateur.get_etat()}
