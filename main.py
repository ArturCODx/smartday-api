"""
SmartDay API — FastAPI server V2
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

app = FastAPI(
    title="SmartDay API",
    description="Planificateur journalier intelligent par Double DQN",
    version="2.0.0",
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
else:
    print(f"ATTENTION : modèle {MODEL_PATH} introuvable")

# État global
planificateur: PlanificateurMensuel = None
plannings_du_mois: dict = {}
creneaux_fixes_noms: dict = {}  # { (jour, debut): nom }


def init_planificateur():
    global planificateur, plannings_du_mois, creneaux_fixes_noms
    planificateur      = PlanificateurMensuel([], {}, 20)
    plannings_du_mois  = {}
    creneaux_fixes_noms = {}


init_planificateur()

# ── Modèles Pydantic ──────────────────────────────────────────────

class ProjetConfig(BaseModel):
    nom:           str
    type:          int
    charge_totale: int
    deadline:      int

class CreneauFixeConfig(BaseModel):
    nom:         str
    jour:        int
    debut:       int
    duree:       int
    type:        int

class TacheRealisee(BaseModel):
    projet_id: str
    duree:     int
    type_t:    int
    debut:     int

class FeedbackSoir(BaseModel):
    taches_realisees:  List[TacheRealisee]
    note_rlhf:         Optional[int] = None
    energie_ressentie: Optional[int] = 3

# ── Routes ────────────────────────────────────────────────────────

@app.get("/")
def racine():
    return {
        "service": "SmartDay API",
        "version": "2.0.0",
        "jour":    planificateur.jour_actuel if planificateur else 0,
    }


@app.get("/etat")
def get_etat():
    return planificateur.get_etat()


@app.post("/projets")
def ajouter_projet(projet: ProjetConfig):
    """Ajoute un projet sans écraser les existants."""
    # Vérifier si le projet existe déjà
    for p in planificateur.projets:
        if p["nom"] == projet.nom:
            raise HTTPException(status_code=400, detail=f"Le projet '{projet.nom}' existe déjà.")

    planificateur.projets.append({
        "nom":               projet.nom,
        "type":              projet.type,
        "charge_totale":     projet.charge_totale,
        "creneaux_restants": projet.charge_totale,
        "deadline":          projet.deadline,
        "termine":           False,
    })
    return {"message": f"Projet '{projet.nom}' ajouté.", "etat": planificateur.get_etat()}


@app.delete("/projets/{nom}")
def supprimer_projet(nom: str):
    """Supprime un projet par son nom."""
    avant = len(planificateur.projets)
    planificateur.projets = [p for p in planificateur.projets if p["nom"] != nom]
    if len(planificateur.projets) == avant:
        raise HTTPException(status_code=404, detail=f"Projet '{nom}' introuvable.")
    return {"message": f"Projet '{nom}' supprimé.", "etat": planificateur.get_etat()}


@app.post("/creneaux")
def ajouter_creneau(creneau: CreneauFixeConfig):
    """Ajoute un créneau fixe sans écraser les existants."""
    jour = creneau.jour
    if jour not in planificateur.planning_fixe:
        planificateur.planning_fixe[jour] = []

    planificateur.planning_fixe[jour].append((creneau.debut, creneau.duree, creneau.type))
    creneaux_fixes_noms[(jour, creneau.debut)] = creneau.nom

    return {"message": f"Créneau '{creneau.nom}' ajouté.", "etat": planificateur.get_etat()}


@app.delete("/creneaux/{jour}/{debut}")
def supprimer_creneau(jour: int, debut: int):
    """Supprime un créneau fixe par jour et heure de début."""
    if jour not in planificateur.planning_fixe:
        raise HTTPException(status_code=404, detail="Créneau introuvable.")

    avant = len(planificateur.planning_fixe[jour])
    planificateur.planning_fixe[jour] = [
        c for c in planificateur.planning_fixe[jour] if c[0] != debut
    ]
    if len(planificateur.planning_fixe[jour]) == avant:
        raise HTTPException(status_code=404, detail="Créneau introuvable.")

    creneaux_fixes_noms.pop((jour, debut), None)
    return {"message": "Créneau supprimé.", "etat": planificateur.get_etat()}


@app.get("/creneaux")
def get_creneaux():
    """Retourne tous les créneaux fixes avec leurs noms."""
    result = []
    for jour, liste in planificateur.planning_fixe.items():
        for debut, duree, type_t in liste:
            result.append({
                "jour":      jour,
                "debut":     debut,
                "heure":     SLOT_LABELS[debut],
                "duree":     duree,
                "duree_min": duree * 30,
                "type":      type_t,
                "type_nom":  TYPE_NOMS[type_t],
                "nom":       creneaux_fixes_noms.get((jour, debut), "Fixe"),
            })
    return {"creneaux": result}


@app.post("/objectif-sport")
def set_objectif_sport(objectif: int):
    planificateur.objectif_sport_semaine = objectif
    return {"message": f"Objectif sport mis à jour : {objectif} créneaux/semaine"}


@app.post("/nouvelle-journee")
def nouvelle_journee():
    if planificateur.jour_actuel >= 30:
        raise HTTPException(status_code=400, detail="Le mois est terminé.")

    jour        = planificateur.jour_actuel
    taches_jour = planificateur.calculer_taches_du_jour()

    if not taches_jour:
        return {"jour": jour + 1, "message": "Tous les projets sont terminés !", "planning": []}

    # Remplacer projet_id "fixe" par le vrai nom du créneau
    jour_semaine = jour % 7
    for t in taches_jour:
        if t["est_fixee"]:
            cle = (jour_semaine, t["creneau_fixe"])
            t["projet_id"] = creneaux_fixes_noms.get(cle, "Fixe")

    profil_du_jour = planificateur.profil_energie.copy()
    bruit          = np.random.normal(0, 0.08, len(profil_du_jour))
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

    planning_json = []
    for debut, duree, type_t, reward, projet_id in env.taches_placees:
        planning_json.append({
            "debut":     debut,
            "heure":     SLOT_LABELS[debut],
            "fin":       SLOT_LABELS[min(debut + duree, N_SLOTS - 1)],
            "duree":     duree,
            "duree_min": duree * 30,
            "type":      type_t,
            "type_nom":  TYPE_NOMS[type_t],
            "couleur":   TYPE_COULEURS[type_t],
            "projet_id": projet_id,
            "reward":    round(reward, 3),
        })

    plannings_du_mois[jour] = {"taches_placees": env.taches_placees, "env": env}

    return {
        "jour":           jour + 1,
        "planning":       planning_json,
        "nb_taches":      len(planning_json),
        "taux_placement": round(len(planning_json) / max(len(taches_jour), 1), 2),
        "profil_energie": profil_du_jour.tolist(),
        "etat":           planificateur.get_etat(),
    }


@app.post("/feedback-soir")
def feedback_soir(feedback: FeedbackSoir):
    jour = planificateur.jour_actuel
    taches_realisees = [
        (t.debut, t.duree, t.type_t, 0.0, t.projet_id)
        for t in feedback.taches_realisees
    ]
    planificateur.feedback_soir(taches_realisees, feedback.energie_ressentie)

    rlhf_applique = False
    if feedback.note_rlhf and feedback.note_rlhf in range(1, 6):
        if jour in plannings_du_mois:
            env = plannings_du_mois[jour]["env"]
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
    historique         = planificateur.historique
    sport_par_semaine  = []
    for h in historique:
        s = h["jour"] // 7
        while len(sport_par_semaine) <= s:
            sport_par_semaine.append(0)
        sport_par_semaine[s] += h["sport"]

    scores = [
        round(sum(r for _, _, _, r, _ in plannings_du_mois[j]["taches_placees"]), 2)
        for j in sorted(plannings_du_mois)
    ]

    # Créneaux fixes formatés
    creneaux_list = []
    jours_noms = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    for jour, liste in planificateur.planning_fixe.items():
        for debut, duree, type_t in liste:
            creneaux_list.append({
                "jour":      jour,
                "jour_nom":  jours_noms[jour],
                "debut":     debut,
                "heure":     SLOT_LABELS[debut],
                "duree_min": duree * 30,
                "type_nom":  TYPE_NOMS[type_t],
                "nom":       creneaux_fixes_noms.get((jour, debut), "Fixe"),
            })

    return {
        "jour_actuel":       planificateur.jour_actuel,
        "jours_restants":    30 - planificateur.jour_actuel,
        "projets":           planificateur.get_etat()["projets"],
        "creneaux_fixes":    creneaux_list,
        "sport_par_semaine": [
            {
                "semaine":          i + 1,
                "creneaux":         s,
                "heures":           round(s * 30 / 60, 1),
                "objectif_atteint": s >= planificateur.objectif_sport_semaine,
            }
            for i, s in enumerate(sport_par_semaine)
        ],
        "scores_journaliers": scores,
        "score_moyen":        round(sum(scores) / len(scores), 2) if scores else 0,
    }


@app.post("/reset")
def reset():
    init_planificateur()
    return {"message": "Nouveau mois démarré !", "etat": planificateur.get_etat()}