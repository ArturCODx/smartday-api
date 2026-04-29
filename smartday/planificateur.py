import numpy as np
from math import ceil
from .constants import PHYSIQUE, PRIORITE_TYPE, PROFIL_BASE


def simuler_fatigue(taches_placees, profil_actuel):
    profil_demain = profil_actuel.copy()
    charge        = sum(duree for _, duree, _, _, _ in taches_placees)
    fatigue       = charge * 0.01
    recuperation  = 0.02
    return np.clip(profil_demain - fatigue + recuperation, 0.1, 1.0)


class PlanificateurMensuel:

    def __init__(self, projets, planning_fixe, objectif_sport_semaine=20):
        self.projets                = [
            {**p, "creneaux_restants": p["charge_totale"], "termine": False}
            for p in projets
        ]
        self.planning_fixe          = planning_fixe
        self.objectif_sport_semaine = objectif_sport_semaine
        self.jour_actuel            = 0
        self.bilan_sport_semaine    = 0
        self.historique             = []
        self.profil_energie         = PROFIL_BASE.copy()

    def calculer_taches_du_jour(self):
        jour_semaine = self.jour_actuel % 7
        taches = []

        for debut, duree, type_t in self.planning_fixe.get(jour_semaine, []):
            taches.append({
                "type": type_t, "duree": duree,
                "est_fixee": True, "creneau_fixe": debut,
                "projet_id": "fixe",
            })

        creneaux_sport = self._calculer_pression_sport()
        if creneaux_sport > 0:
            taches.append({
                "type": PHYSIQUE, "duree": min(creneaux_sport, 2),
                "est_fixee": False, "creneau_fixe": -1,
                "projet_id": "sport",
            })

        projets_actifs = [
            p for p in self.projets
            if not p["termine"] and p["creneaux_restants"] > 0
        ]
        projets_actifs.sort(
            key=lambda p: (p["deadline"] - self.jour_actuel) / max(p["creneaux_restants"], 1)
        )

        for p in projets_actifs:
            session = self._calculer_session(p)
            if session > 0:
                taches.append({
                    "type": p["type"], "duree": session,
                    "est_fixee": False, "creneau_fixe": -1,
                    "projet_id": p["nom"],
                })

        taches.sort(key=lambda t: 0 if t["est_fixee"] else PRIORITE_TYPE[t["type"]])
        return taches

    def feedback_soir(self, taches_realisees, energie_ressentie=3):
        sport_du_jour = 0
        par_projet    = {}

        for debut, duree, type_t, _, projet_id in taches_realisees:
            if projet_id == "sport":
                sport_du_jour += duree
            elif projet_id not in ("fixe", "train"):
                par_projet[projet_id] = par_projet.get(projet_id, 0) + duree

        for p in self.projets:
            realise = par_projet.get(p["nom"], 0)
            p["creneaux_restants"] = max(0, p["creneaux_restants"] - realise)
            if p["creneaux_restants"] == 0:
                p["termine"] = True

        self.bilan_sport_semaine += sport_du_jour
        self.profil_energie       = simuler_fatigue(taches_realisees, self.profil_energie)

        self.historique.append({
            "jour":    self.jour_actuel,
            "sport":   sport_du_jour,
            "projets": par_projet,
            "energie": energie_ressentie,
        })

        self.jour_actuel += 1
        if self.jour_actuel % 7 == 0:
            self.bilan_sport_semaine = 0

    def get_etat(self):
        return {
            "jour_actuel":           self.jour_actuel,
            "semaine":               self.jour_actuel // 7 + 1,
            "bilan_sport_semaine":   self.bilan_sport_semaine,
            "objectif_sport":        self.objectif_sport_semaine,
            "sport_heures":          round(self.bilan_sport_semaine * 30 / 60, 1),
            "projets": [
                {
                    "nom":               p["nom"],
                    "type":              p["type"],
                    "creneaux_restants": p["creneaux_restants"],
                    "deadline":          p["deadline"],
                    "jours_restants":    p["deadline"] - self.jour_actuel,
                    "termine":           p["termine"],
                }
                for p in self.projets
            ],
        }

    def _calculer_pression_sport(self):
        jour_semaine   = self.jour_actuel % 7
        jours_restants = 7 - jour_semaine
        manquants      = self.objectif_sport_semaine - self.bilan_sport_semaine
        if manquants <= 0 or jours_restants <= 0:
            return 0
        return min(ceil(manquants / jours_restants), 4)

    def _calculer_session(self, projet):
        jours_restants = projet["deadline"] - self.jour_actuel
        if jours_restants <= 0:
            return min(projet["creneaux_restants"], 4)
        return min(max(1, ceil(projet["creneaux_restants"] / jours_restants)), 4)
