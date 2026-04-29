import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from math import ceil
from .constants import *


class SmartDayEnv(gym.Env):

    def __init__(self, tasks=None, energy_profile=None):
        super().__init__()
        self._fixed_tasks  = tasks
        self._fixed_energy = energy_profile
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(N_OBS,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_SLOTS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.planning             = np.zeros(N_SLOTS, dtype=np.float32)
        self.taches               = self._fixed_tasks or self._generer_taches()
        base = self._fixed_energy if self._fixed_energy is not None else self._generer_profil_energie()
        self.profil_energie       = base.copy()
        self.index_tache_courante = 0
        self.taches_placees       = []
        self.historique_obs       = []
        self.fin_derniere_tache   = -1
        self.type_derniere_tache  = -1
        return self._get_obs(), {}

    def _generer_profil_energie(self):
        brut  = np.random.rand(N_SLOTS)
        lisse = np.convolve(brut, np.ones(5) / 5, mode="same")
        lisse = lisse / max(lisse.max(), 1e-8)
        return np.clip(lisse, 0.1, 1.0).astype(np.float32)

    def _generer_taches(self):
        nb = random.randint(4, 7)
        taches = []
        for _ in range(nb):
            type_t    = random.randint(0, 3)
            duree     = random.randint(1, 3)
            est_fixee = random.random() < 0.15
            creneau_fixe = random.randint(0, N_SLOTS - duree) if est_fixee else -1
            taches.append({
                "type": type_t, "duree": duree,
                "est_fixee": est_fixee, "creneau_fixe": creneau_fixe,
                "projet_id": "train",
            })
        return taches

    def get_valid_actions(self):
        if self.index_tache_courante >= len(self.taches):
            return []
        tache = self.taches[self.index_tache_courante]
        duree = tache["duree"]
        repos = 0
        if self.type_derniere_tache >= 0:
            repos = RECUPERATION.get(self.type_derniere_tache, {}).get(tache["type"], 0)
        debut_min = self.fin_derniere_tache + repos if self.fin_derniere_tache >= 0 else 0
        valides = []
        for s in range(N_SLOTS):
            if s < debut_min: continue
            if s + duree > N_SLOTS: continue
            if np.any(self.planning[s: s + duree]): continue
            if tache["est_fixee"] and s != tache["creneau_fixe"]: continue
            valides.append(s)
        if not valides and self.fin_derniere_tache >= 0:
            for s in range(N_SLOTS):
                if s + duree > N_SLOTS: continue
                if np.any(self.planning[s: s + duree]): continue
                if tache["est_fixee"] and s != tache["creneau_fixe"]: continue
                valides.append(s)
        return valides

    def get_valid_action_mask(self):
        mask   = np.zeros(N_ACTIONS, dtype=bool)
        valids = self.get_valid_actions()
        for v in valids:
            mask[v] = True
        return mask

    def step(self, action):
        tache  = self.taches[self.index_tache_courante]
        debut  = int(action)
        duree  = tache["duree"]
        valides = self.get_valid_actions()
        if debut not in valides:
            return self._get_obs(), -1.0, False, False, {}

        self.historique_obs.append(self._get_obs().copy())
        self.planning[debut: debut + duree] = 1

        fatigue = FATIGUE_PAR_TYPE[tache["type"]] * duree
        fin     = debut + duree
        self.profil_energie[fin:] = np.clip(self.profil_energie[fin:] - fatigue, 0.05, 1.0)

        recompense = self._calculer_recompense_v3(tache, debut, self.fin_derniere_tache, self.type_derniere_tache)
        self.taches_placees.append((debut, duree, tache["type"], recompense, tache.get("projet_id", "?")))

        self.fin_derniere_tache  = fin
        self.type_derniere_tache = tache["type"]
        self.index_tache_courante += 1

        termine = self.index_tache_courante >= len(self.taches)
        obs     = np.zeros(N_OBS, dtype=np.float32) if termine else self._get_obs()
        return obs, recompense, termine, False, {}

    def _get_obs(self):
        tache = self.taches[self.index_tache_courante]
        infos_tache = np.array([
            tache["duree"] / N_SLOTS,
            *np.eye(4)[tache["type"]],
            float(tache["est_fixee"]),
            max(tache["creneau_fixe"], 0) / N_SLOTS,
        ], dtype=np.float32)
        budget = np.zeros(4, dtype=np.float32)
        for t in self.taches[self.index_tache_courante:]:
            budget[t["type"]] += t["duree"]
        budget /= N_SLOTS
        return np.concatenate([self.planning, infos_tache, self.profil_energie, budget])

    def _calculer_recompense_v3(self, tache, debut, ancienne_fin=-1, ancien_type=-1):
        duree        = tache["duree"]
        energie_moy  = float(np.mean(self.profil_energie[debut: debut + duree]))
        energie_id   = ENERGIE_IDEALE[tache["type"]]
        ecart_e      = abs(energie_moy - energie_id)
        r_energie    = W_ENERGIE * (1 - ecart_e)

        r_etalement = 0.0
        if ancienne_fin >= 0:
            ecart = debut - ancienne_fin
            if ecart >= 2:   r_etalement =  W_ETALEMENT
            elif ecart == 0: r_etalement = -W_ETALEMENT

        r_recup = 0.0
        if ancien_type >= 0:
            repos = RECUPERATION.get(ancien_type, {}).get(tache["type"], 0)
            if ancienne_fin >= 0 and debut - ancienne_fin < repos:
                r_recup = -W_RECUPERATION

        r_matin = 0.0
        if debut < 4:
            taches_matin = sum(1 for t in self.taches_placees if t[0] < 4)
            if taches_matin >= 2:
                r_matin = -W_MATIN

        fatigue_totale = sum(FATIGUE_PAR_TYPE[t[2]] * t[1] for t in self.taches_placees)
        fatigue_progressive = -W_FATIGUE_PROGRESSIVE * max(0, fatigue_totale - SEUIL_FATIGUE_JOUR)

        r_completion = 0.0
        if self.index_tache_courante + 1 >= len(self.taches):
            nb_placees = len(self.taches_placees) + 1
            r_completion = W_COMPLETION_FINALE * (nb_placees / len(self.taches))

        return float(r_energie + r_etalement + r_recup + r_matin + fatigue_progressive + r_completion)
