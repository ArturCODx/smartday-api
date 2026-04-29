import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from .constants import N_OBS, N_ACTIONS, NOTES_RECOMPENSES
from .env import SmartDayEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, nb_obs=N_OBS, nb_actions=N_ACTIONS, nb_neurones=256):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.Linear(nb_obs, nb_neurones), nn.ReLU(),
            nn.Linear(nb_neurones, nb_neurones), nn.ReLU(),
            nn.Linear(nb_neurones, nb_actions),
        )

    def forward(self, x):
        return self.reseau(x)


class ReplayBuffer:
    def __init__(self, capacite=50_000):
        self.memoire = deque(maxlen=capacite)

    def ajouter(self, etat, action, recompense, etat_suivant, termine, masque_suivant=None):
        if masque_suivant is None:
            masque_suivant = np.ones(N_ACTIONS, dtype=bool)
        self.memoire.append((etat, action, recompense, etat_suivant, float(termine), masque_suivant.astype(bool)))

    def echantillonner(self, taille_lot):
        lot = random.sample(self.memoire, taille_lot)
        etats, actions, recompenses, etats_suivants, termines, masques = zip(*lot)
        return (
            np.array(etats,          dtype=np.float32),
            np.array(actions,        dtype=np.int64),
            np.array(recompenses,    dtype=np.float32),
            np.array(etats_suivants, dtype=np.float32),
            np.array(termines,       dtype=np.float32),
            np.array(masques,        dtype=bool),
        )

    def __len__(self):
        return len(self.memoire)


class DQNAgent:
    def __init__(self, nb_obs=N_OBS, nb_actions=N_ACTIONS, nb_neurones=256,
                 lr=1e-4, gamma=0.90, epsilon=0.0, epsilon_min=0.05,
                 epsilon_decay=0.999, taille_lot=64, tau=0.005):
        self.nb_actions    = nb_actions
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.taille_lot    = taille_lot
        self.tau           = tau

        self.reseau_principal = QNetwork(nb_obs, nb_actions, nb_neurones).to(device)
        self.reseau_cible     = QNetwork(nb_obs, nb_actions, nb_neurones).to(device)
        self.reseau_cible.load_state_dict(self.reseau_principal.state_dict())
        self.reseau_cible.eval()
        self.optimiseur = optim.Adam(self.reseau_principal.parameters(), lr=lr)
        self.memoire    = ReplayBuffer()

    def charger(self, chemin: str):
        state_dict = torch.load(chemin, map_location=device)
        self.reseau_principal.load_state_dict(state_dict)
        self.reseau_cible.load_state_dict(state_dict)
        self.reseau_principal.eval()
        self.epsilon = 0.0
        print(f"Modele charge depuis {chemin}")

    def choisir_action(self, obs, env=None):
        actions_valides = env.get_valid_actions() if env else list(range(self.nb_actions))
        if not actions_valides:
            return 0
        if random.random() < self.epsilon:
            return random.choice(actions_valides)
        with torch.no_grad():
            etat     = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = self.reseau_principal(etat).squeeze(0).clone()
            masque   = torch.full((self.nb_actions,), float("-inf"), device=device)
            masque[actions_valides] = q_values[actions_valides]
            return int(masque.argmax().item())

    def memoriser(self, etat, action, recompense, etat_suivant, termine, masque_suivant=None):
        self.memoire.ajouter(etat, action, recompense, etat_suivant, float(termine), masque_suivant)

    def apprendre(self):
        if len(self.memoire) < self.taille_lot:
            return None
        etats, actions, recompenses, etats_suivants, termines, masques = self.memoire.echantillonner(self.taille_lot)
        etats          = torch.tensor(etats,          device=device)
        actions        = torch.tensor(actions,        device=device)
        recompenses    = torch.tensor(recompenses,    device=device)
        etats_suivants = torch.tensor(etats_suivants, device=device)
        termines       = torch.tensor(termines,       device=device)
        masques        = torch.tensor(masques,        device=device)

        q_actuelles = self.reseau_principal(etats).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            q_suiv_principal = self.reseau_principal(etats_suivants).clone()
            q_suiv_principal[~masques] = float("-inf")
            actions_suiv = q_suiv_principal.argmax(1)
            q_suiv_cible = self.reseau_cible(etats_suivants).gather(1, actions_suiv.unsqueeze(1)).squeeze()
            q_cibles = recompenses + self.gamma * q_suiv_cible * (1 - termines)

        perte = nn.HuberLoss()(q_actuelles.squeeze(), q_cibles)
        self.optimiseur.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(self.reseau_principal.parameters(), 1.0)
        self.optimiseur.step()

        for p_c, p_p in zip(self.reseau_cible.parameters(), self.reseau_principal.parameters()):
            p_c.data.copy_(self.tau * p_p.data + (1 - self.tau) * p_c.data)

        return float(perte.item())

    def rlhf(self, env: SmartDayEnv, note: int, nb_iterations: int = 50):
        r_rlhf = NOTES_RECOMPENSES[note]
        n      = len(env.taches_placees)
        for i, (debut, duree, type_t, r_auto, _) in enumerate(env.taches_placees):
            etat         = env.historique_obs[i]     if i     < len(env.historique_obs) else np.zeros(N_OBS, dtype=np.float32)
            etat_suivant = env.historique_obs[i + 1] if i + 1 < len(env.historique_obs) else np.zeros(N_OBS, dtype=np.float32)
            r_finale     = 0.5 * r_auto + 0.5 * r_rlhf
            self.memoriser(etat, debut, r_finale, etat_suivant, float(i + 1 == n))
        for _ in range(nb_iterations):
            self.apprendre()
