# SmartDay API

Serveur FastAPI qui expose le DQN SmartDay V5 via HTTP/JSON.

## Structure

```
smartday_api/
├── main.py                  ← API FastAPI (routes HTTP)
├── requirements.txt         ← dépendances Python
├── Procfile                 ← déploiement Railway
├── smartday_v5_best.pth     ← modèle entraîné (à copier ici)
└── smartday/
    ├── constants.py         ← constantes partagées
    ├── env.py               ← SmartDayEnv (Gymnasium)
    ← agent.py               ← DQNAgent (Double DQN)
    └── planificateur.py     ← PlanificateurMensuel
```

## Setup local

```bash
# 1. Copier le modèle entraîné depuis Colab
cp smartday_v5_best.pth smartday_api/

# 2. Installer les dépendances
cd smartday_api
pip install -r requirements.txt

# 3. Lancer le serveur
uvicorn main:app --reload

# 4. Ouvrir la doc interactive
http://localhost:8000/docs
```

## Endpoints

| Méthode | Route | Description |
|---|---|---|
| GET | `/` | Statut du serveur |
| GET | `/etat` | État du mois (projets, sport) |
| POST | `/nouvelle-journee` | Génère le planning du jour |
| GET | `/planning/{jour}` | Planning d'un jour donné |
| POST | `/feedback-soir` | Enregistre les tâches faites + RLHF |
| GET | `/dashboard` | Stats complètes du mois |
| POST | `/reset` | Remet le mois à zéro |

## Exemple d'appel

```bash
# Générer le planning du jour 1
curl -X POST http://localhost:8000/nouvelle-journee

# Envoyer le feedback du soir avec note RLHF
curl -X POST http://localhost:8000/feedback-soir \
  -H "Content-Type: application/json" \
  -d '{
    "taches_realisees": [
      {"projet_id": "fixe",     "duree": 4, "type_t": 0, "debut": 0},
      {"projet_id": "sport",    "duree": 2, "type_t": 1, "debut": 16},
      {"projet_id": "Rapport IA","duree": 2, "type_t": 0, "debut": 11}
    ],
    "note_rlhf": 4,
    "energie_ressentie": 4
  }'
```

## Déploiement Railway (gratuit)

```bash
# 1. Installer Railway CLI
npm install -g @railway/cli

# 2. Login et déploiement
railway login
railway init
railway up

# 3. Récupérer l'URL publique
railway domain
```

L'URL ressemblera à : `https://smartday-api-production.up.railway.app`
C'est cette URL que l'app Android utilisera.

## Variable d'environnement

```
MODEL_PATH=smartday_v5_best.pth   ← chemin vers le modèle PyTorch
```
