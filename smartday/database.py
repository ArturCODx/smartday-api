import os
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Boolean, JSON
from sqlalchemy.orm import declarative_base, Session

_url = os.getenv("DATABASE_URL", "")
if _url.startswith("postgres://"):
    _url = _url.replace("postgres://", "postgresql://", 1)

engine = create_engine(_url, pool_pre_ping=True) if _url else None
Base = declarative_base()


class DbEtat(Base):
    __tablename__ = "etat"
    id                  = Column(Integer, primary_key=True)
    jour_actuel         = Column(Integer, default=0)
    bilan_sport_semaine = Column(Integer, default=0)
    objectif_sport      = Column(Integer, default=20)
    profil_energie      = Column(JSON)


class DbProjet(Base):
    __tablename__ = "projets"
    nom               = Column(String, primary_key=True)
    type              = Column(Integer)
    charge_totale     = Column(Integer)
    creneaux_restants = Column(Integer)
    deadline          = Column(Integer)
    termine           = Column(Boolean, default=False)


class DbCreneau(Base):
    __tablename__ = "creneaux_fixes"
    jour  = Column(Integer, primary_key=True)
    debut = Column(Integer, primary_key=True)
    duree = Column(Integer)
    type  = Column(Integer)
    nom   = Column(String)


class DbHistorique(Base):
    __tablename__ = "historique"
    jour    = Column(Integer, primary_key=True)
    sport   = Column(Integer)
    projets = Column(JSON)
    energie = Column(Integer)


class DbPlanning(Base):
    __tablename__ = "plannings"
    jour           = Column(Integer, primary_key=True)
    taches_placees = Column(JSON)
    historique_obs = Column(JSON)


def init_db():
    if engine:
        Base.metadata.create_all(engine)


def charger_tout():
    if not engine:
        return None
    with Session(engine) as s:
        etat = s.get(DbEtat, 1)
        if not etat:
            return None

        projets = [
            {
                "nom": p.nom, "type": p.type,
                "charge_totale": p.charge_totale,
                "creneaux_restants": p.creneaux_restants,
                "deadline": p.deadline, "termine": p.termine,
            }
            for p in s.query(DbProjet).all()
        ]

        planning_fixe, creneaux_noms = {}, {}
        for c in s.query(DbCreneau).all():
            planning_fixe.setdefault(c.jour, []).append((c.debut, c.duree, c.type))
            creneaux_noms[(c.jour, c.debut)] = c.nom

        historique = [
            {"jour": h.jour, "sport": h.sport, "projets": h.projets, "energie": h.energie}
            for h in s.query(DbHistorique).order_by(DbHistorique.jour).all()
        ]

        plannings = {
            p.jour: {
                "taches_placees": [tuple(t) for t in p.taches_placees],
                "historique_obs": [np.array(o, dtype=np.float32) for o in p.historique_obs],
            }
            for p in s.query(DbPlanning).all()
        }

        return {
            "jour_actuel":         etat.jour_actuel,
            "bilan_sport_semaine": etat.bilan_sport_semaine,
            "objectif_sport":      etat.objectif_sport,
            "profil_energie":      np.array(etat.profil_energie, dtype="float32"),
            "projets":             projets,
            "planning_fixe":       planning_fixe,
            "creneaux_noms":       creneaux_noms,
            "historique":          historique,
            "plannings":           plannings,
        }


def sauvegarder_etat(jour_actuel, bilan_sport, objectif, profil_energie):
    if not engine:
        return
    with Session(engine) as s:
        etat = s.get(DbEtat, 1)
        if not etat:
            etat = DbEtat(id=1)
            s.add(etat)
        etat.jour_actuel         = jour_actuel
        etat.bilan_sport_semaine = bilan_sport
        etat.objectif_sport      = objectif
        etat.profil_energie      = profil_energie.tolist()
        s.commit()


def sauvegarder_projets(projets):
    if not engine:
        return
    with Session(engine) as s:
        s.query(DbProjet).delete()
        for p in projets:
            s.add(DbProjet(
                nom=p["nom"], type=p["type"],
                charge_totale=p["charge_totale"],
                creneaux_restants=p["creneaux_restants"],
                deadline=p["deadline"], termine=p["termine"],
            ))
        s.commit()


def sauvegarder_creneaux(planning_fixe, creneaux_noms):
    if not engine:
        return
    with Session(engine) as s:
        s.query(DbCreneau).delete()
        for jour, liste in planning_fixe.items():
            for debut, duree, type_t in liste:
                s.add(DbCreneau(
                    jour=jour, debut=debut, duree=duree,
                    type=type_t, nom=creneaux_noms.get((jour, debut), "Fixe"),
                ))
        s.commit()


def sauvegarder_historique(historique):
    if not engine:
        return
    with Session(engine) as s:
        s.query(DbHistorique).delete()
        for h in historique:
            s.add(DbHistorique(
                jour=h["jour"], sport=h["sport"],
                projets=h["projets"], energie=h["energie"],
            ))
        s.commit()


def sauvegarder_planning(jour, taches_placees, historique_obs):
    if not engine:
        return
    with Session(engine) as s:
        existing = s.get(DbPlanning, jour)
        if not existing:
            existing = DbPlanning(jour=jour)
            s.add(existing)
        existing.taches_placees = [list(t) for t in taches_placees]
        existing.historique_obs = [o.tolist() for o in historique_obs]
        s.commit()


def reset_db():
    if not engine:
        return
    with Session(engine) as s:
        s.query(DbPlanning).delete()
        s.query(DbHistorique).delete()
        s.query(DbCreneau).delete()
        s.query(DbProjet).delete()
        s.query(DbEtat).delete()
        s.commit()
