<<<<<<< HEAD
# 🎓 EdTech - Analytics Pédagogique & Prédiction Décrochage

Projet de learning analytics pour analyser les données LMS, personnaliser les parcours éducatifs et prédire les risques de décrochage.

## Livrables Jour 1

- Docker Compose opérationnel
- Services communicants : PostgreSQL, MLflow, FastAPI, Streamlit, Prometheus, Grafana
- Architecture projet validable
- Backlog Agile initial
- Premier pipeline technique de validation connectivité

## Documentation projet

- Cahier des charges global : `docs/cahier_des_charges.md`
- Architecture technique : `docs/architecture.md`
- Backlog Agile : `docs/backlog.md`
- Validation Jour 1 : `docs/validation_jour1.md`

## Architecture technique

```text
Données LMS
   ↓
PostgreSQL
   ↓
Pipeline Pandas
   ↓
Dataset ML-ready
   ↓
MLflow + modèles ML
   ↓
FastAPI
   ↓
Streamlit dashboards
   ↓
Prometheus/Grafana monitoring
```

## Démarrage rapide

1. Copier le fichier d'environnement :

```powershell
Copy-Item .env.example .env
```

2. Lancer les services :

```powershell
docker compose up --build
```

3. Vérifier les services :

- API : http://localhost:8000
- Swagger : http://localhost:8000/docs
- Dashboard Streamlit : http://localhost:8501
- MLflow : http://localhost:5000
- Prometheus : http://localhost:9090
- Grafana : http://localhost:3000

## Identifiants Grafana

- Utilisateur : `admin`
- Mot de passe : `admin`

## Planning

- Jour 1 : Infrastructure & découverte
- Jour 2 : Data Pipeline & Processing
- Jour 3 : Machine Learning & MLOps
- Jour 4 : Deployment & API
- Jour 5 : Monitoring, tests & présentation
=======
# EdTech
>>>>>>> 4cf2d3feea0c4e29ff0e43711b428159bb730067
