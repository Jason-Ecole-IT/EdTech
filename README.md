# 🎓 EdTech - Analytics Pédagogique & Prédiction Décrochage

Projet de learning analytics pour analyser les données LMS, personnaliser les parcours éducatifs et prédire les risques de décrochage.

## Documentation projet

- Cahier des charges global : `docs/cahier_des_charges.md`
- Architecture technique : `docs/architecture.md`
- Backlog Agile : `docs/backlog.md`
- Validation Jour 1 : `docs/validation_jour1.md`
- Validation Jour 2 : `docs/validation_jour2.md`
- Validation Jour 3 : `docs/validation_jour3.md`
- Daily meetings : `docs/daily/`

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

1. Copier le fichier d'environnement : `Copy-Item .env.example .env`

2. Lancer les services : `docker compose up --build`

3. Vérifier les services :

- API : ``http://localhost:8000``
- Swagger : ``http://localhost:8000/docs``
- Dashboard Streamlit : ``http://localhost:8501``
- MLflow : ``http://localhost:5000``
- Prometheus : ``http://localhost:9090``
- Grafana : ``http://localhost:3000``

## Lancer les différents pipelines

- Data pipeline : `python -m src.data.pipeline`
- ML Pipeline : `python -m src.ml.pipeline`

## Identifiants Grafana

- Utilisateur : `admin`
- Mot de passe : `admin`

## Planning

- Jour 1 : Infrastructure & découverte
- Jour 2 : Data Pipeline & Processing
- Jour 3 : Machine Learning & MLOps
- Jour 4 : Deployment & API
- Jour 5 : Monitoring, tests & présentation

## EdTech

Lien Data Set : `https://www.kaggle.com/datasets/abdullah0a/student-dropout-analysis-and-prediction-dataset`
