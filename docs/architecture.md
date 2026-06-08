# Architecture projet

## Vue d'ensemble

Le projet est organisé autour d'une architecture modulaire simple à faire évoluer pendant les cinq jours.

```text
data/
  raw/          Données brutes LMS
  processed/    Données nettoyées
  features/     Datasets prêts pour le ML

src/
  api/          FastAPI
  dashboard/    Streamlit
  data/         Ingestion et processing
  ml/           Entraînement, évaluation, registry
  monitoring/   Rapports et surveillance

infra/
  postgres/     Initialisation base de données
  prometheus/   Configuration monitoring
```

## Services Docker

| Service | Port | Rôle |
| --- | ---: | --- |
| PostgreSQL | 5432 | Stockage données LMS |
| MLflow | 5000 | Tracking expériences ML |
| FastAPI | 8000 | API backend et inférence |
| Streamlit | 8501 | Dashboard interactif |
| Prometheus | 9090 | Collecte métriques |
| Grafana | 3000 | Visualisation monitoring |

## Flux cible

1. Les données LMS sont placées dans `data/raw`.
2. Le pipeline transforme les données vers `data/processed`.
3. Les features ML sont écrites dans `data/features`.
4. Les modèles sont entraînés et suivis dans MLflow.
5. L'API expose les prédictions.
6. Streamlit consomme l'API pour afficher les dashboards.
