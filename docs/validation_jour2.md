# Jour 2 - Data Pipeline & Processing

## Données source

- Fichier : `data/raw/student dropout.csv`
- 649 étudiants, 34 colonnes
- Cible : `Dropped_Out` (False=549 / True=100)
- Taux de décrochage : 15.4%
- Aucune valeur manquante dans les données brutes

## Architecture du pipeline

```text
data/raw/student dropout.csv
    ↓
src/data/loader.py        Chargement + nettoyage ETL
    ↓
src/data/features.py      Feature engineering (10 nouvelles features)
    ↓
src/data/eda.py           EDA + graphiques + rapport JSON
    ↓
src/data/loader.py        Encodage one-hot catégorielles
src/data/validation.py    9 tests qualité automatisés
    ↓
src/data/ml_dataset.py    Standardisation + split train/test
    ↓
data/processed/           Données nettoyées + rapports
data/features/            Datasets ML-ready train/test
```

## Lancer le pipeline

```powershell
python -m src.data.pipeline
```

## Lancer les tests

```powershell
pytest tests/test_pipeline.py -v
```

## Features créées

| Feature | Type | Justification |
| --- | --- | --- |
| `academic_performance_score` | Continu | Moyenne pondérée G1×0.25 + G2×0.35 + Final×0.40 |
| `grade_progression` | Continu | Évolution G1→G2, signal de dynamique scolaire |
| `alcohol_risk_score` | Continu | Consommation pondérée semaine×0.6 + weekend×0.4 |
| `social_risk_index` | Continu [0,1] | Indice sorties + alcool + temps libre non structuré |
| `family_support_score` | Continu | Relations familiales + soutien école + famille |
| `engagement_score` | Continu | Temps étude + aspiration + internet + activités extra |
| `parental_education_level` | Continu | Moyenne éducation père/mère, proxy capital culturel |
| `absence_risk_flag` | Binaire | 1 si absences ≥ 10 (seuil critique littérature) |
| `failure_history_flag` | Binaire | 1 si au moins un échec antérieur |
| `combined_risk_score` | Continu [0,1] | Score de risque global agrégé (monitoring pédagogique) |

## Fichiers produits

| Fichier | Description |
| --- | --- |
| `data/processed/students_clean.parquet` | Données nettoyées et encodées |
| `data/features/students_features.parquet` | Dataset avec toutes les features |
| `data/features/train.parquet` | Split entraînement avec target |
| `data/features/test.parquet` | Split test avec target |
| `data/features/X_train.parquet` | Features entraînement |
| `data/features/X_test.parquet` | Features test |
| `data/features/y_train.parquet` | Labels entraînement |
| `data/features/y_test.parquet` | Labels test |
| `data/processed/eda_report.json` | Statistiques et insights EDA |
| `data/processed/quality_report.json` | Rapport 9 tests de qualité |
| `data/processed/eda_plots/` | Graphiques PNG |
| `data/processed/pipeline.log` | Logs d'exécution |

## Principaux insights EDA

- Taux de décrochage : 15.4%
- Les étudiants décrochés ont un `Final_Grade` moyen nettement inférieur aux actifs
- Le `Number_of_Failures` est le prédicteur le plus fort
- Les absences ≥ 10 multiplient significativement le risque de décrochage
- Le `combined_risk_score` est positivement corrélé avec la cible

## Prochaine étape

Jour 3 : entraînement des modèles ML (baseline + XGBoost), tracking MLflow, sélection et versionnement.
