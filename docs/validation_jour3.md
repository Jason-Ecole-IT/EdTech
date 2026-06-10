# Jour 3 - Machine Learning & MLOps

## Objectifs

- 6 modèles entraînés et comparés via MLflow tracking
- Meilleur modèle sélectionné avec justification métrique
- Tests de validation complets
- Modèle enregistré dans MLflow Registry (stage Production)

## Lancer le pipeline ML

```powershell
python -m src.ml.pipeline
```

## Lancer les tests ML

```powershell
python -m pytest tests/test_ml.py -v
```

## Modèles entraînés

| # | Modèle | Description |
| --- | --- | --- |
| 1 | `baseline_majority` | DummyClassifier - prédit toujours la classe majoritaire |
| 2 | `logistic_regression` | Régression logistique avec class_weight=balanced |
| 3 | `random_forest_v1` | Random Forest 100 arbres, max_depth=10 |
| 4 | `random_forest_v2` | Random Forest 200 arbres, max_depth=15 (optimisé) |
| 5 | `gradient_boosting` | GradientBoostingClassifier sklearn |
| 6 | `xgboost` | XGBoost 200 estimateurs, learning_rate=0.1 |

## Métriques de sélection

### Métrique primaire : F1-score (pondéré)

Justification : le dataset est déséquilibré (85% actifs, 15% décrochés).
L'accuracy seule serait trompeuse — prédire toujours "actif" donnerait 85% d'accuracy.
Le F1 équilibre précision et rappel, adapté à la détection de cas minoritaires.

### Score composite (pondéré)

| Métrique | Poids |
| --- | --- |
| F1-score | 40% |
| ROC-AUC | 30% |
| Accuracy | 20% |
| Recall | 10% |

## Tests de validation

| Test | Description |
| --- | --- |
| Prédictions sans NaN/Inf | Sanité des sorties |
| Reproductibilité | 2 appels = résultats identiques |
| Classes de sortie {0,1} | Contrainte domaine |
| Prédiction échantillon unique | Robustesse |
| Taille sortie = taille entrée | Cohérence dimensionnelle |
| Probabilités dans [0,1] | Calibration |
| Sensibilité aux features | Le modèle réagit aux entrées |

## MLflow Registry

- Nom : `edtech_dropout_classifier`
- Stage : `Production`
- URI de consommation : `models:/edtech_dropout_classifier/Production`

## Fichiers produits

| Fichier | Description |
| --- | --- |
| `data/processed/model_comparison.csv` | Tableau comparatif des modèles |
| `data/processed/model_selection_justification.txt` | Justification sélection |
| `data/processed/model_validation_report.json` | Rapport tests validation |
| `data/processed/jour3_summary.json` | Résumé complet Jour 3 |
| `data/processed/pipeline_ml.log` | Logs d'exécution |
| `models/registry_info.json` | Infos modèle enregistré |
| `models/*_feature_importance.json` | Importance des features par modèle |

## Prochaine étape

Jour 4 : API FastAPI enrichie avec endpoint de prédiction consommant le modèle Production depuis MLflow Registry.
