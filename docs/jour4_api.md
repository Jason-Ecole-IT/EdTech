# Jour 4 - API Prediction depuis MLflow Registry

## Objectif
Ajouter des endpoints de prediction a l'API FastAPI pour predire le decrochage des etudiants en utilisant le modele enregistre dans MLflow Registry.

## Contenu

### 1. Module de prediction (`src/api/predict.py`)
- **StudentFeatures** : modele Pydantic pour les 31 features d'un etudiant
- **PredictionResponse** : modele de reponse (prediction, probabilite, confiance)
- **load_model_from_registry()** : charge le modele depuis MLflow Registry (stage Production)
- **predict_single()** : prediction pour un etudiant
- **predict_batch()** : prediction pour plusieurs etudiants
- **build_feature_dict()** : construit les 61 features (incluant features engineered)

### 2. Endpoints API (`src/api/main.py`)
- **POST /predict** : prediction pour un etudiant
  - Body : JSON avec StudentFeatures
  - Response : PredictionResponse
- **POST /batch_predict** : prediction par lot
  - Body : tableau de StudentFeatures
  - Response : tableau de PredictionResponse

### 3. Dashboard (`src/dashboard/app.py`)
- Formulaire interactif avec les champs principaux
- Bouton "Predire" qui appelle l'API
- Affichage du resultat (Actif/Decroche, probabilite, confiance)

### 4. Tests (`tests/test_api.py`)
- Tests des endpoints /predict et /batch_predict
- Tests de validation des donnees
- Tests du module predict

## Utilisation

### Lancer les services

```bash
# Terminal 1 - MLflow
mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db

# Terminal 2 - FastAPI
uvicorn src.api.main:app --port 8000 --reload

# Terminal 3 - Streamlit Dashboard
streamlit run src/dashboard/app.py
```

### Tester l'API via curl

```bash
# Prediction simple
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 18,
    "travel_time": 2,
    "study_time": 3,
    "failures": 0,
    "family_rel": 4,
    "free_time": 3,
    "going_out": 3,
    "weekday_alcohol": 1,
    "weekend_alcohol": 2,
    "health": 4,
    "absences": 2,
    "grade_1": 14,
    "grade_2": 15,
    "final_grade": 16,
    "mother_edu": 2,
    "father_edu": 2,
    "mother_job": "services",
    "father_job": "other",
    "school": "GP",
    "sex": "M",
    "address": "U",
    "famsize": "GT3",
    "pstatus": "T",
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no"
  }'
```

### Lancer les tests

```bash
pytest tests/test_api.py -v
```

## Resultats

- API FastAPI avec endpoints prediction operationnels
- Integration avec MLflow Registry (modele en Production)
- Dashboard interactif pour les predictions
- Tests API passants

## Architecture

```
Dashboard Streamlit (8501)
        ↓
    FastAPI (8000)
        ↓
    MLflow Registry (5000)
        ↓
    Modele GradientBoosting
```

## Prochaine etape

Jour 5 : Monitoring avec Prometheus + Grafana
