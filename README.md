# 🎓 EdTech - Prédiction d'Abandon Scolaire

Projet de machine learning pour prédire le risque de décrochage scolaire des élèves à l'aide de l'IA.

## 📋 Description

Ce projet utilise un modèle RandomForest avec `class_weight='balanced'` pour prédire la probabilité d'abandon scolaire des étudiants. Le modèle a été entraîné sur des données historiques et est calibré pour fournir des probabilités significatives.

## 🏗️ Architecture

```text
Données brutes (CSV)
   ↓
Modèle RandomForest (class_weight='balanced')
   ↓
FastAPI (API de prédiction)
   ↓
Streamlit Dashboard (Interface utilisateur)
   ↓
MLflow (Tracking des modèles)
```

## 🚀 Démarrage rapide

### Prérequis

- Python 3.8+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Lancer l'application

Ouvre **3 terminaux** dans le répertoire du projet :

**Terminal 1 — MLflow**
```bash
mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns/artifacts
```

**Terminal 2 — FastAPI**
```bash
uvicorn src.api.main:app --port 8000 --reload
```

**Terminal 3 — Streamlit Dashboard**
```bash
streamlit run src/dashboard/app.py
```

### Accéder aux services

- **Dashboard Streamlit** : http://localhost:8501
- **API FastAPI** : http://localhost:8000
- **API Documentation (Swagger)** : http://localhost:8000/docs
- **MLflow UI** : http://localhost:5000

## 📊 Dashboard

Le dashboard Streamlit propose 2 onglets :

1. **Import par Lot** : Téléchargez un fichier CSV avec les données des élèves pour obtenir des prédictions en masse
2. **Élèves à Risque** : Visualisez et filtrez les élèves identifiés comme étant à risque d'abandon

### Format CSV attendu

Les colonnes requises pour le fichier CSV :
- School, Gender, Age, Address, Family_Size, Parental_Status
- Mother_Education, Father_Education, Mother_Job, Father_Job
- Reason_for_Choosing_School, Guardian, Travel_Time, Study_Time
- Number_of_Failures, School_Support, Family_Support, Extra_Paid_Class
- Extra_Curricular_Activities, Attended_Nursery, Wants_Higher_Education
- Internet_Access, In_Relationship, Family_Relationship, Free_Time, Going_Out
- Weekend_Alcohol_Consumption, Weekday_Alcohol_Consumption, Health_Status
- Number_of_Absences, Grade_1, Grade_2, Final_Grade, Dropped_Out

### Fichier de test

Un fichier de test est disponible : `test_students.csv` avec 12 élèves (10 dropout, 2 non-dropout).

## 🤖 Modèle ML

**Modèle actuel** : RandomForestClassifier
- **Paramètres** : n_estimators=200, max_depth=15, min_samples_split=5, class_weight='balanced'
- **Seuil de prédiction** : 0.5 (50%)
- **Calibration** : Bonne calibration grâce à class_weight='balanced'

**Critères les plus importants** :
1. Final_Grade (29.5%)
2. Grade_2 (13.7%)
3. Grade_1 (11.5%)
4. Number_of_Failures (3.3%)
5. School (2.8%)

## 🔧 Scripts utiles

- `train_simple.py` : Entraîner le modèle RandomForest avec class_weight='balanced'
- `verify_predictions.py` : Vérifier la précision des prédictions par rapport aux données réelles
- `create_test_file.py` : Créer un fichier de test avec des étudiants dropout et non-dropout

## 📈 Performance

Le modèle atteint une précision de **91.7%** sur le jeu de test (11/12 prédictions correctes).

## 📂 Structure du projet

```
PROJECT5/
├── data/
│   ├── raw/student dropout.csv          # Données brutes
│   └── test_students.csv                # Fichier de test
├── mlruns/                              # Artefacts MLflow
├── src/
│   ├── api/
│   │   └── predict.py                   # API de prédiction
│   └── dashboard/
│       └── app.py                       # Dashboard Streamlit
├── train_simple.py                      # Script d'entraînement
└── test_students.csv                    # Fichier de test
```

## 📚 Dataset

Source : [Kaggle Student Dropout Analysis](https://www.kaggle.com/datasets/abdullah0a/student-dropout-analysis-and-prediction-dataset)

## 🎯 Fonctionnalités

- ✅ Prédiction en lot via CSV
- ✅ Filtrage des élèves à risque par âge, école, sexe et probabilité
- ✅ Export des résultats en CSV
- ✅ Interface en français
- ✅ Modèle calibré avec probabilités significatives
