# Cahier des charges - Projet EdTech Analytics Pédagogique

## 1. Contexte

Les plateformes LMS génèrent de nombreuses données sur l'activité des étudiants : connexions, progression, quiz, devoirs, notes, temps passé, retards et interactions. Ces données sont souvent sous-exploitées alors qu'elles peuvent aider les établissements à mieux accompagner les apprenants.

Ce projet vise à construire une application intelligente de learning analytics permettant d'analyser les comportements pédagogiques, de détecter les risques de décrochage et d'aider les enseignants à prendre de meilleures décisions.

## 2. Objectif général

Développer une solution EdTech capable de :

- centraliser et transformer des données LMS ;
- visualiser les indicateurs pédagogiques clés ;
- prédire le risque de décrochage étudiant ;
- segmenter les profils d'apprenants ;
- proposer une base pour la personnalisation des parcours ;
- suivre les modèles avec une démarche MLOps ;
- présenter une application démontrable, documentée et monitorée.

## 3. Objectifs métier

- Améliorer le suivi pédagogique des étudiants.
- Identifier rapidement les étudiants à risque.
- Aider les enseignants à prioriser leurs actions.
- Donner aux administrateurs une vision globale des cohortes.
- Valoriser l'usage de l'intelligence artificielle dans un contexte socialement utile.

## 4. Utilisateurs cibles

### Étudiant

- Consulter sa progression.
- Comprendre son niveau d'engagement.
- Identifier ses points faibles.
- Recevoir des recommandations simples.

### Enseignant

- Visualiser les étudiants à risque.
- Suivre l'activité par cours ou par cohorte.
- Identifier les étudiants peu engagés.
- Adapter l'accompagnement pédagogique.

### Administrateur

- Suivre les indicateurs globaux.
- Comparer les cohortes.
- Observer les tendances de décrochage.
- Contrôler la qualité et la disponibilité de l'application.

## 5. Périmètre fonctionnel

### Ingestion et traitement des données

- Charger des données LMS simulées ou réelles.
- Nettoyer les données.
- Contrôler la qualité des données.
- Construire des features utiles pour le Machine Learning.
- Stocker les données dans PostgreSQL.

### Analyse exploratoire

- Étudier la distribution des notes.
- Analyser l'engagement des étudiants.
- Observer les connexions, retards, absences et interactions.
- Identifier les corrélations avec le décrochage.

### Machine Learning

- Construire un modèle baseline.
- Entraîner un modèle de classification du risque de décrochage.
- Comparer plusieurs modèles.
- Suivre les expériences avec MLflow.
- Enregistrer le meilleur modèle.

### Dashboard

- Afficher des KPIs pédagogiques.
- Visualiser les étudiants à risque.
- Présenter les tendances d'engagement.
- Afficher des graphiques interactifs.
- Préparer une interface multi-rôles.

### API

- Exposer une API FastAPI.
- Fournir un endpoint de santé.
- Fournir un endpoint de prédiction.
- Connecter le dashboard à l'API.

### Monitoring

- Préparer Prometheus pour la collecte de métriques.
- Préparer Grafana pour les dashboards techniques.
- Prévoir un suivi de la qualité modèle et des données.

## 6. Périmètre non fonctionnel

- Application exécutable en local avec Docker Compose.
- Projet compatible avec une machine sans GPU.
- Code organisé par modules.
- Documentation claire pour installation, lancement et démonstration.
- Séparation entre données brutes, données transformées et features ML.
- Respect des bonnes pratiques de sécurité : pas de secret dans le code.

## 7. Technologies prévues

| Domaine | Technologies |
|---|---|
| Data processing | Python, Pandas |
| Base de données | PostgreSQL |
| Machine Learning | scikit-learn, XGBoost |
| Clustering | K-Means |
| Tracking ML | MLflow |
| API | FastAPI |
| Dashboard | Streamlit, Plotly |
| Monitoring | Prometheus, Grafana |
| Qualité données / drift | Evidently AI |
| Conteneurisation | Docker, Docker Compose |

## 8. Architecture cible

```text
Données LMS
   ↓
Ingestion / nettoyage
   ↓
PostgreSQL
   ↓
Feature engineering
   ↓
Dataset ML-ready
   ↓
Training ML + MLflow
   ↓
Model Registry
   ↓
FastAPI
   ↓
Dashboard Streamlit
   ↓
Monitoring Prometheus / Grafana
```

## 9. Données attendues

Les données LMS pourront contenir :

- identifiant étudiant ;
- cours suivi ;
- nombre de connexions ;
- temps passé sur la plateforme ;
- progression dans les modules ;
- notes aux quiz ;
- devoirs rendus ;
- retards ;
- absences ;
- interactions forum ;
- statut final : actif, à risque ou décroché.

## 10. Indicateurs clés

- Taux de complétion des cours.
- Score moyen par étudiant.
- Nombre moyen de connexions.
- Temps moyen passé sur le LMS.
- Taux de retard dans les devoirs.
- Pourcentage d'étudiants à risque.
- Distribution des profils d'apprenants.
- Performance du modèle : accuracy, recall, precision, F1-score, ROC-AUC.

## 11. Contraintes

- Ressources : 4-5 cores CPU, 8-12GB RAM, 20GB stockage.
- GPU non nécessaire.
- Exécution locale prioritaire.
- Données sensibles à anonymiser si elles sont réelles.
- Résultats ML à expliquer avec prudence pour éviter les biais.

## 12. Exigences éthiques

- Ne pas utiliser le modèle pour sanctionner automatiquement un étudiant.
- Présenter le score de risque comme une aide à la décision.
- Surveiller les biais possibles.
- Favoriser l'accompagnement pédagogique.
- Documenter les limites du modèle.

## 13. Planning prévisionnel

### Jour 1 - Infrastructure & Découverte

- Analyse du besoin.
- Conception architecture.
- Backlog Agile.
- Setup Docker Compose.
- Services de base communicants.

### Jour 2 - Data Pipeline & Processing

- Ingestion des données.
- Feature engineering.
- Nettoyage et validation.
- Dataset ML-ready.

### Jour 3 - Machine Learning & MLOps

- Modèle baseline.
- Modèle avancé.
- Tracking MLflow.
- Sélection et versionnement du meilleur modèle.

### Jour 4 - Deployment & API

- API FastAPI.
- Endpoints d'inférence.
- Dashboard Streamlit.
- Connexion frontend-backend.

### Jour 5 - Monitoring, Tests & Présentation

- Monitoring Prometheus/Grafana.
- Tests end-to-end.
- Documentation finale.
- Préparation démo et présentation.

## 14. Livrables finaux attendus

- Application Dockerisée.
- Pipeline de données fonctionnel.
- Dataset ML-ready.
- Modèle entraîné et suivi avec MLflow.
- API REST fonctionnelle.
- Dashboard interactif.
- Monitoring technique.
- Documentation projet.
- Présentation professionnelle de 15 minutes.

## 15. Critères de réussite

- Le projet se lance avec `docker compose up --build`.
- Les services principaux sont accessibles localement.
- Le pipeline produit un dataset exploitable.
- Le modèle donne des performances mesurables.
- Le dashboard permet de comprendre les indicateurs clés.
- Le projet est documenté et reproductible.
