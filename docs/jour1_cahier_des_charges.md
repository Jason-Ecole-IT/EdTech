# Jour 1 - Cahier des charges synthétique

## Objectif projet

Construire un système intelligent d'analytics pédagogique capable d'analyser les données LMS, d'identifier les étudiants à risque de décrochage et de proposer des indicateurs actionnables aux étudiants, enseignants et administrateurs.

## Utilisateurs cibles

- Étudiant : suivre sa progression, recevoir des recommandations et comprendre ses points faibles.
- Enseignant : détecter les étudiants à risque, suivre l'engagement et adapter l'accompagnement.
- Administrateur : piloter la performance globale des cohortes et surveiller les indicateurs clés.

## Fonctionnalités cibles

- Ingestion de données LMS.
- Nettoyage et feature engineering.
- Dashboard pédagogique multi-rôles.
- Modèle de prédiction du risque de décrochage.
- Segmentation des profils apprenants.
- Suivi MLOps avec MLflow.
- Monitoring technique et qualité.

## Contraintes

- Ressources modestes : CPU uniquement, 8-12GB RAM.
- Stack reproductible avec Docker Compose.
- Données structurées stockées dans PostgreSQL.
- Interface démontrable en local.

## Définition de succès Jour 1

- Les services démarrent avec Docker Compose.
- L'API expose `/health`.
- Le dashboard Streamlit affiche l'état des services.
- PostgreSQL est initialisé.
- MLflow est accessible.
- Le backlog et l'architecture sont documentés.
