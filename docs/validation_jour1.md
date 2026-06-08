# Validation Jour 1

## Checklist livrables

- Docker Compose présent : `docker-compose.yml`
- Services définis : PostgreSQL, MLflow, FastAPI, Streamlit, Prometheus, Grafana
- API de santé : `GET /health`
- Dashboard de validation : Streamlit
- Base PostgreSQL initialisée : schéma `lms`
- Documentation architecture : `docs/architecture.md`
- Backlog Agile : `docs/backlog.md`

## Commandes de validation

```powershell
Copy-Item .env.example .env
docker compose up --build
```

## URLs de test

- `http://localhost:8000/health`
- `http://localhost:8000/docs`
- `http://localhost:8501`
- `http://localhost:5000`
- `http://localhost:9090`
- `http://localhost:3000`

## Résultat attendu

L'endpoint `/health` doit retourner une réponse proche de :

```json
{
  "api": "healthy",
  "database": "connected",
  "mlflow": "connected",
  "timestamp": "..."
}
```
