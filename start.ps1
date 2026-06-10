Write-Host "🎓 EdTech - Lancement des services..." -ForegroundColor Cyan

$root = $PSScriptRoot

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; Write-Host '🚀 FastAPI' -ForegroundColor Green; uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"

Start-Sleep -Seconds 2

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; Write-Host '📊 MLflow' -ForegroundColor Yellow; mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns/artifacts"

Start-Sleep -Seconds 2

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; Write-Host '🎓 Streamlit' -ForegroundColor Magenta; streamlit run src/dashboard/app.py --server.port 8501"

Write-Host ""
Write-Host "✅ Services en cours de démarrage..." -ForegroundColor Green
Write-Host ""
Write-Host "  API     → http://localhost:8000" -ForegroundColor White
Write-Host "  Swagger → http://localhost:8000/docs" -ForegroundColor White
Write-Host "  MLflow  → http://localhost:5000" -ForegroundColor White
Write-Host "  Dashboard → http://localhost:8501" -ForegroundColor White
Write-Host ""
Write-Host "Attente ouverture navigateur (5s)..." -ForegroundColor Gray
Start-Sleep -Seconds 5

Start-Process "http://localhost:8501"
