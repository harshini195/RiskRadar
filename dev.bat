@echo off
echo Starting RiskRadar Services...

:: 1. Start Flask Backend
echo Starting Flask Backend...
start cmd /k "call venv\Scripts\activate && cd backend && python app.py"

:: 2. Start Frontend
echo Starting Frontend Dev Server...
start cmd /k "cd frontend && npm run dev"

echo All services launched!