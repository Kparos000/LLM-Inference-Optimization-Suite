# Product Platform Backend

Read-only FastAPI backend for the interactive AI Inference Engineering
Platform. It serves compact UI contracts from saved repository artifacts and
does not run inference.

Windows PowerShell:

```powershell
python -m uvicorn main:app --app-dir platform/backend --reload --port 8000
```

