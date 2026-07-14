# Product Platform Backend

Read-only FastAPI backend for the interactive AI Inference Engineering
Platform. It serves compact UI contracts from saved repository artifacts and
does not run inference.

Windows PowerShell from `platform/frontend/`:

```powershell
npm run backend
```

Equivalent command from the repository root:

```powershell
python -m uvicorn main:app --app-dir platform/backend --reload --reload-dir . --host 127.0.0.1 --port 8011
```
