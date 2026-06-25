"""Main entry point for running the EHR Copilot API."""

import uvicorn

from ehr_copilot.api.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("ehr_copilot.main:app", host="0.0.0.0", port=8000, reload=True)
