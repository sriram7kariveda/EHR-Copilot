"""FastAPI middleware for request tracing, timing, and patient-scope enforcement."""

from __future__ import annotations

import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique ``X-Request-ID`` header into every request and response.

    If the incoming request already carries an ``X-Request-ID`` header the
    value is preserved; otherwise a new UUID4 is generated.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Make available to downstream handlers via request state.
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Measure wall-clock request processing time and add it as a header.

    The ``X-Response-Time-Ms`` response header contains the elapsed time in
    milliseconds (two decimal places).
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response


class PatientScopeMiddleware(BaseHTTPMiddleware):
    """Enforce that query requests reference a loaded patient.

    For ``POST /query`` requests, the middleware peeks at the request body to
    extract the ``patient_id`` field and verifies that the patient has been
    loaded into the index registry stored on ``app.state.index_registry``.

    If the patient is not loaded, the middleware short-circuits with a
    ``422 Unprocessable Entity`` response before the route handler runs.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method == "POST" and request.url.path.rstrip("/") == "/query":
            # Read the raw body and cache it so downstream handlers can still
            # consume it.  Starlette allows body re-reads when stored in state.
            body_bytes = await request.body()
            try:
                body = json.loads(body_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return Response(
                    content=json.dumps({"detail": "Invalid JSON body"}),
                    status_code=400,
                    media_type="application/json",
                )

            patient_id = body.get("patient_id")
            if not patient_id:
                return Response(
                    content=json.dumps({"detail": "patient_id is required"}),
                    status_code=422,
                    media_type="application/json",
                )

            index_registry = getattr(request.app.state, "index_registry", None)
            if index_registry is not None:
                loaded_patients = index_registry.list_patients()
                if patient_id not in loaded_patients:
                    return Response(
                        content=json.dumps(
                            {
                                "detail": (
                                    f"Patient '{patient_id}' is not loaded. "
                                    f"Loaded patients: {loaded_patients}"
                                )
                            }
                        ),
                        status_code=422,
                        media_type="application/json",
                    )

        return await call_next(request)
