"""Application-wide request security and observability controls."""

from __future__ import annotations

import re
import time
import uuid

from flask import Flask, Response, g, request


def configure_request_controls(app: Flask) -> None:
    """Add request identifiers, access logs, and secure response headers."""

    @app.before_request
    def begin_request() -> None:
        candidate = request.headers.get("X-Request-ID", "")
        g.request_id = candidate if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", candidate) else str(uuid.uuid4())
        g.request_started_at = time.perf_counter()

    @app.after_request
    def secure_and_log(response: Response) -> Response:
        response.headers["X-Request-ID"] = g.get("request_id", "unknown")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; "
            "style-src 'self' https://cdn.jsdelivr.net; "
            "script-src 'self' https://cdn.jsdelivr.net; font-src 'self' https://cdn.jsdelivr.net; "
            "connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        )
        if request.endpoint != "static":
            response.headers.setdefault("Cache-Control", "no-store")
        elapsed_ms = (time.perf_counter() - g.get("request_started_at", time.perf_counter())) * 1000
        app.logger.info(
            "request_complete method=%s path=%s status=%s duration_ms=%.2f request_id=%s",
            request.method, request.path, response.status_code, elapsed_ms, g.get("request_id", "unknown"),
        )
        return response