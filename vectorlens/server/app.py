"""FastAPI application for VectorLens dashboard."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from vectorlens.session_bus import bus
from vectorlens.server.api import router as api_router

logger = logging.getLogger(__name__)

# Max request body size: 1MB — prevents memory exhaustion via large POST bodies
MAX_REQUEST_BODY = 1 * 1024 * 1024

# Allowed WebSocket origins — CORS does NOT apply to WebSockets, so we enforce manually
_ALLOWED_ORIGINS = {
    "http://127.0.0.1:7756",
    "http://localhost:7756",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
}


class RequestSizeLimitMiddleware:
    """Pure ASGI middleware that enforces MAX_REQUEST_BODY by wrapping receive().

    BaseHTTPMiddleware's approach of hacking _stream doesn't work in modern
    Starlette — downstream routes use ASGI's receive(), not _stream. This
    middleware intercepts at the ASGI layer instead, buffering the body and
    re-serving it via a replacement receive callable.

    Handles both Content-Length and Transfer-Encoding: chunked correctly.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        content_length = headers.get(b"content-length")

        # Fast path: honour Content-Length when present
        if content_length:
            if int(content_length) > MAX_REQUEST_BODY:
                await _send_413(send)
                return
            await self.app(scope, receive, send)
            return

        # Only buffer mutation methods with unknown body length
        if method not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return

        # Chunked / unknown: buffer body and check size
        body = b""
        more = True
        while more:
            message = await receive()
            body += message.get("body", b"")
            more = message.get("more_body", False)
            if len(body) > MAX_REQUEST_BODY:
                await _send_413(send)
                return

        # Replace receive with a callable that replays the buffered body
        async def buffered_receive() -> dict:
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, buffered_receive, send)


async def _send_413(send: Any) -> None:
    """Send a 413 Request Entity Too Large response at the ASGI level."""
    import json
    body = json.dumps({"detail": "Request body too large"}).encode()
    await send({
        "type": "http.response.start",
        "status": 413,
        "headers": [
            [b"content-type", b"application/json"],
            [b"content-length", str(len(body)).encode()],
        ],
    })
    await send({"type": "http.response.body", "body": body})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _server_event_loop
    _server_event_loop = asyncio.get_running_loop()
    _setup_bus_subscriptions()
    logger.info("VectorLens server started")
    yield
    _server_event_loop = None
    logger.info("VectorLens server stopped")


app = FastAPI(
    title="VectorLens",
    description="RAG debugging tool with real-time tracing",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestSizeLimitMiddleware)

# CORS middleware — localhost only (prevents cross-origin data exfiltration)
# "*" with allow_credentials=True would let any webpage steal session data
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:7756",
        "http://localhost:7756",
        "http://127.0.0.1:5173",  # Vite dev server
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

# WebSocket state
_connected_websockets: set[WebSocket] = set()
_ws_lock = __import__("threading").Lock()


async def _broadcast_event(event_type: str, event: Any) -> None:
    """Broadcast an event to all connected WebSocket clients."""
    try:
        event_json = _serialize_event(event_type, event)
    except Exception as e:
        logger.error(f"Failed to serialize event: {e}")
        return

    disconnected = set()
    with _ws_lock:
        for ws in _connected_websockets:
            try:
                await ws.send_json(event_json)
            except Exception as e:
                logger.debug(f"Failed to send event to WebSocket: {e}")
                disconnected.add(ws)

    # Clean up disconnected clients
    with _ws_lock:
        for ws in disconnected:
            _connected_websockets.discard(ws)


def _serialize_event(event_type: str, event: Any) -> dict[str, Any]:
    """Serialize an event object to JSON-serializable dict."""
    from dataclasses import asdict, is_dataclass
    import time

    return {
        "type": event_type,
        "data": asdict(event) if is_dataclass(event) else event,
        "timestamp": time.time(),
    }


def _make_event_handler(event_type: str) -> Any:
    """Create a thread-safe event handler that broadcasts to WebSockets."""
    def handler(event: Any) -> None:
        """Handle event from bus and schedule broadcast on the server event loop."""
        loop = _server_event_loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(_broadcast_event(event_type, event), loop)
        else:
            logger.debug(f"No running event loop for {event_type}, skipping broadcast")
    return handler


# Reference to the uvicorn event loop — set during lifespan startup
_server_event_loop: asyncio.AbstractEventLoop | None = None


# Subscribe to all event types from the bus
def _setup_bus_subscriptions() -> None:
    """Set up subscriptions to bus events."""
    for event_type in ["vector_query", "llm_request", "llm_response", "attribution"]:
        bus.subscribe(event_type, _make_event_handler(event_type))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time event streaming.

    CORS does not apply to WebSockets — browsers allow cross-origin WS
    connections silently. We must validate the Origin header ourselves
    to prevent Cross-Site WebSocket Hijacking (CSWSH).
    """
    origin = websocket.headers.get("origin", "")
    if origin and origin not in _ALLOWED_ORIGINS:
        logger.warning(f"WebSocket rejected: disallowed origin '{origin}'")
        await websocket.close(code=1008)  # 1008 = Policy Violation
        return

    await websocket.accept()
    with _ws_lock:
        _connected_websockets.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Received from client: {data}")
    except WebSocketDisconnect:
        logger.debug("Client disconnected from WebSocket")
    except Exception as e:
        # Break the loop on any unexpected error — prevents infinite CPU spin
        # in half-closed socket states that never raise WebSocketDisconnect.
        logger.debug(f"WebSocket closed unexpectedly: {e}")
    finally:
        with _ws_lock:
            _connected_websockets.discard(websocket)


# Mount API router
app.include_router(api_router, prefix="/api")


# Serve static dashboard or fallback
def _get_dashboard_path() -> Path | None:
    """Find dashboard dist folder."""
    candidates = [
        Path(__file__).parent.parent.parent / "dashboard" / "dist",
        Path(__file__).parent.parent.parent / "dist",
    ]
    for p in candidates:
        if p.exists() and (p / "index.html").exists():
            return p
    return None


_dashboard_path = _get_dashboard_path()

if _dashboard_path:
    app.mount("/", StaticFiles(directory=str(_dashboard_path), html=True), name="dashboard")
else:
    # Fallback HTML
    @app.get("/", response_class=HTMLResponse)
    async def serve_dashboard() -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>VectorLens</title>
            <style>
                body { font-family: sans-serif; padding: 2rem; }
                h1 { color: #333; }
                p { color: #666; margin-bottom: 1rem; }
                code { background: #f4f4f4; padding: 0.2rem 0.4rem; border-radius: 3px; }
                .status { background: #e8f5e9; padding: 1rem; border-radius: 4px; margin-top: 1rem; }
            </style>
        </head>
        <body>
            <h1>VectorLens</h1>
            <p>Dashboard is loading...</p>
            <p>Build the React dashboard and place it in <code>dashboard/dist/</code> to see the UI.</p>
            <div class="status">
                <strong>Server Status:</strong> Running
            </div>
            <p>API available at <code>/api</code> | WebSocket at <code>/ws</code></p>
        </body>
        </html>
        """


