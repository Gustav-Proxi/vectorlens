"""VectorLens — See why your RAG hallucinates.

Usage:
    import vectorlens
    vectorlens.serve()          # start dashboard + install interceptors
    vectorlens.new_session()    # start a fresh tracing session
    vectorlens.stop()           # stop dashboard
"""
from __future__ import annotations

import logging
import threading
import time
import webbrowser
from typing import Any

import uvicorn

from vectorlens.cag import CAGSession
from vectorlens.interceptors import install_all, uninstall_all
from vectorlens.pipeline import setup_auto_attribution
from vectorlens.server.app import app
from vectorlens.session_bus import bus

logger = logging.getLogger(__name__)

# Global server state
_server_thread: threading.Thread | None = None
_server_instance: uvicorn.Server | None = None
_server_lock = threading.Lock()


def serve(
    host: str = "127.0.0.1",
    port: int = 7756,
    open_browser: bool = True,
    auto_intercept: bool = True,
) -> str:
    """
    Start the VectorLens dashboard server in a background thread.
    Install all interceptors.
    Open browser to dashboard URL.

    Args:
        host: Server host (default: 127.0.0.1)
        port: Server port (default: 7756)
        open_browser: Whether to open browser automatically (default: True)
        auto_intercept: Whether to auto-install interceptors (default: True)

    Returns:
        The dashboard URL.
    """
    global _server_thread, _server_instance

    with _server_lock:
        if _server_thread is not None and _server_thread.is_alive():
            logger.warning("Server is already running")
            return get_session_url(None)

        # Install interceptors if requested
        if auto_intercept:
            installed = install_all()
            logger.info(f"Installed interceptors: {installed}")

        # Wire up auto-attribution pipeline
        setup_auto_attribution()

        # Create uvicorn config
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
            access_log=False,
        )
        _server_instance = uvicorn.Server(config)

        # Run server in background thread
        def run_server() -> None:
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_server_instance.serve())

        _server_thread = threading.Thread(target=run_server, daemon=True)
        _server_thread.start()

        # Wait for server to start
        time.sleep(0.5)

        # Open browser
        dashboard_url = f"http://{host}:{port}"
        if open_browser:
            threading.Thread(
                target=lambda: webbrowser.open(dashboard_url),
                daemon=True,
            ).start()

        logger.info(f"VectorLens server running at {dashboard_url}")
        return dashboard_url


def stop() -> None:
    """Stop the dashboard server and uninstall interceptors."""
    global _server_thread, _server_instance

    with _server_lock:
        if _server_instance is not None:
            logger.info("Stopping VectorLens server...")
            _server_instance.should_exit = True

            # Wait for thread to finish (with timeout)
            if _server_thread is not None:
                _server_thread.join(timeout=5)
            _server_thread = None
            _server_instance = None

        # Uninstall interceptors
        uninstall_all()
        logger.info("Uninstalled interceptors")


def new_session() -> str:
    """
    Start a fresh tracing session.

    Returns:
        Session ID.
    """
    session = bus.new_session()
    logger.info(f"Created new session: {session.id}")
    return session.id


def cag_session(documents: list) -> CAGSession:
    """Context manager for Cache-Augmented Generation (CAG) sessions.

    Register the full document corpus that was loaded into the LLM context.
    VectorLens will attribute hallucinations to specific documents using
    cosine similarity — no retrieval step needed.

    Args:
        documents: List of documents. Each can be:
            - str: raw text
            - dict with 'text' key (and optional 'id', 'title')

    Example:
        docs = [
            {"id": "report", "title": "Q4 Report", "text": "Revenue was $5M..."},
            "Plain text document also works.",
        ]

        with vectorlens.cag_session(docs):
            prompt = "\\n\\n".join(d["text"] for d in docs) + "\\n\\nQ: " + query
            response = client.chat.completions.create(...)
    """
    return CAGSession(documents)


def get_session_url(session_id: str | None = None) -> str:
    """
    Return URL to view a specific session in dashboard.

    Args:
        session_id: Session ID (if None, returns base dashboard URL)

    Returns:
        Dashboard URL for the session.
    """
    # This assumes the server is running on localhost:7756
    # In production, this would use the actual server host/port from config
    base_url = "http://127.0.0.1:7756"
    if session_id:
        return f"{base_url}/sessions/{session_id}"
    return base_url


__all__ = [
    "serve",
    "stop",
    "new_session",
    "get_session_url",
    "cag_session",
]
