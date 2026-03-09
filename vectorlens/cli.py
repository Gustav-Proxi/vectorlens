"""Command-line interface for VectorLens."""
from __future__ import annotations

import argparse
import logging
import signal
import sys

import vectorlens

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="vectorlens",
        description="VectorLens — RAG debugging tool with real-time tracing",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # serve subcommand
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the VectorLens dashboard server",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=7756,
        help="Server port (default: 7756)",
    )
    serve_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open browser automatically",
    )
    serve_parser.add_argument(
        "--no-intercept",
        action="store_true",
        help="Do not auto-install interceptors",
    )

    args = parser.parse_args()

    if args.command == "serve":
        # Start the server
        try:
            url = vectorlens.serve(
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
                auto_intercept=not args.no_intercept,
            )
            logger.info(f"Server started at {url}")
            logger.info("Press Ctrl+C to stop")

            # Block until interrupt
            def signal_handler(signum: int, frame: object) -> None:
                logger.info("Shutting down...")
                vectorlens.stop()
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.pause()  # On Unix; on Windows this will not work
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            vectorlens.stop()
            sys.exit(0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
