from __future__ import annotations
import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure project-wide logging. Call once at startup."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "crewai", "litellm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
