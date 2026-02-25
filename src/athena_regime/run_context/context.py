# src/athena_regime/run_context/context.py
from __future__ import annotations
import json, logging, secrets, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from src.athena_regime.config.schema import AppConfig


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class RunContext:
    """
    Created once per CLI invocation. Carries run_id, logger, and artifact
    paths. Passed explicitly into engines; never accessed via global state.
    """

    def __init__(self, config: AppConfig, run_id: str | None = None) -> None:
        self.config = config
        self.run_id = run_id or self._new_run_id()
        self.run_dir = config.run.runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "artifacts").mkdir(exist_ok=True)
        self.logger = self._build_logger(config.run.log_level)

    # ── Public API ─────────────────────────────────────────────────────────

    def log_config(self, cfg_dict: dict) -> None:
        self._write_json("config.json", cfg_dict)
        self.logger.info("run_id=%s config logged", self.run_id)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        self._write_json("metrics.json", metrics)
        self.logger.info("metrics logged: %s", list(metrics.keys()))

    def artifact(self, name: str) -> Path:
        """Return path for a named artifact inside this run's artifacts/ dir."""
        return self.run_dir / "artifacts" / name

    # ── Private ────────────────────────────────────────────────────────────

    @staticmethod
    def _new_run_id() -> str:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"{ts}_{secrets.token_hex(4)}"

    def _build_logger(self, level: str) -> logging.Logger:
        logger = logging.getLogger(f"athena.{self.run_id}")
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False  # don't leak to root logger

        # Structured JSON -> file
        fh = logging.FileHandler(self.run_dir / "logs.log", encoding="utf-8")
        fh.setFormatter(_JsonFormatter())
        fh.setLevel(logging.DEBUG)

        # Human-readable -> stdout
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _write_json(self, name: str, data: dict) -> None:
        with open(self.run_dir / name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
