from __future__ import annotations

import os
import time
from pathlib import Path


class FileLock:
    def __init__(self, lock_path: Path, timeout_seconds: int = 30, poll_seconds: float = 0.1) -> None:
        self.lock_path = Path(lock_path)
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self._fd: int | None = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.timeout_seconds
        while True:
            try:
                self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self._fd, str(os.getpid()).encode("ascii"))
                return
            except FileExistsError:
                if time.time() >= deadline:
                    raise TimeoutError(f"Timed out waiting for lock: {self.lock_path}")
                time.sleep(self.poll_seconds)

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self.lock_path.exists():
            self.lock_path.unlink()
