from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MetadataStore:
    """Read/write metadata for partition fingerprints and dependency diffs."""

    def __init__(self, meta_root: Path) -> None:
        self.meta_root = Path(meta_root)
        self.meta_root.mkdir(parents=True, exist_ok=True)
        self.dataset_versions_path = self.meta_root / "dataset_versions.json"

    def load_versions(self) -> dict[str, Any]:
        if not self.dataset_versions_path.exists():
            return {"datasets": {}}
        raw = self.dataset_versions_path.read_text(encoding="utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Preserve unreadable legacy file and recover with a clean metadata document.
            backup = self.dataset_versions_path.with_suffix(".json.bak")
            backup.write_text(raw, encoding="utf-8")
            return {"datasets": {}}

    def save_versions(self, versions: dict[str, Any]) -> None:
        self.dataset_versions_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_versions_path.write_text(
            json.dumps(versions, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def update_partition(
        self,
        dataset_name: str,
        partition_key: str,
        fingerprint: str,
        row_count: int,
        path: str,
    ) -> dict[str, Any]:
        versions = self.load_versions()
        datasets = versions.setdefault("datasets", {})
        dataset_entry = datasets.setdefault(dataset_name, {"partitions": {}})
        partitions = dataset_entry.setdefault("partitions", {})
        old = partitions.get(partition_key)
        ts = datetime.now(timezone.utc).isoformat()
        partitions[partition_key] = {
            "fingerprint": fingerprint,
            "row_count": int(row_count),
            "path": path,
            "last_write_ts": ts,
        }
        self.save_versions(versions)
        return {"old": old, "new": partitions[partition_key], "changed": (old or {}).get("fingerprint") != fingerprint}

    def partitions_with_changed_fingerprints(
        self,
        dataset_name: str,
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> list[str]:
        old_parts = previous.get("datasets", {}).get(dataset_name, {}).get("partitions", {})
        new_parts = current.get("datasets", {}).get(dataset_name, {}).get("partitions", {})
        changed: list[str] = []
        for key, info in new_parts.items():
            old_fp = old_parts.get(key, {}).get("fingerprint")
            if old_fp != info.get("fingerprint"):
                changed.append(key)
        return sorted(changed)
