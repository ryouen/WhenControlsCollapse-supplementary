"""
Phase config.json writer.

Every phase writes a config.json with execution metadata.
See spec/WCC_naming_conventions.md §3.4 for the required fields.
"""

import json
import platform
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def get_platform_tag() -> str:
    """Return a short platform identifier for config.json."""
    system = platform.system()
    machine = platform.machine()
    if system == "Windows":
        return f"windows_{machine}"
    elif system == "Darwin":
        return f"mac_{machine}"
    return f"{system.lower()}_{machine}"


@contextmanager
def phase_timer(output_dir: Path, model_id: str = None, **extra_fields):
    """
    Context manager that records timing and writes config.json on exit.

    Usage:
        with phase_timer(Path("tasks/ioi/20_scoring"), model="gpt2",
                         n_heads=144) as cfg:
            # ... do work ...
            cfg["custom_field"] = some_value

    On exit, writes output_dir/config.json with:
        elapsed_sec, platform, completed_at, model (if given), plus extra_fields.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(extra_fields)
    start = time.perf_counter()

    try:
        yield config
    finally:
        config["elapsed_sec"] = round(time.perf_counter() - start, 2)
        config["platform"] = get_platform_tag()
        config["completed_at"] = datetime.now(timezone.utc).isoformat()
        if model_id is not None:
            config["model"] = model_id

        config_path = output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
