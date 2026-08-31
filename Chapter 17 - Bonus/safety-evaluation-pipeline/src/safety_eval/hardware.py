"""Portable, conservative machine inspection and execution-profile selection."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Accelerator:
    kind: str
    name: str
    memory_bytes: int | None


@dataclass(frozen=True)
class Machine:
    schema_version: str
    operating_system: str
    architecture: str
    cpu_count: int
    memory_bytes: int | None
    accelerators: tuple[Accelerator, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["accelerators"] = [asdict(item) for item in self.accelerators]
        return data


def _memory_bytes() -> int | None:
    try:
        if platform.system() == "Darwin":
            return int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
                )
            )
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return None


def _accelerators() -> tuple[Accelerator, ...]:
    detected: list[Accelerator] = []
    try:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                detected.append(Accelerator("cuda", properties.name, properties.total_memory))
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            detected.append(Accelerator("mps", "Apple Metal Performance Shaders", None))
    except ImportError:
        pass
    return tuple(detected)


def inspect_machine() -> Machine:
    return Machine(
        schema_version="1.0",
        operating_system=platform.system().lower(),
        architecture=platform.machine().lower(),
        cpu_count=os.cpu_count() or 1,
        memory_bytes=_memory_bytes(),
        accelerators=_accelerators(),
    )


def load_profiles(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as profile_file:
        payload = json.load(profile_file)
    return payload["profiles"]


def recommend_profile(machine: Machine, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    accelerator_kinds = {item.kind for item in machine.accelerators}
    available_memory_gib = (machine.memory_bytes or 0) / (1024**3)

    if "cuda" in accelerator_kinds:
        largest_vram = max(
            (item.memory_bytes or 0 for item in machine.accelerators if item.kind == "cuda"),
            default=0,
        ) / (1024**3)
        target = "full_accelerator" if largest_vram >= 40 else "sampled_accelerator"
    elif "mps" in accelerator_kinds and available_memory_gib >= 24:
        target = "sampled_accelerator"
    else:
        target = "smoke_cpu"

    return next(profile for profile in profiles if profile["profile_id"] == target)
