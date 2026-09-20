"""Verify the local research environment and optionally write a JSON report."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "PyYAML": "yaml",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "pyarrow": "pyarrow",
    "streamlit": "streamlit",
    "plotly": "plotly",
    "pytest": "pytest",
    "torch": "torch",
}


def collect_environment() -> dict[str, object]:
    versions: dict[str, str] = {}
    imported: dict[str, object] = {}
    for package, module_name in MODULES.items():
        module = importlib.import_module(module_name)
        imported[module_name] = module
        versions[package] = str(getattr(module, "__version__", "unknown"))

    torch = imported["torch"]
    cuda_available = bool(torch.cuda.is_available())
    device = torch.device("cuda" if cuda_available else "cpu")
    tensor_sum = float(torch.tensor([1.0, 2.0], device=device).sum().cpu())
    result: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
        "torch_device": str(device),
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "tensor_smoke_sum": tensor_sum,
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        result["gpu"] = {
            "name": properties.name,
            "memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    result = collect_environment()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
