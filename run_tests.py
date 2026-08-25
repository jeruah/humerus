#!/usr/bin/env python3
"""Ejecuta la suite de tests con cobertura y (opcionalmente) el linter.

Uso:
    python run_tests.py            # pytest + cobertura (umbral en pyproject.toml)
    python run_tests.py --lint     # además ejecuta ruff check
"""

import subprocess
import sys


def main() -> int:
    commands = [
        ([sys.executable, "-m", "pytest"], "Tests con cobertura"),
    ]
    if "--lint" in sys.argv:
        commands.append(
            ([sys.executable, "-m", "ruff", "check", "humero", "examples", "tests"], "Lint (ruff)")
        )

    ok = True
    for cmd, label in commands:
        print(f"\n=== {label} ===")
        print(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        ok = ok and result.returncode == 0

    print("\n✓ Todos los pasos OK" if ok else "\n✗ Hubo fallos")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
