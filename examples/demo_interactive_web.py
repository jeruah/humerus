"""Wrapper de compatibilidad: la demo principal vive en ``humero.web.server``.

Uso:
    python examples/demo_interactive_web.py --synthetic-demo --no-browser

O, si el paquete está instalado:
    humero-demo --synthetic-demo --no-browser
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from humero.web.server import *
from humero.web.server import main

if __name__ == "__main__":
    main()
