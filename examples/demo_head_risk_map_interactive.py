"""Wrapper de compatibilidad: el mapa de riesgo interactivo vive en ``humero.web.risk_map_interactive``.

Uso:
    python examples/demo_head_risk_map_interactive.py --synthetic-demo --no-browser

O, si el paquete está instalado:
    humero-risk-map-interactive --synthetic-demo --no-browser
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from humero.web.risk_map_interactive import *
from humero.web.risk_map_interactive import main

if __name__ == "__main__":
    main()
