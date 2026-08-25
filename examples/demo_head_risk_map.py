"""Wrapper de compatibilidad: el mapa de riesgo vive en ``humero.web.risk_map``.

Uso:
    python examples/demo_head_risk_map.py --synthetic-demo --output /tmp/risk.html

O, si el paquete está instalado:
    humero-risk-map --synthetic-demo --output /tmp/risk.html
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from humero.web.risk_map import *
from humero.web.risk_map import main

if __name__ == "__main__":
    main()
