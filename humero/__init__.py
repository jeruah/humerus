"""humero: aproximación de esfera en cabeza humeral a partir de STL.

Paquete instalable con el pipeline completo:

- Carga/limpieza/discretización de mallas STL (trimesh).
- Estimación del eje diafisario robusto.
- Detección de la esfera articular por RANSAC + prior de esfericidad local.
- Validación morfológica (ROC, medial/posterior offset).
- Auditoría JSON-friendly y visualización 3D (Plotly / matplotlib).

Uso típico::

    from humero import STLLoader, MeshCleaner, SphereRansacFitter

    mesh = STLLoader.load("humerus.stl")
    cleaned = MeshCleaner().clean(mesh.vertices, mesh.faces)
    result = SphereRansacFitter().fit(cleaned)
"""

from pathlib import Path

from humero.approximation.sphere import SphericalApproximator
from humero.audit.trail import AuditManager, AuditTrail
from humero.axis.longitudinal import AxisApproximator
from humero.config import MorphologyReference
from humero.geometry.sphere import SphereGeometry
from humero.mesh.cleaner import CleanedMesh, MeshCleaner
from humero.mesh.discretizer import MeshDiscretizer
from humero.mesh.loader import STLLoader, STLMesh
from humero.optimization.best_fit import HumeralHeadBestFitSearch
from humero.optimization.refinement import SphereOptimizer
from humero.optimization.sphere_ransac import SphereRansacConfig, SphereRansacFitter
from humero.validation.sphere import SphereValidator
from humero.validation.viability import SeedValidator
from humero.visualization.interactive_web import InteractiveWeb3D
from humero.visualization.visualizer import InteractiveVisualizer, Visualizer3D

__version__ = "0.1.0"

__all__ = [
    "AuditManager",
    "AuditTrail",
    "AxisApproximator",
    "CleanedMesh",
    "HumeralHeadBestFitSearch",
    "InteractiveVisualizer",
    "InteractiveWeb3D",
    "MeshCleaner",
    "MeshDiscretizer",
    "MorphologyReference",
    "STLLoader",
    "STLMesh",
    "SeedValidator",
    "SphereGeometry",
    "SphereOptimizer",
    "SphereRansacConfig",
    "SphereRansacFitter",
    "SphereValidator",
    "SphericalApproximator",
    "Visualizer3D",
    "sample_data_dir",
    "sample_stl",
]


def sample_data_dir() -> Path:
    """Directorio con los STL de muestra incluidos en el paquete."""
    return Path(__file__).parent / "data" / "sample_humeri"


def sample_stl(name: str) -> Path:
    """Ruta a un STL de muestra dentro del paquete (o levanta FileNotFoundError)."""
    path = sample_data_dir() / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path
