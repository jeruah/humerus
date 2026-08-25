"""Configuración central del pipeline de esfera/eje humeral.

Centraliza rangos, tolerancias y referencias morfológicas que antes estaban
duplicados en validación, RANSAC, best-fit, demos y tests. Cambiar un valor
aquí propaga a todo el pipeline.
"""

from dataclasses import dataclass

import numpy as np

# Rango plausible de radio de curvatura (ROC) de la cabeza humeral (mm).
ROC_MIN: float = 17.0
ROC_MAX: float = 40.0
RADIUS_RANGE: tuple[float, float] = (ROC_MIN, ROC_MAX)

# Tolerancias por defecto (mm).
DEFAULT_DISTANCE_TOLERANCE: float = 1.5
DEFAULT_MAX_ERROR: float = 2.0

# Direcciones anatómicas por defecto (aproximación de marco global).
# NOTA: son una aproximación; una fase futura debe derivar un marco
# anatómico específico del lado y orientación del STL.
DEFAULT_MEDIAL_DIRECTION: np.ndarray = np.array([1.0, 0.0, 0.0])
DEFAULT_POSTERIOR_DIRECTION: np.ndarray = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class MorphologyReference:
    """Rangos, medias y desviaciones de referencia morfológica humeral."""

    min_roc: float = 17.0
    max_roc: float = 30.0
    mean_roc: float = 22.5
    sd_roc: float = 2.8
    min_medial_offset: float = 1.0
    max_medial_offset: float = 14.0
    mean_medial_offset: float = 6.8
    sd_medial_offset: float = 2.5
    min_posterior_offset: float = 0.0
    max_posterior_offset: float = 10.0
    mean_posterior_offset: float = 2.0
    sd_posterior_offset: float = 2.0


DEFAULT_MORPHOLOGY_REFERENCE = MorphologyReference()


def anatomical_directions() -> tuple[np.ndarray, np.ndarray]:
    """Direcciones medial/posterior por defecto para el marco transversal."""
    return DEFAULT_MEDIAL_DIRECTION.copy(), DEFAULT_POSTERIOR_DIRECTION.copy()
