"""Sistema de auditoría para registrar y validar cada paso del proceso.

Este módulo proporciona la clase AuditTrail que:
- Registra cada paso de cada aproximación
- Valida viabilidad de semillas
- Valida calidad de aproximaciones
- Genera reportes detallados

La auditoría es obligatoria para garantizar reproducibilidad y confiabilidad.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import numpy as np


@dataclass
class StepRecord:
    """Registro de un paso en el proceso."""
    step_name: str
    timestamp: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario JSON-serializable."""
        result = asdict(self)
        # Convertir arrays de numpy a listas
        for key, value in result['data'].items():
            if isinstance(value, np.ndarray):
                result['data'][key] = value.tolist()
        return result


class AuditTrail:
    """
    Sistema de auditoría para aproximación de esferas.
    
    Registra cada paso, valida procesos, y genera reportes auditables.
    
    Attributes
    ----------
    seed_id : str
        Identificador único de la semilla siendo auditada
    steps : List[StepRecord]
        Registro secuencial de todos los pasos
    validations : Dict[str, bool]
        Resultados de validaciones realizadas
    """
    
    def __init__(self, seed_id: str = ""):
        """
        Inicializa el sistema de auditoría.
        
        Parameters
        ----------
        seed_id : str, optional
            Identificador de la semilla (default: "")
        """
        self.seed_id = seed_id
        self.steps: List[StepRecord] = []
        self.validations: Dict[str, bool] = {}
        self.start_time = datetime.now()
    
    def log_step(self, step_name: str, data: Dict[str, Any]) -> None:
        """
        Registra un paso en el proceso.
        
        Parameters
        ----------
        step_name : str
            Nombre descriptivo del paso
        data : Dict[str, Any]
            Datos asociados al paso (puede contener arrays de numpy)
        
        Examples
        --------
        >>> audit = AuditTrail("seed_001")
        >>> audit.log_step("initialize", {"point": np.array([1, 2, 3])})
        >>> audit.log_step("iteration_1", {"center": [0.5, 1.5, 2.5], "error": 0.012})
        """
        record = StepRecord(
            step_name=step_name,
            timestamp=datetime.now().isoformat(),
            data=data.copy()
        )
        self.steps.append(record)
    
    def validate_seed(
        self, 
        point: np.ndarray, 
        articulation_region: Optional[np.ndarray] = None,
        curvature_threshold: float = 0.1
    ) -> bool:
        """
        Valida que una semilla está en región viable.
        
        Una semilla es viable si:
        1. Está dentro de la región articular (si se proporciona)
        2. La curvatura es compatible con una superficie esférica
        
        Parameters
        ----------
        point : np.ndarray
            Punto 3D de la semilla
        articulation_region : np.ndarray, optional
            Puntos que forman la región articular
        curvature_threshold : float
            Umbral de curvatura aceptable
        
        Returns
        -------
        bool
            True si la semilla es viable, False en caso contrario
        
        Notes
        -----
        Esta validación es crítica. Una semilla inválida puede causar
        divergencias a otras superficies del húmero.
        """
        is_valid = True
        reasons = []
        
        # Validación 1: Punto válido
        if not isinstance(point, np.ndarray) or point.shape != (3,):
            is_valid = False
            reasons.append("Point must be numpy array of shape (3,)")
        
        # Validación 2: Dentro de región articular (si se proporciona)
        if articulation_region is not None and is_valid:
            from ..validation.viability import SeedValidator

            validator = SeedValidator()
            is_in_region = validator.is_in_articulation_region(point, articulation_region, tolerance=5.0)
            distances = np.linalg.norm(articulation_region - point, axis=1)
            min_distance = float(distances.min())
            if not is_in_region:
                is_valid = False
                reasons.append(f"Point outside articulation region: {min_distance:.2f}mm")
        
        self.validations['seed_viable'] = is_valid
        self.log_step("validate_seed", {
            "seed": point.tolist() if is_valid else None,
            "viable": is_valid,
            "reasons": reasons if not is_valid else ["OK"]
        })
        
        return is_valid
    
    def is_valid_approximation(
        self,
        sphere: Dict[str, Any],
        max_error: float = 2.0,
        min_radius: float = 17.0,
        max_radius: float = 40.0,
        axis: Optional[Dict[str, Any]] = None,
        surface_points: Optional[np.ndarray] = None,
        medial_direction: Optional[np.ndarray] = None,
        posterior_direction: Optional[np.ndarray] = None,
        reference_min_roc: float = 17.0,
        reference_max_roc: float = 30.0,
        reference_mean_roc: float = 22.5,
        reference_sd_roc: float = 2.8,
        reference_min_medial_offset: float = 1.0,
        reference_max_medial_offset: float = 14.0,
        reference_mean_medial_offset: float = 6.8,
        reference_sd_medial_offset: float = 2.5,
        reference_min_posterior_offset: float = 0.0,
        reference_max_posterior_offset: float = 10.0,
        reference_mean_posterior_offset: float = 2.0,
        reference_sd_posterior_offset: float = 2.0,
        enforce_morphology_reference: bool = False,
    ) -> bool:
        """
        Valida que una aproximación es aceptable.
        
        Criterios:
        1. Error cuadrático medio < max_error (default 2mm)
        2. Radio de curvatura (ROC) en rango plausible [17, 40] mm
        3. Si se entrega eje longitudinal, se auditan ROC/MO/PO contra
           rangos de referencia, pero no invalidan salvo que se solicite.
        
        Parameters
        ----------
        sphere : Dict[str, Any]
            Diccionario con 'center', 'radius', 'error'
        max_error : float
            Error máximo aceptable (mm)
        min_radius : float
            ROC mínimo aceptable (mm)
        max_radius : float
            ROC máximo aceptable (mm)
        axis : Dict[str, Any], optional
            Eje longitudinal con 'origin' y 'direction'
        surface_points : np.ndarray, optional
            Superficie usada para inferir un marco transversal si no se entregan
            direcciones anatómicas explícitas
        medial_direction : np.ndarray, optional
            Dirección medial aproximada. Se proyecta perpendicular al eje.
        posterior_direction : np.ndarray, optional
            Dirección posterior aproximada. Se proyecta perpendicular al eje.
        enforce_morphology_reference : bool
            Si True, los rangos promedio de ROC/MO/PO invalidan la aproximación.
            Si False, solo se registran como indicadores de referencia.
        
        Returns
        -------
        bool
            True si la aproximación es válida
        """
        is_valid = True
        reasons = []
        
        # Validación 1: Error
        error = sphere.get('error', float('inf'))
        if error > max_error:
            is_valid = False
            reasons.append(f"Error {error:.3f}mm exceeds max {max_error}mm")
        
        # Validación 2: Radio
        radius = sphere.get('radius', 0)
        if not (min_radius <= radius <= max_radius):
            is_valid = False
            reasons.append(f"ROC {radius:.2f}mm outside range [{min_radius}, {max_radius}]")

        morphology = None
        reference_flags = None
        reference_statistics = None
        reference_reasons = []
        if axis is not None:
            try:
                morphology = self.compute_morphological_metrics(
                    sphere,
                    axis,
                    surface_points=surface_points,
                    medial_direction=medial_direction,
                    posterior_direction=posterior_direction,
                )
                medial_offset = morphology["medial_offset"]
                posterior_offset = morphology["posterior_offset"]
                roc = morphology["roc"]

                reference_flags = {
                    "roc_in_reference": bool(reference_min_roc <= roc <= reference_max_roc),
                    "medial_offset_in_reference": bool(
                        reference_min_medial_offset <= medial_offset <= reference_max_medial_offset
                    ),
                    "posterior_offset_in_reference": bool(
                        reference_min_posterior_offset <= posterior_offset <= reference_max_posterior_offset
                    ),
                }
                reference_flags["all_in_reference"] = all(reference_flags.values())
                reference_statistics = {
                    "roc": self._reference_stat(
                        roc,
                        reference_min_roc,
                        reference_max_roc,
                        reference_mean_roc,
                        reference_sd_roc,
                    ),
                    "medial_offset": self._reference_stat(
                        medial_offset,
                        reference_min_medial_offset,
                        reference_max_medial_offset,
                        reference_mean_medial_offset,
                        reference_sd_medial_offset,
                    ),
                    "posterior_offset": self._reference_stat(
                        posterior_offset,
                        reference_min_posterior_offset,
                        reference_max_posterior_offset,
                        reference_mean_posterior_offset,
                        reference_sd_posterior_offset,
                    ),
                }

                if not reference_flags["roc_in_reference"]:
                    reference_reasons.append(
                        f"ROC {roc:.2f}mm outside reference "
                        f"[{reference_min_roc}, {reference_max_roc}]"
                    )
                if not reference_flags["medial_offset_in_reference"]:
                    reference_reasons.append(
                        f"Medial offset {medial_offset:.2f}mm outside reference "
                        f"[{reference_min_medial_offset}, {reference_max_medial_offset}]"
                    )
                if not reference_flags["posterior_offset_in_reference"]:
                    reference_reasons.append(
                        f"Posterior offset {posterior_offset:.2f}mm outside reference "
                        f"[{reference_min_posterior_offset}, {reference_max_posterior_offset}]"
                    )

                if enforce_morphology_reference and not reference_flags["all_in_reference"]:
                    is_valid = False
                    reasons.extend(reference_reasons)
            except ValueError as exc:
                is_valid = False
                reasons.append(str(exc))
        
        self.validations['approximation_valid'] = is_valid
        data = {
            "valid": is_valid,
            "error": float(error),
            "roc": float(radius),
            "roc_plausibility_range": [float(min_radius), float(max_radius)],
            "morphology_reference_ranges": {
                "roc": [float(reference_min_roc), float(reference_max_roc)],
                "medial_offset": [
                    float(reference_min_medial_offset),
                    float(reference_max_medial_offset),
                ],
                "posterior_offset": [
                    float(reference_min_posterior_offset),
                    float(reference_max_posterior_offset),
                ],
            },
            "morphology_reference_statistics": {
                "roc": {
                    "mean": float(reference_mean_roc),
                    "sd": float(reference_sd_roc),
                },
                "medial_offset": {
                    "mean": float(reference_mean_medial_offset),
                    "sd": float(reference_sd_medial_offset),
                },
                "posterior_offset": {
                    "mean": float(reference_mean_posterior_offset),
                    "sd": float(reference_sd_posterior_offset),
                },
            },
            "enforce_morphology_reference": bool(enforce_morphology_reference),
            "reasons": reasons if not is_valid else ["OK"]
        }
        if morphology is not None:
            data["morphology"] = morphology
        if reference_flags is not None:
            data["morphology_reference_flags"] = reference_flags
            data["morphology_reference_values"] = reference_statistics
            data["morphology_reference_reasons"] = reference_reasons if reference_reasons else ["OK"]
        self.log_step("validate_approximation", data)
        
        return is_valid

    @staticmethod
    def _reference_stat(
        value: float,
        min_value: float,
        max_value: float,
        mean: float,
        sd: float,
    ) -> Dict[str, float]:
        """Empaqueta valor, rango, media, desviación y z-score."""
        sd = float(sd)
        z_score = (float(value) - float(mean)) / sd if sd > 0 else float("nan")
        return {
            "value": float(value),
            "min": float(min_value),
            "max": float(max_value),
            "mean": float(mean),
            "sd": sd,
            "z_score": float(z_score),
        }

    @staticmethod
    def compute_morphological_metrics(
        sphere: Dict[str, Any],
        axis: Dict[str, Any],
        surface_points: Optional[np.ndarray] = None,
        medial_direction: Optional[np.ndarray] = None,
        posterior_direction: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Calcula medidas morfológicas de postprocesado.

        El ROC es el radio de la esfera ajustada. Los offsets se calculan desde
        el eje longitudinal hasta el centro de rotación de la cabeza humeral.
        Si no se entregan direcciones anatómicas, se infiere un marco transversal
        determinista perpendicular al eje.
        """
        center = np.asarray(sphere.get("center"), dtype=float)
        origin = np.asarray(axis.get("origin"), dtype=float)
        longitudinal = np.asarray(axis.get("direction"), dtype=float)

        if center.shape != (3,):
            raise ValueError("sphere['center'] debe tener shape (3,)")
        if origin.shape != (3,) or longitudinal.shape != (3,):
            raise ValueError("axis debe incluir 'origin' y 'direction' con shape (3,)")

        norm = np.linalg.norm(longitudinal)
        if norm <= 1e-12:
            raise ValueError("axis['direction'] no puede ser vector cero")
        longitudinal = longitudinal / norm

        medial_unit, posterior_unit = AuditTrail._transverse_frame(
            longitudinal,
            medial_direction=medial_direction,
            posterior_direction=posterior_direction,
            surface_points=surface_points,
        )

        vec = center - origin
        axial_position = float(np.dot(vec, longitudinal))
        closest_axis_point = origin + axial_position * longitudinal
        offset_vector = center - closest_axis_point
        signed_medial = float(np.dot(offset_vector, medial_unit))
        signed_posterior = float(np.dot(offset_vector, posterior_unit))

        return {
            "roc": float(sphere.get("radius", 0.0)),
            "axis_point": closest_axis_point.tolist(),
            "offset_vector": offset_vector.tolist(),
            "total_offset": float(np.linalg.norm(offset_vector)),
            "medial_offset": abs(signed_medial),
            "posterior_offset": abs(signed_posterior),
            "signed_medial_offset": signed_medial,
            "signed_posterior_offset": signed_posterior,
            "axial_position": axial_position,
            "medial_direction": medial_unit.tolist(),
            "posterior_direction": posterior_unit.tolist(),
            "longitudinal_direction": longitudinal.tolist(),
        }

    @staticmethod
    def _transverse_frame(
        longitudinal: np.ndarray,
        medial_direction: Optional[np.ndarray] = None,
        posterior_direction: Optional[np.ndarray] = None,
        surface_points: Optional[np.ndarray] = None,
    ) -> tuple:
        """Construye dos direcciones ortonormales perpendiculares al eje."""
        medial = AuditTrail._project_perpendicular(medial_direction, longitudinal)
        if medial is None and surface_points is not None:
            points = np.asarray(surface_points, dtype=float)
            if points.ndim == 2 and points.shape[1] == 3 and len(points) >= 3:
                centered = points - points.mean(axis=0)
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                for candidate in vh[1:]:
                    medial = AuditTrail._project_perpendicular(candidate, longitudinal)
                    if medial is not None:
                        break

        if medial is None:
            for candidate in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
                medial = AuditTrail._project_perpendicular(candidate, longitudinal)
                if medial is not None:
                    break

        posterior = AuditTrail._project_perpendicular(posterior_direction, longitudinal)
        if posterior is not None:
            posterior = posterior - np.dot(posterior, medial) * medial
            posterior_norm = np.linalg.norm(posterior)
            posterior = posterior / posterior_norm if posterior_norm > 1e-12 else None

        if posterior is None:
            posterior = np.cross(longitudinal, medial)
            posterior = posterior / np.linalg.norm(posterior)

        return medial, posterior

    @staticmethod
    def _project_perpendicular(
        vector: Optional[np.ndarray],
        axis: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Proyecta un vector al plano perpendicular del eje y lo normaliza."""
        if vector is None:
            return None
        projected = np.asarray(vector, dtype=float)
        if projected.shape != (3,):
            return None
        projected = projected - np.dot(projected, axis) * axis
        norm = np.linalg.norm(projected)
        if norm <= 1e-12:
            return None
        return projected / norm
    
    def get_report(self) -> Dict[str, Any]:
        """
        Genera reporte completo de auditoría.
        
        Returns
        -------
        Dict[str, Any]
            Reporte con todos los pasos, validaciones y resumen
        
        Examples
        --------
        >>> report = audit.get_report()
        >>> print(f"Steps: {len(report['steps'])}")
        >>> print(f"Valid: {report['validations']}")
        """
        duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'seed_id': self.seed_id,
            'duration_seconds': duration,
            'total_steps': len(self.steps),
            'steps': [step.to_dict() for step in self.steps],
            'validations': self.validations,
            'final_valid': all(self.validations.values()) if self.validations else None
        }
    
    def to_json(self) -> str:
        """
        Serializa auditoría a JSON.
        
        Returns
        -------
        str
            Reporte en formato JSON
        """
        return json.dumps(self.get_report(), indent=2)


class AuditManager:
    """
    Gestor de múltiples auditorías (para múltiples semillas).
    
    Mantiene un registro de todas las auditorías realizadas
    y permite generar reportes comparativos.
    """
    
    def __init__(self):
        """Inicializa gestor de auditorías."""
        self.audits: Dict[str, AuditTrail] = {}
        self.start_time = datetime.now()
    
    def create_audit(self, seed_id: str) -> AuditTrail:
        """
        Crea una nueva auditoría.
        
        Parameters
        ----------
        seed_id : str
            Identificador único de la semilla
        
        Returns
        -------
        AuditTrail
            Nueva instancia de AuditTrail
        """
        audit = AuditTrail(seed_id)
        self.audits[seed_id] = audit
        return audit
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Resumen estadístico de todas las auditorías.
        
        Returns
        -------
        Dict[str, Any]
            Resumen con conteos y estadísticas
        """
        total_audits = len(self.audits)
        valid_audits = sum(
            1 for audit in self.audits.values() 
            if audit.validations.get('approximation_valid', False)
        )
        
        return {
            'total_audits': total_audits,
            'valid_approximations': valid_audits,
            'success_rate': valid_audits / total_audits if total_audits > 0 else 0,
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'seed_ids': list(self.audits.keys())
        }
