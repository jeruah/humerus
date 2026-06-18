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
        min_radius: float = 20.0,
        max_radius: float = 40.0,
        bone_length: Optional[float] = None,
        surface_points: Optional[np.ndarray] = None,
        diameter_length_ratio: float = 0.155,
        ratio_tolerance: float = 0.05
    ) -> bool:
        """
        Valida que una aproximación es aceptable.
        
        Criterios:
        1. Error cuadrático medio < max_error (default 2mm)
        2. Radio en rango fisiológico [20, 40] mm
        3. La correlacion con la longitud del hueso, diámetro/longitud ≈ 15.5% ± 5%
        
        Parameters
        ----------
        sphere : Dict[str, Any]
            Diccionario con 'center', 'radius', 'error'
        max_error : float
            Error máximo aceptable (mm)
        min_radius : float
            Radio mínimo aceptable (mm)
        max_radius : float
            Radio máximo aceptable (mm)
        bone_length : float, optional
            Longitud longitudinal del húmero en mm
        surface_points : np.ndarray, optional
            Superficie completa para estimar longitud si bone_length no se entrega
        diameter_length_ratio : float
            Relación esperada diámetro esfera / longitud hueso
        ratio_tolerance : float
            Tolerancia absoluta de la relación. 0.009 equivale a ±0.9 puntos porcentuales.
        
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
            reasons.append(f"Radius {radius:.2f}mm outside range [{min_radius}, {max_radius}]")

        if bone_length is None and surface_points is not None:
            bone_length = self._estimate_bone_length(surface_points)

        ratio = None
        if bone_length is not None:
            bone_length = float(bone_length)
            if bone_length <= 0:
                is_valid = False
                reasons.append("Bone length must be positive")
            else:
                diameter = 2.0 * float(radius)
                ratio = diameter / bone_length
                min_ratio = diameter_length_ratio - ratio_tolerance
                max_ratio = diameter_length_ratio + ratio_tolerance
                if not (min_ratio <= ratio <= max_ratio):
                    is_valid = False
                    reasons.append(
                        f"Diameter/bone_length ratio {ratio:.4f} outside "
                        f"[{min_ratio:.4f}, {max_ratio:.4f}]"
                    )
        
        self.validations['approximation_valid'] = is_valid
        self.log_step("validate_approximation", {
            "valid": is_valid,
            "error": float(error),
            "radius": float(radius),
            "bone_length": float(bone_length) if bone_length is not None else None,
            "diameter_length_ratio": float(ratio) if ratio is not None else None,
            "expected_ratio": float(diameter_length_ratio),
            "ratio_tolerance": float(ratio_tolerance),
            "reasons": reasons if not is_valid else ["OK"]
        })
        
        return is_valid

    @staticmethod
    def _estimate_bone_length(surface_points: np.ndarray) -> float:
        """Estima longitud del hueso proyectando la nube sobre su eje PCA."""
        points = np.asarray(surface_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("surface_points debe tener shape (N, 3)")
        centered = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        projections = centered @ axis
        return float(projections.max() - projections.min())
    
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
