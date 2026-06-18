"""Aproximación del eje longitudinal del húmero."""

import numpy as np
from typing import Tuple, Dict, Any
from sklearn.decomposition import PCA


class AxisApproximator:
    """
    Aproxima eje longitudinal del húmero.
    
    El eje se extiende desde la cabeza articular hasta el extremo distal.
    
    Métodos:
    - PCA para encontrar dirección principal
    - Parametrización de línea 3D
    - Validación de alineación
    """
    
    @staticmethod
    def compute_longitudinal_axis(
        surface_points: np.ndarray,
        method: str = "shaft_pca",
        shaft_trim_fraction: float = 0.18
    ) -> Dict[str, Any]:
        """
        Calcula eje longitudinal del húmero.
        
        Parameters
        ----------
        surface_points : np.ndarray
            Puntos de la superficie (shape: (N, 3))
        method : str
            Método: "shaft_pca", "pca" o "least_squares_line"
        shaft_trim_fraction : float
            Fracción proximal y distal a ignorar al usar "shaft_pca".
        
        Returns
        -------
        Dict[str, any]
            {
                'origin': np.ndarray (3,),  # Punto de inicio (cabeza)
                'direction': np.ndarray (3,),  # Dirección unitaria
                'distal_point': np.ndarray (3,),  # Extremo distal
                'length': float,  # Longitud del eje (mm)
                'validation': Dict  # Validaciones
            }
        
        Notes
        -----
        El eje se parametriza como:
        P(t) = origin + t * direction
        
        donde t ∈ [0, length]
        """
        surface_points = np.asarray(surface_points, dtype=float)
        if surface_points.ndim != 2 or surface_points.shape[1] != 3:
            raise ValueError("surface_points debe tener shape (N, 3)")
        if len(surface_points) < 2:
            raise ValueError("Se requieren al menos 2 puntos para calcular el eje")

        shaft_points = surface_points
        if method == "shaft_pca":
            direction, point_on_line, shaft_points = AxisApproximator._compute_axis_shaft_pca(
                surface_points,
                trim_fraction=shaft_trim_fraction,
            )
        elif method == "pca":
            direction, point_on_line = AxisApproximator._compute_axis_pca(surface_points)
        elif method == "least_squares_line":
            direction, point_on_line = AxisApproximator._compute_axis_least_squares(surface_points)
        else:
            raise ValueError("method debe ser 'shaft_pca', 'pca' o 'least_squares_line'")

        projections = (surface_points - point_on_line) @ direction
        min_t = float(projections.min())
        max_t = float(projections.max())
        end_a = point_on_line + min_t * direction
        end_b = point_on_line + max_t * direction

        head_position = AxisApproximator.find_head_position(surface_points)
        if np.linalg.norm(end_b - head_position) < np.linalg.norm(end_a - head_position):
            origin = end_b
            distal_point = end_a
            direction = -direction
        else:
            origin = end_a
            distal_point = end_b

        length = float(np.linalg.norm(distal_point - origin))
        axis = {
            "origin": origin,
            "direction": direction / np.linalg.norm(direction),
            "distal_point": distal_point,
            "length": length,
            "head_position": head_position,
            "method": method,
            "shaft_trim_fraction": float(shaft_trim_fraction) if method == "shaft_pca" else 0.0,
            "axis_fit_point_count": int(len(shaft_points)),
            "total_point_count": int(len(surface_points)),
        }
        axis["validation"] = AxisApproximator.validate_axis(axis, surface_points)
        return axis

    @staticmethod
    def _compute_axis_shaft_pca(
        surface_points: np.ndarray,
        trim_fraction: float = 0.18
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula el eje usando PCA solo en la diáfisis/tallo.

        Primero obtiene una dirección global aproximada. Luego proyecta los
        puntos sobre esa dirección y descarta los extremos proximal y distal,
        donde la cabeza y el extremo distal suelen tener geometría irregular y
        alta densidad de puntos. El PCA final se ajusta sobre la zona central.
        """
        points = np.asarray(surface_points, dtype=float)
        trim = float(np.clip(trim_fraction, 0.0, 0.45))
        initial_direction, initial_centroid = AxisApproximator._compute_axis_pca(points)
        projections = (points - initial_centroid) @ initial_direction

        low = np.quantile(projections, trim)
        high = np.quantile(projections, 1.0 - trim)
        shaft_mask = (projections >= low) & (projections <= high)
        shaft_points = points[shaft_mask]

        if len(shaft_points) < max(8, int(0.1 * len(points))):
            shaft_points = points

        direction, centroid = AxisApproximator._compute_axis_pca(shaft_points)
        if np.dot(direction, initial_direction) < 0:
            direction = -direction
        return direction, centroid, shaft_points
    
    @staticmethod
    def _compute_axis_pca(
        surface_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula eje usando PCA.
        
        Parameters
        ----------
        surface_points : np.ndarray
            Puntos de superficie
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (direction, centroid)
        """
        pca = PCA(n_components=1)
        pca.fit(surface_points)
        
        direction = pca.components_[0]
        direction = direction / np.linalg.norm(direction)
        centroid = pca.mean_
        
        return direction, centroid
    
    @staticmethod
    def _compute_axis_least_squares(
        surface_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula eje usando mínimos cuadrados.
        
        Minimiza suma de distancias perpendiculares a la línea.
        
        Parameters
        ----------
        surface_points : np.ndarray
            Puntos de superficie
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (direction, point_on_line)
        """
        centered = surface_points - surface_points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        direction = direction / np.linalg.norm(direction)
        return direction, surface_points.mean(axis=0)
    
    @staticmethod
    def find_head_position(
        surface_points: np.ndarray,
        head_radius: float = 30.0,
        head_region_fraction: float = 0.2
    ) -> np.ndarray:
        """
        Identifica posición de cabeza del húmero.
        
        La cabeza corresponde a la región esférica articular.
        
        Parameters
        ----------
        surface_points : np.ndarray
            Puntos de superficie
        head_radius : float
            Radio estimado de cabeza (mm)
        head_region_fraction : float
            Fracción de altura que corresponde a cabeza
        
        Returns
        -------
        np.ndarray
            Posición de cabeza (shape: (3,))
        """
        direction, centroid = AxisApproximator._compute_axis_pca(surface_points)
        projections = (surface_points - centroid) @ direction
        cutoff = np.quantile(projections, 1.0 - head_region_fraction)
        high_end = surface_points[projections >= cutoff]
        low_end = surface_points[projections <= np.quantile(projections, head_region_fraction)]

        high_score = np.mean(np.linalg.norm(high_end - high_end.mean(axis=0), axis=1))
        low_score = np.mean(np.linalg.norm(low_end - low_end.mean(axis=0), axis=1))
        return high_end.mean(axis=0) if high_score >= low_score else low_end.mean(axis=0)
    
    @staticmethod
    def find_distal_position(
        surface_points: np.ndarray,
        head_position: np.ndarray
    ) -> np.ndarray:
        """
        Identifica extremo distal del húmero.
        
        Parameters
        ----------
        surface_points : np.ndarray
            Puntos de superficie
        head_position : np.ndarray
            Posición de cabeza
        
        Returns
        -------
        np.ndarray
            Posición distal (shape: (3,))
        """
        distances = np.linalg.norm(surface_points - head_position, axis=1)
        return surface_points[int(np.argmax(distances))]
    
    @staticmethod
    def validate_axis(
        axis_dict: Dict,
        surface_points: np.ndarray
    ) -> Dict[str, bool]:
        """
        Valida que el eje aproximado es correcto.
        
        Validaciones:
        1. Eje pasa por región de cabeza
        2. Eje apunta hacia región distal
        3. Longitud es fisiológicamente razonable
        
        Parameters
        ----------
        axis_dict : Dict
            Diccionario de eje
        surface_points : np.ndarray
            Puntos de superficie
        
        Returns
        -------
        Dict[str, bool]
            Resultados de validaciones
        """
        validations = {
            'passes_through_head': False,
            'points_to_distal': False,
            'reasonable_length': False,
            'overall_valid': False
        }
        
        origin = axis_dict["origin"]
        direction = axis_dict["direction"]
        distal = axis_dict["distal_point"]
        length = axis_dict["length"]
        head_position = axis_dict.get("head_position", origin)

        head_distance = AxisApproximator.distance_point_to_axis(head_position, origin, direction)
        validations["passes_through_head"] = bool(head_distance <= max(15.0, 0.15 * length))
        validations["points_to_distal"] = bool(np.dot(distal - origin, direction) > 0)
        validations["reasonable_length"] = bool(20.0 <= length <= 500.0)
        
        validations['overall_valid'] = all(
            validations[key]
            for key in ("passes_through_head", "points_to_distal", "reasonable_length")
        )
        return validations
    
    @staticmethod
    def distance_point_to_axis(
        point: np.ndarray,
        axis_origin: np.ndarray,
        axis_direction: np.ndarray
    ) -> float:
        """
        Calcula distancia de un punto al eje.
        
        Parameters
        ----------
        point : np.ndarray
            Punto (shape: (3,))
        axis_origin : np.ndarray
            Origen del eje (shape: (3,))
        axis_direction : np.ndarray
            Dirección unitaria (shape: (3,))
        
        Returns
        -------
        float
            Distancia perpendicular (mm)
        """
        # Vector desde origen del eje al punto
        vec = point - axis_origin
        
        # Proyección en dirección del eje
        proj_length = np.dot(vec, axis_direction)
        projection = proj_length * axis_direction
        
        # Distancia perpendicular
        perpendicular = vec - projection
        distance = np.linalg.norm(perpendicular)
        
        return distance
