"""Aproximación del eje longitudinal del húmero."""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, QhullError


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
        method: str = "diaphyseal_slice_axis",
        shaft_trim_fraction: float = 0.18,
        shaft_end_trim_bins: int = 2,
        slice_count: int = 28,
        voxel_size: float = 3.0,
        shaft_radius_quantile: float = 0.60,
        complete_length_threshold: float = 240.0,
        crop_fraction: float = 0.20,
        slice_spike_ratio: float = 2.5,
        ransac_iterations: int = 128,
        ransac_residual_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calcula eje longitudinal del húmero.
        
        Parameters
        ----------
        surface_points : np.ndarray
            Puntos de la superficie (shape: (N, 3))
        method : str
            Método: "diaphyseal_slice_axis", "shaft_pca", "pca" o
            "least_squares_line"
        shaft_trim_fraction : float
            Fracción proximal y distal a ignorar al usar "shaft_pca".
        shaft_end_trim_bins : int
            Número de bins no vacíos a descartar en cada extremo del tallo.
        slice_count : int
            Número de cortes transversales para detectar diáfisis.
        voxel_size : float
            Tamaño de voxel en mm para neutralizar densidad antes del eje inicial.
        shaft_radius_quantile : float
            Cuantil de radio transversal usado para conservar cortes tipo tallo.
        complete_length_threshold : float
            Longitud proyectada mínima para tratar el húmero como completo.
        crop_fraction : float
            Fracción proximal/distal a descartar. En modelos incompletos solo
            se descarta la fracción proximal de cabeza.
        slice_spike_ratio : float
            Factor relativo para descartar slices con área/perímetro anómalos.
        ransac_iterations : int
            Iteraciones del ajuste robusto RANSAC.
        ransac_residual_threshold : float, optional
            Umbral de distancia para inliers RANSAC. Si no se entrega se estima.
        
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
        axis_diagnostics = {}
        if method == "diaphyseal_slice_axis":
            direction, point_on_line, shaft_points, axis_diagnostics = (
                AxisApproximator._compute_axis_diaphyseal_slice(
                    surface_points,
                    slice_count=slice_count,
                    voxel_size=voxel_size,
                    shaft_radius_quantile=shaft_radius_quantile,
                    shaft_end_trim_bins=shaft_end_trim_bins,
                    complete_length_threshold=complete_length_threshold,
                    crop_fraction=crop_fraction,
                    slice_spike_ratio=slice_spike_ratio,
                    ransac_iterations=ransac_iterations,
                    ransac_residual_threshold=ransac_residual_threshold,
                )
            )
        elif method == "shaft_pca":
            direction, point_on_line, shaft_points = AxisApproximator._compute_axis_shaft_pca(
                surface_points,
                trim_fraction=shaft_trim_fraction,
            )
        elif method == "pca":
            direction, point_on_line = AxisApproximator._compute_axis_pca(surface_points)
        elif method == "least_squares_line":
            direction, point_on_line = AxisApproximator._compute_axis_least_squares(surface_points)
        else:
            raise ValueError(
                "method debe ser 'diaphyseal_slice_axis', 'shaft_pca', "
                "'pca' o 'least_squares_line'"
            )

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
        axis.update(axis_diagnostics)
        axis["validation"] = AxisApproximator.validate_axis(axis, surface_points)
        return axis

    @staticmethod
    def _compute_axis_diaphyseal_slice(
        surface_points: np.ndarray,
        slice_count: int = 28,
        voxel_size: float = 3.0,
        shaft_radius_quantile: float = 0.60,
        shaft_end_trim_bins: int = 2,
        complete_length_threshold: float = 240.0,
        crop_fraction: float = 0.20,
        slice_spike_ratio: float = 2.5,
        ransac_iterations: int = 128,
        ransac_residual_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Calcula el eje desde cortes transversales de la diáfisis.

        Esta variante no usa la esfera. Sigue el flujo:
        1. PCA global solo para orientar el eje superior-inferior.
        2. Clasificación completo/incompleto por longitud proyectada.
        3. Crop proximal/distal si está completo, o solo proximal si está incompleto.
        4. Slices en el ROI, filtrando spikes de área/perímetro.
        5. RANSAC sobre centros de slices retenidos.
        """
        points = np.asarray(surface_points, dtype=float)
        neutral_points = AxisApproximator._voxel_downsample(points, voxel_size)
        if len(neutral_points) < 8:
            neutral_points = points

        rough_direction, rough_centroid = AxisApproximator._compute_axis_pca(neutral_points)
        head_position = AxisApproximator.find_head_position(neutral_points)
        if np.dot(head_position - rough_centroid, rough_direction) < 0:
            rough_direction = -rough_direction

        projections = (neutral_points - rough_centroid) @ rough_direction
        projected_length = float(projections.max() - projections.min())
        is_complete = bool(projected_length >= complete_length_threshold)
        crop = float(np.clip(crop_fraction, 0.0, 0.45))

        low_q = float(np.quantile(projections, crop)) if is_complete else float(projections.min())
        high_q = float(np.quantile(projections, 1.0 - crop))
        roi_mask = (projections >= low_q) & (projections <= high_q)
        roi_points = neutral_points[roi_mask]
        roi_projections = projections[roi_mask]
        if len(roi_points) < max(12, int(0.05 * len(neutral_points))):
            roi_points = neutral_points
            roi_projections = projections
            low_q = float(projections.min())
            high_q = float(projections.max())

        bin_count = max(10, int(slice_count))
        edges = np.linspace(float(roi_projections.min()), float(roi_projections.max()), bin_count + 1)

        min_slice_points = max(6, int(0.003 * len(roi_points)))
        slice_centroids = []
        slice_points = []
        slice_indices = []
        slice_radii = []
        slice_areas = []
        slice_perimeters = []

        for idx in range(bin_count):
            if idx == bin_count - 1:
                mask = (roi_projections >= edges[idx]) & (roi_projections <= edges[idx + 1])
            else:
                mask = (roi_projections >= edges[idx]) & (roi_projections < edges[idx + 1])
            pts = roi_points[mask]
            if len(pts) < min_slice_points:
                continue

            centroid = pts.mean(axis=0)
            vecs = pts - centroid
            axial = np.outer(vecs @ rough_direction, rough_direction)
            transverse = vecs - axial
            radial_distances = np.linalg.norm(transverse, axis=1)
            radius = float(np.percentile(radial_distances, 90))
            transverse_2d = AxisApproximator._project_to_transverse_2d(transverse, rough_direction)
            area, perimeter = AxisApproximator._cross_section_area_perimeter(transverse_2d)

            slice_centroids.append(centroid)
            slice_points.append(pts)
            slice_indices.append(idx)
            slice_radii.append(radius)
            slice_areas.append(area)
            slice_perimeters.append(perimeter)

        if len(slice_centroids) < 4:
            direction, centroid, shaft_points = AxisApproximator._compute_axis_shaft_pca(neutral_points)
            return direction, centroid, shaft_points, {
                "axis_fit_strategy": "fallback_shaft_pca",
                "density_neutral_point_count": int(len(neutral_points)),
                "projected_length": projected_length,
                "is_complete_humerus": is_complete,
                "slice_count": int(bin_count),
            }

        radii = np.asarray(slice_radii, dtype=float)
        areas = np.asarray(slice_areas, dtype=float)
        perimeters = np.asarray(slice_perimeters, dtype=float)
        radius_threshold = float(np.quantile(radii, np.clip(shaft_radius_quantile, 0.1, 0.9)))
        area_threshold = AxisApproximator._spike_threshold(areas, slice_spike_ratio)
        perimeter_threshold = AxisApproximator._spike_threshold(perimeters, slice_spike_ratio)
        candidate = (areas <= area_threshold) & (perimeters <= perimeter_threshold)
        segments = AxisApproximator._continuous_shaft_segments(
            np.asarray(slice_centroids, dtype=float),
            np.asarray(slice_indices, dtype=int),
            candidate,
            radii,
        )

        if not segments:
            segments = [(0, len(slice_centroids))]

        best_start, best_stop = max(
            segments,
            key=lambda segment: (
                segment[1] - segment[0],
                -float(np.mean(radii[segment[0]:segment[1]])),
            ),
        )
        trim_bins = min(
            max(0, int(shaft_end_trim_bins)),
            max(0, (best_stop - best_start - 2) // 2),
        )
        start = best_start + trim_bins
        stop = best_stop - trim_bins
        if stop - start < 2:
            start, stop = best_start, best_stop

        selected = np.zeros(len(slice_centroids), dtype=bool)
        selected[start:stop] = candidate[start:stop]
        if np.count_nonzero(selected) < 2:
            selected[start:stop] = True

        retained_centroids = np.asarray(slice_centroids, dtype=float)[selected]
        retained_points = np.vstack([pts for pts, keep in zip(slice_points, selected) if keep])

        direction, centroid, inlier_mask, residual_threshold = AxisApproximator._ransac_line_fit(
            retained_centroids,
            iterations=ransac_iterations,
            residual_threshold=ransac_residual_threshold,
        )
        if np.dot(direction, rough_direction) < 0:
            direction = -direction
        inlier_centroids = retained_centroids[inlier_mask]
        if len(inlier_centroids) >= 2:
            centroid = inlier_centroids.mean(axis=0)

        return direction, centroid, retained_points, {
            "axis_fit_strategy": "rough_pca_crop_slice_filter_ransac",
            "density_neutral_point_count": int(len(neutral_points)),
            "projected_length": projected_length,
            "complete_length_threshold": float(complete_length_threshold),
            "is_complete_humerus": is_complete,
            "crop_fraction": crop,
            "crop_mode": "head_and_tail" if is_complete else "head_only",
            "crop_projection_range": [float(low_q), float(high_q)],
            "roi_point_count": int(len(roi_points)),
            "slice_count": int(bin_count),
            "slice_valid_count": int(len(slice_centroids)),
            "shaft_radius_threshold": float(radius_threshold),
            "slice_area_threshold": float(area_threshold),
            "slice_perimeter_threshold": float(perimeter_threshold),
            "spike_filtered_slice_count": int(np.count_nonzero(candidate)),
            "shaft_segment_start_slice": int(slice_indices[start]),
            "shaft_segment_stop_slice": int(slice_indices[stop - 1]),
            "shaft_retained_slice_count": int(len(retained_centroids)),
            "axis_fit_centerline_point_count": int(len(retained_centroids)),
            "ransac_inlier_count": int(np.count_nonzero(inlier_mask)),
            "ransac_outlier_count": int(len(inlier_mask) - np.count_nonzero(inlier_mask)),
            "ransac_residual_threshold": float(residual_threshold),
        }

    @staticmethod
    def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
        """Reduce densidad conservando un punto representativo por voxel."""
        if voxel_size <= 0:
            return points
        shifted = points - points.min(axis=0)
        keys = np.floor(shifted / float(voxel_size)).astype(np.int64)
        _, unique_indices = np.unique(keys, axis=0, return_index=True)
        return points[np.sort(unique_indices)]

    @staticmethod
    def _project_to_transverse_2d(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
        """Proyecta puntos transversales 3D a coordenadas 2D del plano de corte."""
        axis = axis / np.linalg.norm(axis)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, axis)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        basis_u = helper - np.dot(helper, axis) * axis
        basis_u = basis_u / np.linalg.norm(basis_u)
        basis_v = np.cross(axis, basis_u)
        basis_v = basis_v / np.linalg.norm(basis_v)
        return np.column_stack((points @ basis_u, points @ basis_v))

    @staticmethod
    def _cross_section_area_perimeter(points_2d: np.ndarray) -> Tuple[float, float]:
        """Estima área y perímetro de una sección con casco convexo 2D."""
        points_2d = np.asarray(points_2d, dtype=float)
        if len(points_2d) < 3:
            return 0.0, 0.0
        try:
            hull = ConvexHull(points_2d)
            return float(hull.volume), float(hull.area)
        except QhullError:
            mins = points_2d.min(axis=0)
            maxs = points_2d.max(axis=0)
            width, height = maxs - mins
            return float(width * height), float(2.0 * (width + height))

    @staticmethod
    def _spike_threshold(values: np.ndarray, ratio: float) -> float:
        """Umbral robusto para descartar spikes altos de área/perímetro."""
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return float("inf")
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        robust = median + 6.0 * (1.4826 * mad if mad > 0 else max(median, 1.0))
        relative = median * max(float(ratio), 1.0)
        return float(max(robust, relative))

    @staticmethod
    def _ransac_line_fit(
        points: np.ndarray,
        iterations: int = 128,
        residual_threshold: Optional[float] = None,
        random_seed: int = 13,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Ajusta una recta 3D con RANSAC y refina con PCA sobre inliers."""
        points = np.asarray(points, dtype=float)
        if len(points) < 2:
            raise ValueError("Se requieren al menos 2 puntos para RANSAC")
        if len(points) == 2:
            direction = points[1] - points[0]
            norm = np.linalg.norm(direction)
            if norm <= 1e-12:
                direction = np.array([0.0, 0.0, 1.0])
            else:
                direction = direction / norm
            return direction, points.mean(axis=0), np.ones(len(points), dtype=bool), 0.0

        initial_direction, initial_point = AxisApproximator._compute_axis_pca(points)
        initial_distances = AxisApproximator._distances_to_line(points, initial_point, initial_direction)
        if residual_threshold is None:
            median = float(np.median(initial_distances))
            mad = float(np.median(np.abs(initial_distances - median)))
            residual_threshold = max(2.0, median + 2.5 * (1.4826 * mad if mad > 0 else median))

        rng = np.random.default_rng(random_seed)
        best_inliers = initial_distances <= residual_threshold
        best_score = (int(np.count_nonzero(best_inliers)), -float(initial_distances[best_inliers].mean() if np.any(best_inliers) else np.inf))

        max_iterations = max(1, int(iterations))
        for _ in range(max_iterations):
            i, j = rng.choice(len(points), size=2, replace=False)
            direction = points[j] - points[i]
            norm = np.linalg.norm(direction)
            if norm <= 1e-12:
                continue
            direction = direction / norm
            distances = AxisApproximator._distances_to_line(points, points[i], direction)
            inliers = distances <= residual_threshold
            if np.count_nonzero(inliers) < 2:
                continue
            mean_distance = float(distances[inliers].mean())
            score = (int(np.count_nonzero(inliers)), -mean_distance)
            if score > best_score:
                best_score = score
                best_inliers = inliers

        if np.count_nonzero(best_inliers) < 2:
            best_inliers = np.ones(len(points), dtype=bool)
        direction, centroid = AxisApproximator._compute_axis_pca(points[best_inliers])
        return direction, centroid, best_inliers, float(residual_threshold)

    @staticmethod
    def _distances_to_line(points: np.ndarray, point_on_line: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Distancias perpendiculares de puntos a una recta 3D."""
        direction = direction / np.linalg.norm(direction)
        vecs = points - point_on_line
        projections = np.outer(vecs @ direction, direction)
        return np.linalg.norm(vecs - projections, axis=1)

    @staticmethod
    def _continuous_shaft_segments(
        centroids: np.ndarray,
        slice_indices: np.ndarray,
        candidate: np.ndarray,
        radii: np.ndarray,
    ) -> list:
        """Agrupa cortes candidatos y corta discontinuidades geométricas."""
        if len(centroids) == 0:
            return []

        jumps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        median_jump = float(np.median(jumps)) if len(jumps) else 0.0
        median_radius = float(np.median(radii)) if len(radii) else 0.0
        jump_threshold = max(3.0 * median_jump, 1.5 * median_radius, 8.0)

        segments = []
        start = None
        previous_valid = None
        for idx, is_candidate in enumerate(candidate):
            if not is_candidate:
                if start is not None:
                    segments.append((start, idx))
                    start = None
                previous_valid = None
                continue

            discontinuous = (
                previous_valid is not None
                and (
                    slice_indices[idx] != slice_indices[previous_valid] + 1
                    or np.linalg.norm(centroids[idx] - centroids[previous_valid]) > jump_threshold
                )
            )
            if start is None or discontinuous:
                if start is not None:
                    segments.append((start, idx))
                start = idx
            previous_valid = idx

        if start is not None:
            segments.append((start, len(candidate)))

        return [segment for segment in segments if segment[1] - segment[0] >= 2]

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
