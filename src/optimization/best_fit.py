"""Búsqueda heurística de best-fit sphere para cabeza humeral."""

from typing import Any, Dict, List, Optional

import numpy as np

from ..approximation.sphere import SphericalApproximator
from ..audit.trail import AuditTrail
from ..axis.longitudinal import AxisApproximator


class HumeralHeadBestFitSearch:
    """
    Busca automáticamente esferas candidatas y las ordena por una función de costo.

    La puntuación combina ajuste local, plausibilidad morfológica poblacional y
    cobertura de puntos de la cabeza sobre la superficie de la esfera.
    """

    def __init__(
        self,
        n_seeds: int = 80,
        top_k: int = 5,
        initial_radius: float = 22.5,
        max_error: float = 2.0,
        random_seed: int = 11,
        surface_tolerance: Optional[float] = None,
        proximal_fraction: float = 0.35,
    ):
        self.n_seeds = int(n_seeds)
        self.top_k = int(top_k)
        self.initial_radius = float(initial_radius)
        self.max_error = float(max_error)
        self.random_seed = int(random_seed)
        self.surface_tolerance = surface_tolerance
        self.proximal_fraction = float(proximal_fraction)
        self.approximator = SphericalApproximator(max_iterations=30, convergence_threshold=1e-5)

    def search(
        self,
        surface_points: np.ndarray,
        surface_normals: np.ndarray,
        axis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ejecuta búsqueda automática y retorna ranking de esferas."""
        points = np.asarray(surface_points, dtype=float)
        normals = np.asarray(surface_normals, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("surface_points debe tener shape (N, 3)")
        if normals.shape != points.shape:
            raise ValueError("surface_normals debe tener shape (N, 3)")

        axis = axis or AxisApproximator.compute_longitudinal_axis(points)
        head_region_indices, head_side = self._head_region_indices(points, axis)
        seed_indices = self._select_seed_indices(points, axis, head_region_indices)

        candidates: List[Dict[str, Any]] = []
        for rank_seed, point_index in enumerate(seed_indices):
            seed = points[point_index]
            audit = AuditTrail(f"best_fit_seed_{rank_seed:04d}")
            try:
                sphere = self.approximator.approximate_from_seed(
                    seed,
                    points,
                    normals,
                    audit_trail=audit,
                    initial_radius=self.initial_radius,
                )
                audit.is_valid_approximation(
                    sphere,
                    max_error=self.max_error,
                    axis=axis,
                    surface_points=points,
                    medial_direction=np.array([1.0, 0.0, 0.0]),
                    posterior_direction=np.array([0.0, 1.0, 0.0]),
                )
                validation = self._latest_validation(audit)
                score_parts = self._score_candidate(
                    sphere,
                    validation,
                    points,
                    normals,
                    head_region_indices,
                )
                candidates.append({
                    "seed_index": int(point_index),
                    "seed": seed,
                    "sphere": sphere,
                    "valid": bool(audit.get_report()["final_valid"]),
                    "audit": audit.get_report(),
                    **score_parts,
                })
            except Exception as exc:
                candidates.append({
                    "seed_index": int(point_index),
                    "seed": seed,
                    "sphere": None,
                    "valid": False,
                    "score": float("inf"),
                    "error_message": str(exc),
                    "coverage_count": 0,
                    "coverage_ratio": 0.0,
                })

        candidates.sort(key=lambda item: item["score"])
        return {
            "axis": axis,
            "head_region_count": int(len(head_region_indices)),
            "head_side": head_side,
            "seed_indices": [int(index) for index in seed_indices],
            "candidate_count": int(len(candidates)),
            "valid_candidate_count": int(sum(1 for item in candidates if item.get("valid"))),
            "best": candidates[0] if candidates else None,
            "top_candidates": candidates[:max(1, self.top_k)],
            "all_candidates": candidates,
        }

    def _head_region_indices(self, points: np.ndarray, axis: Dict[str, Any]) -> tuple:
        """Selecciona una región proximal amplia donde debe estar la cabeza."""
        origin = np.asarray(axis["origin"], dtype=float)
        distal = np.asarray(axis["distal_point"], dtype=float)
        direction = distal - origin
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            direction = np.asarray(axis["direction"], dtype=float)
            norm = np.linalg.norm(direction)
        direction = direction / norm
        projections = (points - origin) @ direction
        radial_distance = self._axis_radial_distance(points, origin, direction)
        length = float(np.max(projections) - np.min(projections))
        max_head_depth = min(90.0, max(35.0, self.proximal_fraction * float(axis["length"])))

        low_limit = float(np.min(projections))
        high_limit = float(np.max(projections))
        end_depth = min(max_head_depth, max(20.0, 0.25 * length))
        low_end = projections <= low_limit + end_depth
        high_end = projections >= high_limit - end_depth
        low_spread = self._end_radial_spread(radial_distance[low_end])
        high_spread = self._end_radial_spread(radial_distance[high_end])

        if high_spread >= low_spread:
            mask = projections >= high_limit - max_head_depth
            head_side = "high_projection"
        else:
            mask = projections <= low_limit + max_head_depth
            head_side = "low_projection"

        indices = np.where(mask)[0]
        if len(indices) < max(20, int(0.03 * len(points))):
            if head_side == "high_projection":
                cutoff = np.quantile(projections, max(0.6, 1.0 - self.proximal_fraction))
                indices = np.where(projections >= cutoff)[0]
            else:
                cutoff = np.quantile(projections, min(0.4, self.proximal_fraction))
                indices = np.where(projections <= cutoff)[0]
        return indices, head_side

    def _select_seed_indices(
        self,
        points: np.ndarray,
        axis: Dict[str, Any],
        head_region_indices: np.ndarray,
    ) -> np.ndarray:
        """Escoge semillas dispersas en región proximal, priorizando puntos lejos del eje."""
        rng = np.random.default_rng(self.random_seed)
        region_points = points[head_region_indices]
        origin = axis["origin"]
        direction = axis["direction"] / np.linalg.norm(axis["direction"])
        radial_distance = self._axis_radial_distance(region_points, origin, direction)
        threshold = np.quantile(radial_distance, 0.65)
        candidate_indices = head_region_indices[radial_distance >= threshold]
        if len(candidate_indices) == 0:
            candidate_indices = head_region_indices

        n = min(self.n_seeds, len(candidate_indices))
        selected = rng.choice(candidate_indices, size=n, replace=False)
        return np.asarray(selected, dtype=int)

    @staticmethod
    def _axis_radial_distance(points: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Distancia perpendicular de puntos a una recta de eje."""
        vecs = points - origin
        axial = np.outer(vecs @ direction, direction)
        return np.linalg.norm(vecs - axial, axis=1)

    @staticmethod
    def _end_radial_spread(radial_distance: np.ndarray) -> float:
        """Mide robustamente que tan expandido es un extremo del hueso."""
        if len(radial_distance) == 0:
            return 0.0
        return float(np.percentile(radial_distance, 90))

    def _score_candidate(
        self,
        sphere: Dict[str, Any],
        validation: Dict[str, Any],
        points: np.ndarray,
        normals: np.ndarray,
        head_region_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """Calcula costo combinado para un candidato."""
        coverage = self._surface_coverage(sphere, points, normals, head_region_indices)
        reference_values = validation.get("morphology_reference_values", {})
        z_scores = {
            key: abs(float(value.get("z_score", 0.0)))
            for key, value in reference_values.items()
        }
        morphology_penalty = float(np.mean([min(score, 4.0) ** 2 for score in z_scores.values()])) if z_scores else 0.0
        rmse_norm = float(sphere["error"]) / max(self.max_error, 1e-6)
        coverage_penalty = 1.0 - coverage["coverage_ratio"]
        convergence_penalty = 0.0 if sphere.get("converged", False) else 0.15
        reference_penalty = 0.0 if validation.get("morphology_reference_flags", {}).get("all_in_reference", False) else 0.25

        score = (
            0.40 * rmse_norm
            + 0.25 * morphology_penalty
            + 0.30 * coverage_penalty
            + convergence_penalty
            + reference_penalty
        )

        return {
            "score": float(score),
            "score_components": {
                "rmse_norm": rmse_norm,
                "morphology_penalty": morphology_penalty,
                "coverage_penalty": coverage_penalty,
                "convergence_penalty": convergence_penalty,
                "reference_penalty": reference_penalty,
            },
            "coverage_count": coverage["coverage_count"],
            "coverage_ratio": coverage["coverage_ratio"],
            "coverage_tolerance": coverage["coverage_tolerance"],
            "morphology_z_scores": z_scores,
            "morphology_reference_flags": validation.get("morphology_reference_flags", {}),
            "morphology_reference_values": reference_values,
        }

    def _surface_coverage(
        self,
        sphere: Dict[str, Any],
        points: np.ndarray,
        normals: np.ndarray,
        head_region_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """Cuenta puntos proximales compatibles con la superficie de la esfera."""
        center = np.asarray(sphere["center"], dtype=float)
        radius = float(sphere["radius"])
        head_points = points[head_region_indices]
        head_normals = normals[head_region_indices]
        distances = np.linalg.norm(head_points - center, axis=1)
        tolerance = (
            float(self.surface_tolerance)
            if self.surface_tolerance is not None
            else max(1.25, 0.06 * radius)
        )
        shell_mask = np.abs(distances - radius) <= tolerance

        radial = head_points - center
        radial_norm = np.linalg.norm(radial, axis=1)
        normal_norm = np.linalg.norm(head_normals, axis=1)
        valid = (radial_norm > 1e-12) & (normal_norm > 1e-12)
        normal_mask = np.ones(len(head_points), dtype=bool)
        normal_mask[valid] = np.sum(
            (radial[valid] / radial_norm[valid, None])
            * (head_normals[valid] / normal_norm[valid, None]),
            axis=1,
        ) >= 0.15

        coverage_mask = shell_mask & normal_mask
        coverage_count = int(np.count_nonzero(coverage_mask))
        denominator = max(1, len(head_region_indices))
        return {
            "coverage_count": coverage_count,
            "coverage_ratio": float(coverage_count / denominator),
            "coverage_tolerance": float(tolerance),
        }

    @staticmethod
    def _latest_validation(audit: AuditTrail) -> Dict[str, Any]:
        """Obtiene los datos del último paso validate_approximation."""
        for step in reversed(audit.get_report()["steps"]):
            if step["step_name"] == "validate_approximation":
                return step["data"]
        return {}
