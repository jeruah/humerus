"""Discretización de superficies de mallas 3D."""

import numpy as np
from typing import Tuple, Optional


class MeshDiscretizer:
    """
    Discretiza superficies de mallas 3D (STL).
    
    Implementar:
    - Muestreo uniforme de superficie
    - Muestreo adaptativo (más puntos en regiones curvas)
    - Cálculo de normales en puntos muestreados
    """
    
    def discretize_uniform(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        n_samples: int = 5000,
        random_seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Muestreo uniforme de la superficie.
        
        Parameters
        ----------
        vertices : np.ndarray
            Vértices de la malla (shape: (N, 3))
        faces : np.ndarray
            Facetas (índices) (shape: (M, 3))
        n_samples : int
            Número de puntos a samplear
        random_seed : int, optional
            Semilla para reproducibilidad
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (sampled_points, sampled_normals)
            sampled_points: shape (n_samples, 3)
            sampled_normals: shape (n_samples, 3)
        
        Notes
        -----
        Usar área de faceta como peso para muestreo uniforme en superficie.
        """
        rng = np.random.default_rng(random_seed)
        areas, face_normals = self._face_areas_and_normals(vertices, faces)
        total_area = areas.sum()
        if total_area <= 0:
            raise ValueError("La malla no tiene área superficial positiva")

        probabilities = areas / total_area
        face_idx = rng.choice(len(faces), size=n_samples, p=probabilities)
        triangles = vertices[faces[face_idx]]

        r1 = rng.random(n_samples)
        r2 = rng.random(n_samples)
        sqrt_r1 = np.sqrt(r1)
        weights = np.column_stack((1.0 - sqrt_r1, sqrt_r1 * (1.0 - r2), sqrt_r1 * r2))
        sampled_points = np.einsum("ij,ijk->ik", weights, triangles)
        sampled_normals = face_normals[face_idx]

        return sampled_points, sampled_normals
    
    def discretize_adaptive(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        target_samples: int = 5000,
        curvature_data: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Muestreo adaptativo (más denso en regiones curvas).
        
        Regiones con mayor curvatura reciben más puntos.
        
        Parameters
        ----------
        vertices : np.ndarray
            Vértices de la malla
        faces : np.ndarray
            Facetas
        target_samples : int
            Número objetivo de puntos
        curvature_data : np.ndarray, optional
            Datos de curvatura por faceta
        random_seed : int, optional
            Semilla para reproducibilidad
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (sampled_points, sampled_normals)
        
        Notes
        -----
        Si curvature_data no se proporciona, usar curvatura de área o estimada.
        """
        if curvature_data is None:
            return self.discretize_uniform(vertices, faces, target_samples, random_seed)

        rng = np.random.default_rng(random_seed)
        areas, face_normals = self._face_areas_and_normals(vertices, faces)
        weights = np.asarray(curvature_data, dtype=float)
        if weights.ndim > 1:
            weights = np.linalg.norm(weights, axis=1)
        if len(weights) != len(faces):
            raise ValueError("curvature_data debe tener un valor por faceta")

        weights = np.maximum(weights, 0.0)
        weights = areas * (1.0 + weights / max(weights.max(), 1e-12))
        probabilities = weights / weights.sum()
        face_idx = rng.choice(len(faces), size=target_samples, p=probabilities)
        triangles = vertices[faces[face_idx]]

        r1 = rng.random(target_samples)
        r2 = rng.random(target_samples)
        sqrt_r1 = np.sqrt(r1)
        bary = np.column_stack((1.0 - sqrt_r1, sqrt_r1 * (1.0 - r2), sqrt_r1 * r2))
        sampled_points = np.einsum("ij,ijk->ik", bary, triangles)

        return sampled_points, face_normals[face_idx]
    
    def extract_articulation_region(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        center_estimate: Optional[np.ndarray] = None,
        radius_search: float = 40.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae región articular de la malla.
        
        Parameters
        ----------
        vertices : np.ndarray
            Vértices de la malla
        faces : np.ndarray
            Facetas
        center_estimate : np.ndarray, optional
            Estimación del centro de cabeza (shape: (3,))
        radius_search : float
            Radio de búsqueda alrededor del centro (mm)
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (articulation_vertices, articulation_faces)
        """
        if center_estimate is None:
            center_estimate = vertices.mean(axis=0)

        distances = np.linalg.norm(vertices - center_estimate, axis=1)
        vertex_mask = distances <= radius_search
        face_mask = np.any(vertex_mask[faces], axis=1)
        selected_faces = faces[face_mask]

        if len(selected_faces) == 0:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)

        used_vertices = np.unique(selected_faces)
        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        remapped_faces = np.vectorize(remap.get)(selected_faces)

        return vertices[used_vertices], remapped_faces.astype(int)

    @staticmethod
    def _face_areas_and_normals(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula área y normal unitaria por triángulo."""
        triangles = vertices[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        areas = 0.5 * lengths
        normals = np.zeros_like(cross, dtype=float)
        valid = lengths > 1e-12
        normals[valid] = cross[valid] / lengths[valid, None]
        return areas, normals
