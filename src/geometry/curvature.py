"""Cálculo de curvatura en mallas triangulares."""

import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree
from .differential import DifferentialAnalyzer


@dataclass
class CurvatureData:
    """Datos de curvatura en un punto."""
    principal_k1: float      # Curvatura principal mayor
    principal_k2: float      # Curvatura principal menor
    mean_curvature: float    # H = (k1 + k2) / 2
    gaussian_curvature: float  # K = k1 * k2
    normal: np.ndarray       # Vector normal unitario


class CurvatureCalculator:
    """
    Calcula curvatura principal, media y Gaussiana en mallas triangulares.
    
    Métodos:
    - Curvatura usando operadores discretos
    - Curvatura usando ajuste local de superficie
    - Identificación de regiones esféricas
    """
    
    @staticmethod
    def compute_vertex_curvature(
        vertex_idx: int,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: np.ndarray
    ) -> CurvatureData:
        """
        Calcula curvatura en un vértice de la malla.
        
        Parameters
        ----------
        vertex_idx : int
            Índice del vértice
        vertices : np.ndarray
            Vértices de la malla (shape: (N, 3))
        faces : np.ndarray
            Facetas (shape: (M, 3))
        normals : np.ndarray
            Normales de la malla (shape: (N, 3))
        
        Returns
        -------
        CurvatureData
            Datos de curvatura (κ₁, κ₂, H, K)
        
        Notes
        -----
        Usar método de Meyer et al. (2003) para operadores discretos,
        o ajuste local de superficie cuadrática.
        
        Referencias:
        - Meyer, M., Desbrun, M., Schröder, P., Barr, A. H. (2003)
          "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds"
        """
        vertex_normal = CurvatureCalculator._vertex_normals(vertices, faces, normals)[vertex_idx]
        neighbor_idx = CurvatureCalculator._vertex_neighbors(vertex_idx, faces)
        if len(neighbor_idx) < 6:
            tree = cKDTree(vertices)
            _, neighbor_idx = tree.query(vertices[vertex_idx], k=min(12, len(vertices)))
            neighbor_idx = np.asarray(neighbor_idx)
            neighbor_idx = neighbor_idx[neighbor_idx != vertex_idx]

        return CurvatureCalculator.compute_point_curvature(
            vertices[vertex_idx],
            vertices[np.asarray(list(neighbor_idx), dtype=int)],
            vertex_normal,
        )
    
    @staticmethod
    def compute_point_curvature(
        point: np.ndarray,
        neighbors: np.ndarray,
        normal: np.ndarray
    ) -> CurvatureData:
        """
        Calcula curvatura en un punto arbitrario usando vecinos.
        
        Parameters
        ----------
        point : np.ndarray
            Punto de interés (shape: (3,))
        neighbors : np.ndarray
            Puntos vecinos (shape: (M, 3))
        normal : np.ndarray
            Normal en el punto (shape: (3,))
        
        Returns
        -------
        CurvatureData
            Datos de curvatura
        
        Notes
        -----
        Usar ajuste de superficie cuadrática local:
        z = a*x² + b*y² + c*xy + d*x + e*y + f
        
        Las curvaturas principales son eigenvalores de la Hessiana.
        """
        hessian, _, _ = DifferentialAnalyzer.fit_quadric_surface(point, neighbors, normal)
        k1, k2, _, _ = DifferentialAnalyzer.compute_principal_curvatures(hessian)
        mean = 0.5 * (k1 + k2)
        gaussian = k1 * k2
        normal = normal / max(np.linalg.norm(normal), 1e-12)
        return CurvatureData(
            principal_k1=float(k1),
            principal_k2=float(k2),
            mean_curvature=float(mean),
            gaussian_curvature=float(gaussian),
            normal=normal,
        )
    
    @staticmethod
    def compute_all_curvatures(
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: np.ndarray
    ) -> np.ndarray:
        """
        Calcula curvatura para todos los vértices.
        
        Parameters
        ----------
        vertices : np.ndarray
            Vértices de la malla
        faces : np.ndarray
            Facetas
        normals : np.ndarray
            Normales de vértices
        
        Returns
        -------
        np.ndarray
            Array con [k1, k2, H, K] para cada vértice
            shape: (N, 4)
        """
        vertex_normals = CurvatureCalculator._vertex_normals(vertices, faces, normals)
        tree = cKDTree(vertices)
        result = np.zeros((len(vertices), 4), dtype=float)

        k = min(16, len(vertices))
        for idx, point in enumerate(vertices):
            _, neighbor_idx = tree.query(point, k=k)
            neighbor_idx = np.asarray(neighbor_idx)
            neighbor_idx = neighbor_idx[neighbor_idx != idx]
            try:
                data = CurvatureCalculator.compute_point_curvature(
                    point, vertices[neighbor_idx], vertex_normals[idx]
                )
                result[idx] = [
                    data.principal_k1,
                    data.principal_k2,
                    data.mean_curvature,
                    data.gaussian_curvature,
                ]
            except (ValueError, np.linalg.LinAlgError):
                result[idx] = np.nan

        return result
    
    @staticmethod
    def compute_point_cloud_curvatures(
        points: np.ndarray,
        normals: np.ndarray,
        k: int = 16
    ) -> np.ndarray:
        """
        Calcula curvatura para una nube de puntos sin topología de malla.

        A diferencia de `compute_all_curvatures` (que usa `faces` para
        vecinos topológicos), aquí los vecinos se buscan por distancia
        dentro de la propia nube de puntos. Útil para puntos muestreados
        con `MeshDiscretizer`, que no son vértices de la malla original.

        Parameters
        ----------
        points : np.ndarray
            Puntos muestreados (shape: (N, 3))
        normals : np.ndarray
            Normal por punto (shape: (N, 3))
        k : int
            Número de vecinos más cercanos a usar (incluyéndose el punto)

        Returns
        -------
        np.ndarray
            Array con [k1, k2, H, K] por punto (shape: (N, 4))
        """
        points = np.asarray(points, dtype=float)
        normals = np.asarray(normals, dtype=float)
        tree = cKDTree(points)
        k = min(k, len(points))
        result = np.zeros((len(points), 4), dtype=float)

        for idx, point in enumerate(points):
            _, neighbor_idx = tree.query(point, k=k)
            neighbor_idx = np.asarray(neighbor_idx)
            neighbor_idx = neighbor_idx[neighbor_idx != idx]
            try:
                data = CurvatureCalculator.compute_point_curvature(
                    point, points[neighbor_idx], normals[idx]
                )
                result[idx] = [
                    data.principal_k1,
                    data.principal_k2,
                    data.mean_curvature,
                    data.gaussian_curvature,
                ]
            except (ValueError, np.linalg.LinAlgError):
                result[idx] = np.nan

        return result

    @staticmethod
    def sphericity_score(
        curvatures: np.ndarray,
        radius_estimate: float = 30.0,
        tolerance: float = 0.15
    ) -> np.ndarray:
        """
        Puntúa qué tan cerca está cada punto de una superficie esférica.

        Combina la misma anisotropía y error de radio que usa
        `find_spherical_region`, pero como score continuo en [0, 1] en
        vez de una máscara booleana: 0 = coincide con la esfera esperada,
        1 = muy alejado (curvatura NaN también se marca como 1).

        Parameters
        ----------
        curvatures : np.ndarray
            Array de curvaturas (shape: (N, 4)), cada fila [k1, k2, H, K]
        radius_estimate : float
            Radio esperado de la esfera (mm)
        tolerance : float
            Anisotropía/error de radio que se considera "límite" (score 1.0)

        Returns
        -------
        np.ndarray
            Score por punto en [0, 1] (shape: (N,))
        """
        expected = 1.0 / radius_estimate
        k1 = np.abs(curvatures[:, 0])
        k2 = np.abs(curvatures[:, 1])
        mean = np.abs(curvatures[:, 2])

        finite = np.all(np.isfinite(curvatures[:, :4]), axis=1)
        anisotropy = np.abs(k1 - k2) / np.maximum(np.maximum(k1, k2), 1e-12)
        mean_error = np.abs(mean - expected) / expected

        score = 0.5 * (anisotropy / tolerance) + 0.5 * (mean_error / tolerance)
        score = np.clip(score, 0.0, 1.0)
        score[~finite] = 1.0
        return score

    @staticmethod
    def find_spherical_region(
        curvatures: np.ndarray,
        vertices: np.ndarray,
        radius_estimate: float = 30.0,
        tolerance: float = 0.15
    ) -> np.ndarray:
        """
        Identifica región donde curvatura es compatible con esfera.
        
        Una región es esférica si κ₁ ≈ κ₂ ≈ 1/R
        
        Parameters
        ----------
        curvatures : np.ndarray
            Array de curvaturas (shape: (N, 4))
            Cada fila: [k1, k2, H, K]
        vertices : np.ndarray
            Vértices correspondientes (shape: (N, 3))
        radius_estimate : float
            Radio esperado de la esfera (mm)
        tolerance : float
            Tolerancia relativa (ej: 0.15 = ±15%)
        
        Returns
        -------
        np.ndarray
            Índices de vértices en región esférica
        
        Notes
        -----
        Criterios de región esférica:
        - |k1 - k2| < tolerance * max(|k1|, |k2|)
        - mean_curvature ≈ 1/radius_estimate ± tolerance
        """
        expected = 1.0 / radius_estimate
        k1 = np.abs(curvatures[:, 0])
        k2 = np.abs(curvatures[:, 1])
        mean = np.abs(curvatures[:, 2])

        finite = np.all(np.isfinite(curvatures[:, :4]), axis=1)
        anisotropy = np.abs(k1 - k2) / np.maximum(np.maximum(k1, k2), 1e-12)
        mean_error = np.abs(mean - expected) / expected
        mask = finite & (anisotropy <= tolerance) & (mean_error <= tolerance)
        return np.where(mask)[0]
    
    @staticmethod
    def is_local_sphere(
        curvature: CurvatureData,
        radius_estimate: float = 30.0,
        tolerance: float = 0.15
    ) -> bool:
        """
        Valida si curvatura local es compatible con esfera.
        
        Parameters
        ----------
        curvature : CurvatureData
            Datos de curvatura
        radius_estimate : float
            Radio esperado
        tolerance : float
            Tolerancia relativa
        
        Returns
        -------
        bool
            True si compatible con esfera
        """
        # Curvatura esperada de esfera
        expected_curvature = 1.0 / radius_estimate
        
        # Verificar que κ₁ ≈ κ₂ (superficie isotrópica)
        k1 = abs(curvature.principal_k1)
        k2 = abs(curvature.principal_k2)
        k_diff = abs(k1 - k2)
        k_avg = max(k1, k2)
        
        if k_avg > 0:
            anisotropy_ratio = k_diff / k_avg
            if anisotropy_ratio > tolerance:
                return False  # No es esférica
        
        # Verificar que curvatura media ≈ 1/R
        curvature_error = abs(
            abs(curvature.mean_curvature) - expected_curvature
        )
        
        if expected_curvature > 0:
            relative_error = curvature_error / expected_curvature
            if relative_error > tolerance:
                return False  # Fuera de rango
        
        return True

    @staticmethod
    def _vertex_neighbors(vertex_idx: int, faces: np.ndarray) -> np.ndarray:
        """Obtiene vecinos topológicos de un vértice."""
        touching = faces[np.any(faces == vertex_idx, axis=1)]
        return np.setdiff1d(np.unique(touching), np.array([vertex_idx]))

    @staticmethod
    def _vertex_normals(vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Promedia normales de facetas para obtener normales por vértice."""
        if normals.shape == vertices.shape:
            lengths = np.linalg.norm(normals, axis=1)
            out = np.zeros_like(normals, dtype=float)
            valid = lengths > 1e-12
            out[valid] = normals[valid] / lengths[valid, None]
            return out

        vertex_normals = np.zeros_like(vertices, dtype=float)
        for face, normal in zip(faces, normals):
            vertex_normals[face] += normal
        lengths = np.linalg.norm(vertex_normals, axis=1)
        valid = lengths > 1e-12
        vertex_normals[valid] /= lengths[valid, None]
        return vertex_normals
