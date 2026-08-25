"""Discretización de superficies de mallas 3D basada en trimesh.

El muestreo uniforme y adaptativo usa ``trimesh.sample.sample_surface``, que
reparte puntos proporcionalmente al área de cada triángulo y devuelve el
índice de faceta para recuperar las normales correspondientes.
"""


import numpy as np
import trimesh


class MeshDiscretizer:
    """
    Discretiza superficies de mallas 3D (STL).

    - Muestreo uniforme de superficie.
    - Muestreo adaptativo (más puntos en regiones curvas).
    - Cálculo de normales en puntos muestreados.
    """

    def discretize_uniform(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        n_samples: int = 5000,
        random_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        """
        mesh = self._build_mesh(vertices, faces)
        if mesh.area <= 0:
            raise ValueError("La malla no tiene área superficial positiva")
        points, face_idx = trimesh.sample.sample_surface(mesh, int(n_samples), seed=random_seed)
        normals = np.asarray(mesh.face_normals, dtype=float)[face_idx]
        return np.asarray(points, dtype=float), np.asarray(normals, dtype=float)

    def discretize_adaptive(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        target_samples: int = 5000,
        curvature_data: np.ndarray | None = None,
        random_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Muestreo adaptativo (más denso en regiones curvas).

        Parameters
        ----------
        vertices : np.ndarray
            Vértices de la malla
        faces : np.ndarray
            Facetas
        target_samples : int
            Número objetivo de puntos
        curvature_data : np.ndarray, optional
            Curvatura escalar o vectorial por faceta (shape: (M,) o (M, d))
        random_seed : int, optional
            Semilla para reproducibilidad

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (sampled_points, sampled_normals)

        Notes
        -----
        Si ``curvature_data`` no se proporciona, se usa muestreo uniforme.
        """
        if curvature_data is None:
            return self.discretize_uniform(vertices, faces, target_samples, random_seed)

        mesh = self._build_mesh(vertices, faces)
        weights = np.asarray(curvature_data, dtype=float)
        if weights.ndim > 1:
            weights = np.linalg.norm(weights, axis=1)
        if len(weights) != len(mesh.faces):
            raise ValueError("curvature_data debe tener un valor por faceta")

        weights = np.maximum(weights, 0.0)
        area = np.asarray(mesh.area_faces, dtype=float)
        if weights.max() > 1e-12:
            face_weight = area * (1.0 + weights / weights.max())
        else:
            face_weight = area
        if float(np.sum(face_weight)) <= 0:
            raise ValueError("La malla no tiene área superficial positiva")

        points, face_idx = trimesh.sample.sample_surface(
            mesh, int(target_samples), face_weight=face_weight, seed=random_seed
        )
        normals = np.asarray(mesh.face_normals, dtype=float)[face_idx]
        return np.asarray(points, dtype=float), np.asarray(normals, dtype=float)

    def extract_articulation_region(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        center_estimate: np.ndarray | None = None,
        radius_search: float = 40.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extrae región articular de la malla como subconjunto de vértices/caras.

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
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=int)
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
    def _build_mesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
        """Construye una malla trimesh sin post-procesamiento."""
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=int)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices debe tener shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces debe tener shape (M, 3)")
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
