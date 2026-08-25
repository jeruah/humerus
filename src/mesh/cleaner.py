"""Limpieza topológica de mallas triangulares basada en trimesh.

Se usa ``trimesh`` para deduplicar vértices, eliminar caras degeneradas y
duplicadas y — lo más importante — normalizar el winding para que las
normales queden consistentes (outward). Esto evita que normales invertidas
rompan silenciosamente el RANSAC de esfera o el lado articular.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh


@dataclass
class CleanedMesh:
    """Malla limpia con métricas de triángulo y conectividad."""

    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    face_centroids: np.ndarray
    adjacency: list[list[int]]
    cleaning_report: dict[str, Any]


class MeshCleaner:
    """Prepara una malla STL para algoritmos basados en conectividad."""

    def __init__(
        self,
        vertex_precision: int = 6,
        min_area: float = 1e-8,
        keep_largest_component: bool = True,
    ):
        # vertex_precision se conserva para compatibilidad de API: el dedup
        # de vértices lo resuelve trimesh (merge dentro de tol.merge).
        self.vertex_precision = int(vertex_precision)
        self.min_area = float(min_area)
        self.keep_largest_component = bool(keep_largest_component)

    def clean(self, vertices: np.ndarray, faces: np.ndarray) -> CleanedMesh:
        """Elimina degenerados, deduplica vértices y construye adyacencia."""
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=int)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices debe tener shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces debe tener shape (M, 3)")

        original_vertex_count = len(vertices)
        original_face_count = len(faces)
        finite_vertex_mask = np.all(np.isfinite(vertices), axis=1)
        valid_face_mask = np.all(finite_vertex_mask[faces], axis=1)
        removed_nonfinite_faces = int(original_face_count - np.count_nonzero(valid_face_mask))
        vertices = vertices[finite_vertex_mask]
        if np.count_nonzero(finite_vertex_mask) != original_vertex_count:
            remap = -np.ones(original_vertex_count, dtype=int)
            remap[np.where(finite_vertex_mask)[0]] = np.arange(len(vertices))
            faces = remap[faces[valid_face_mask]]
        else:
            faces = faces[valid_face_mask]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        areas = np.asarray(mesh.area_faces, dtype=float)
        nondegenerate = np.isfinite(areas) & (areas > self.min_area)
        removed_degenerate_faces = int(np.count_nonzero(~nondegenerate))
        if not np.any(nondegenerate):
            raise ValueError("La malla no conserva triángulos válidos después de limpieza")

        mesh.update_faces(nondegenerate)
        # Dedup de vértices, caras duplicadas y fijado de winding/normales.
        mesh.process(validate=True)

        adjacency = MeshCleaner.build_adjacency(np.asarray(mesh.faces, dtype=int))
        component_labels, component_sizes = MeshCleaner.connected_components(adjacency)
        kept_component_count = len(component_sizes)
        removed_small_component_faces = 0

        if self.keep_largest_component and len(component_sizes) > 1:
            largest = int(np.argmax(component_sizes))
            keep = component_labels == largest
            removed_small_component_faces = int(np.count_nonzero(~keep))
            mesh.update_faces(keep)
            mesh.remove_unreferenced_vertices()
            adjacency = MeshCleaner.build_adjacency(np.asarray(mesh.faces, dtype=int))
            component_labels, component_sizes = MeshCleaner.connected_components(adjacency)

        report = {
            "original_vertex_count": int(original_vertex_count),
            "original_face_count": int(original_face_count),
            "clean_vertex_count": len(mesh.vertices),
            "clean_face_count": len(mesh.faces),
            "removed_nonfinite_faces": removed_nonfinite_faces,
            "removed_degenerate_faces": removed_degenerate_faces,
            "component_count_before_filter": kept_component_count,
            "component_count": len(component_sizes),
            "removed_small_component_faces": removed_small_component_faces,
            "total_area": float(np.sum(np.asarray(mesh.area_faces, dtype=float))),
        }

        return CleanedMesh(
            vertices=np.asarray(mesh.vertices, dtype=float),
            faces=np.asarray(mesh.faces, dtype=int),
            face_normals=np.asarray(mesh.face_normals, dtype=float),
            face_areas=np.asarray(mesh.area_faces, dtype=float),
            face_centroids=np.asarray(mesh.triangles_center, dtype=float),
            adjacency=adjacency,
            cleaning_report=report,
        )

    def vertex_normals(self, mesh: CleanedMesh) -> np.ndarray:
        """Calcula normales por vértice ponderadas por área de cara."""
        normals = np.zeros_like(mesh.vertices, dtype=float)
        weighted = mesh.face_normals * mesh.face_areas[:, None]
        for face, normal in zip(mesh.faces, weighted):
            normals[face] += normal
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        normals[valid] /= lengths[valid, None]
        return normals

    @staticmethod
    def build_adjacency(faces: np.ndarray) -> list[list[int]]:
        """Construye adyacencia de triángulos por aristas compartidas."""
        faces = np.asarray(faces, dtype=int)
        edge_to_faces: dict[tuple, list[int]] = {}
        for face_index, face in enumerate(faces):
            edges = (
                tuple(sorted((int(face[0]), int(face[1])))),
                tuple(sorted((int(face[1]), int(face[2])))),
                tuple(sorted((int(face[2]), int(face[0])))),
            )
            for edge in edges:
                edge_to_faces.setdefault(edge, []).append(face_index)

        adjacency: list[list[int]] = [[] for _ in range(len(faces))]
        for owners in edge_to_faces.values():
            if len(owners) < 2:
                continue
            for owner in owners:
                adjacency[owner].extend(other for other in owners if other != owner)
        return [sorted(set(neighbors)) for neighbors in adjacency]

    @staticmethod
    def connected_components(adjacency: list[list[int]]) -> tuple:
        """Etiqueta componentes conectados de una lista de adyacencia."""
        labels = -np.ones(len(adjacency), dtype=int)
        sizes = []
        label = 0
        for start in range(len(adjacency)):
            if labels[start] >= 0:
                continue
            stack = [start]
            labels[start] = label
            size = 0
            while stack:
                current = stack.pop()
                size += 1
                for neighbor in adjacency[current]:
                    if labels[neighbor] < 0:
                        labels[neighbor] = label
                        stack.append(neighbor)
            sizes.append(size)
            label += 1
        return labels, np.asarray(sizes, dtype=int)

    @staticmethod
    def _compact_vertices(vertices: np.ndarray, faces: np.ndarray) -> tuple:
        """Descarta vértices no usados y remapea caras (legacy)."""
        used = np.unique(faces)
        remap = -np.ones(len(vertices), dtype=int)
        remap[used] = np.arange(len(used))
        return vertices[used], remap[faces]
