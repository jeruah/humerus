"""Carga de archivos STL ASCII y binarios basada en trimesh.

Se usa ``trimesh`` para el parsing porque maneja de forma robusta la
detección ASCII/binario, vértices duplicados y consistencia de winding.
La API pública de la clase ``STLLoader`` se conserva para no romper a los
consumidores del pipeline (eje, RANSAC, demos y tests).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class STLMesh:
    """Estructura de datos para malla STL."""

    vertices: np.ndarray  # shape: (N, 3)
    faces: np.ndarray     # shape: (M, 3), índices de vértices
    normals: np.ndarray   # shape: (M, 3), normales de facetas


class STLLoader:
    """Cargador de archivos STL (ASCII y binarios) con trimesh."""

    @staticmethod
    def load(filepath: str) -> STLMesh:
        """
        Carga archivo STL (detecta automáticamente el formato).

        Parameters
        ----------
        filepath : str
            Ruta al archivo STL

        Returns
        -------
        STLMesh
            Estructura con vértices, facetas y normales de faceta

        Raises
        ------
        FileNotFoundError
            Si el archivo no existe
        ValueError
            Si el formato no es un STL válido

        Examples
        --------
        >>> mesh = STLLoader.load("humerus.stl")
        >>> print(f"Vértices: {mesh.vertices.shape}")
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(filepath)
        # process=True deduplica vértices, elimina caras degeneradas y hace
        # consistente el winding/normales (evita normales invertidas).
        mesh = trimesh.load(str(path), force="mesh", process=True)
        return STLLoader._from_trimesh(mesh)

    @staticmethod
    def _looks_binary(path: Path) -> bool:
        """Detecta STL binario comparando tamaño esperado con el conteo de facetas."""
        size = path.stat().st_size
        if size < 84:
            return False
        with path.open("rb") as fh:
            header = fh.read(80)
            count_bytes = fh.read(4)
        if len(count_bytes) != 4:
            return False
        facet_count = np.frombuffer(count_bytes, dtype="<u4")[0]
        expected_size = 84 + int(facet_count) * 50
        if expected_size == size:
            return True
        return b"\0" in header

    @staticmethod
    def load_ascii(filepath: str) -> STLMesh:
        """Carga un STL en formato ASCII (rechaza binario)."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(filepath)
        if STLLoader._looks_binary(path):
            raise ValueError("El archivo es STL binario y se solicitó formato ASCII")
        return STLLoader.load(filepath)

    @staticmethod
    def load_binary(filepath: str) -> STLMesh:
        """Carga un STL en formato binario (rechaza ASCII)."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(filepath)
        if not STLLoader._looks_binary(path):
            raise ValueError("El archivo es STL ASCII y se solicitó formato binario")
        return STLLoader.load(filepath)

    @staticmethod
    def _from_trimesh(mesh: trimesh.Trimesh) -> STLMesh:
        """Convierte una malla trimesh a la estructura STLMesh del proyecto."""
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError("Archivo STL inválido o sin facetas")
        normals = np.asarray(mesh.face_normals, dtype=float)
        if len(normals) != len(faces):
            normals = STLLoader.compute_normals(vertices, faces)
        return STLMesh(vertices=vertices, faces=faces, normals=normals)

    @staticmethod
    def compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Calcula normales de cada faceta.

        Parameters
        ----------
        vertices : np.ndarray
            Vértices de la malla (shape: (N, 3))
        faces : np.ndarray
            Índices de facetas (shape: (M, 3))

        Returns
        -------
        np.ndarray
            Normales unitarias (shape: (M, 3))
        """
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=int)
        triangles = vertices[faces]
        edges_1 = triangles[:, 1] - triangles[:, 0]
        edges_2 = triangles[:, 2] - triangles[:, 0]
        normals = np.cross(edges_1, edges_2)
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        result = np.zeros_like(normals, dtype=float)
        result[valid] = normals[valid] / lengths[valid, None]
        return result
