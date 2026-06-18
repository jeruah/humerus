"""Carga de archivos STL ASCII y binarios."""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
import struct


@dataclass
class STLMesh:
    """Estructura de datos para malla STL."""
    vertices: np.ndarray  # shape: (N, 3)
    faces: np.ndarray     # shape: (M, 3), índices de vértices
    normals: np.ndarray   # shape: (M, 3), normales de facetas


class STLLoader:
    """
    Cargador de archivos STL (ASCII y binarios).
    
    Implementar:
    - Carga de STL ASCII
    - Carga de STL binarios
    - Validación de malla
    - Cálculo de normales
    """
    
    @staticmethod
    def load(filepath: str) -> STLMesh:
        """
        Carga archivo STL (detecta automáticamente formato).
        
        Parameters
        ----------
        filepath : str
            Ruta al archivo STL
        
        Returns
        -------
        STLMesh
            Estructura con vértices, facetas y normales
        
        Raises
        ------
        FileNotFoundError
            Si archivo no existe
        ValueError
            Si formato no es válido
        
        Examples
        --------
        >>> mesh = STLLoader.load("humerus.stl")
        >>> print(f"Vértices: {mesh.vertices.shape}")
        >>> print(f"Facetas: {mesh.faces.shape}")
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(filepath)

        if STLLoader._looks_binary(path):
            return STLLoader.load_binary(filepath)
        return STLLoader.load_ascii(filepath)

    @staticmethod
    def _looks_binary(path: Path) -> bool:
        """Detecta STL binario comparando tamaño esperado con contador de facetas."""
        size = path.stat().st_size
        if size < 84:
            return False

        with path.open("rb") as fh:
            header = fh.read(80)
            count_bytes = fh.read(4)

        if len(count_bytes) != 4:
            return False

        facet_count = struct.unpack("<I", count_bytes)[0]
        expected_size = 84 + facet_count * 50
        if expected_size == size:
            return True

        return b"\0" in header
    
    @staticmethod
    def load_ascii(filepath: str) -> STLMesh:
        """Carga STL en formato ASCII."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(filepath)

        raw_vertices = []
        raw_normals = []
        current_normal = np.zeros(3, dtype=float)

        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "facet" and len(parts) >= 5 and parts[1] == "normal":
                    current_normal = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                elif parts[0] == "vertex" and len(parts) >= 4:
                    raw_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    if len(raw_vertices) % 3 == 0:
                        raw_normals.append(current_normal.copy())

        if len(raw_vertices) == 0 or len(raw_vertices) % 3 != 0:
            raise ValueError("Archivo STL ASCII inválido o sin facetas")

        return STLLoader._deduplicate(np.asarray(raw_vertices, dtype=float), np.asarray(raw_normals))
    
    @staticmethod
    def load_binary(filepath: str) -> STLMesh:
        """Carga STL en formato binario."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(filepath)

        raw_vertices = []
        raw_normals = []
        with path.open("rb") as fh:
            fh.read(80)
            count_bytes = fh.read(4)
            if len(count_bytes) != 4:
                raise ValueError("Archivo STL binario inválido")

            facet_count = struct.unpack("<I", count_bytes)[0]
            for _ in range(facet_count):
                data = fh.read(50)
                if len(data) != 50:
                    raise ValueError("Archivo STL binario truncado")
                values = struct.unpack("<12fH", data)
                normal = np.array(values[0:3], dtype=float)
                raw_vertices.extend([
                    values[3:6],
                    values[6:9],
                    values[9:12],
                ])
                raw_normals.append(normal)

        return STLLoader._deduplicate(np.asarray(raw_vertices, dtype=float), np.asarray(raw_normals))

    @staticmethod
    def _deduplicate(raw_vertices: np.ndarray, raw_normals: np.ndarray) -> STLMesh:
        """Convierte vértices STL repetidos por faceta a una malla indexada."""
        unique_vertices, inverse = np.unique(raw_vertices, axis=0, return_inverse=True)
        faces = inverse.reshape((-1, 3)).astype(int)

        normals = raw_normals.astype(float)
        computed = STLLoader.compute_normals(unique_vertices, faces)
        normal_lengths = np.linalg.norm(normals, axis=1)
        invalid = normal_lengths < 1e-12
        if len(normals) != len(faces) or np.any(invalid):
            normals = computed
        else:
            normals = normals / normal_lengths[:, None]

        return STLMesh(vertices=unique_vertices, faces=faces, normals=normals)
    
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
        triangles = vertices[faces]
        edges_1 = triangles[:, 1] - triangles[:, 0]
        edges_2 = triangles[:, 2] - triangles[:, 0]
        normals = np.cross(edges_1, edges_2)
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        result = np.zeros_like(normals, dtype=float)
        result[valid] = normals[valid] / lengths[valid, None]
        return result
