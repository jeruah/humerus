"""Módulo de visualización 3D para esfera y eje del húmero.

Este módulo proporciona herramientas para visualizar:
- Malla triangular del húmero
- Esfera aproximada (en rojo)
- Eje longitudinal (en rojo)
- Puntos de superficie y semillas
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Union
import os
import matplotlib

# Usar Agg backend que siempre funciona sin GUI
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import matplotlib.patches as mpatches


class Visualizer3D:
    """
    Visualizador 3D para malla, esfera y eje del húmero.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Inicializa visualizador.
        
        Parameters
        ----------
        figsize : Tuple[int, int]
            Tamaño de figura en pulgadas
        """
        self.figsize = figsize
        self.fig = None
        self.ax = None
    
    def create_figure(self):
        """Crea figura 3D."""
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.set_title('Aproximación de Esfera en Húmero')
    
    def plot_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: str = 'lightblue',
        alpha: float = 0.3,
        edgecolor: str = 'gray',
        linewidth: float = 0.1
    ):
        """
        Grafica malla triangular.
        
        Parameters
        ----------
        vertices : np.ndarray
            Vértices (shape: (N, 3))
        faces : np.ndarray
            Facetas (shape: (M, 3))
        color : str
            Color de los triángulos
        alpha : float
            Transparencia [0, 1]
        edgecolor : str
            Color de aristas
        linewidth : float
            Grosor de aristas
        """
        if self.ax is None:
            self.create_figure()
        
        # Graficar triángulos
        for face in faces:
            triangle = vertices[face]
            
            # Cerrar triángulo
            triangle_closed = np.vstack([triangle, triangle[0]])
            
            self.ax.plot(
                triangle_closed[:, 0],
                triangle_closed[:, 1],
                triangle_closed[:, 2],
                color=edgecolor,
                linewidth=linewidth,
                alpha=0.5
            )
        
        # Graficar vértices
        self.ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            c=color,
            s=1,
            alpha=alpha
        )
    
    def plot_sphere(
        self,
        center: np.ndarray,
        radius: float,
        color: str = 'red',
        alpha: float = 0.3,
        wireframe: bool = True,
        n_points: int = 30
    ):
        """
        Grafica esfera aproximada en ROJO.
        
        Parameters
        ----------
        center : np.ndarray
            Centro de la esfera (shape: (3,))
        radius : float
            Radio de la esfera (mm)
        color : str
            Color de la esfera (default: 'red')
        alpha : float
            Transparencia
        wireframe : bool
            Si mostrar wireframe o superficie
        n_points : int
            Número de puntos para la parametrización
        
        Examples
        --------
        >>> viz = Visualizer3D()
        >>> viz.create_figure()
        >>> viz.plot_sphere(np.array([10, 20, 30]), 25.0)
        """
        # Crear esfera parametrizada
        u = np.linspace(0, 2 * np.pi, n_points)
        v = np.linspace(0, np.pi, n_points)
        
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        
        if wireframe:
            self.ax.plot_wireframe(
                x, y, z,
                color=color,
                alpha=alpha,
                linewidth=0.5
            )
        else:
            self.ax.plot_surface(
                x, y, z,
                color=color,
                alpha=alpha
            )
        
        # Marcar centro
        self.ax.scatter(
            center[0], center[1], center[2],
            color=color,
            s=100,
            marker='o',
            label=f'Centro ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})'
        )
    
    def plot_axis(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        length: float,
        color: str = 'red',
        linewidth: float = 3,
        label: str = 'Eje longitudinal'
    ):
        """
        Grafica eje longitudinal como línea 3D en ROJO.
        
        Parameters
        ----------
        origin : np.ndarray
            Punto inicial del eje (cabeza) (shape: (3,))
        direction : np.ndarray
            Dirección unitaria (shape: (3,))
        length : float
            Longitud del eje (mm)
        color : str
            Color de la línea (default: 'red')
        linewidth : float
            Grosor de la línea
        label : str
            Etiqueta para la leyenda
        
        Examples
        --------
        >>> viz = Visualizer3D()
        >>> viz.create_figure()
        >>> origin = np.array([0, 0, 0])
        >>> direction = np.array([0, 0, 1]) / np.sqrt(1)
        >>> viz.plot_axis(origin, direction, 100.0)
        """
        # Punto final del eje
        end_point = origin + direction * length
        
        # Graficar línea
        self.ax.plot(
            [origin[0], end_point[0]],
            [origin[1], end_point[1]],
            [origin[2], end_point[2]],
            color=color,
            linewidth=linewidth,
            label=label
        )
        
        # Marcar inicio (cabeza)
        self.ax.scatter(
            origin[0], origin[1], origin[2],
            color=color,
            s=100,
            marker='^',
            label='Cabeza'
        )
        
        # Marcar fin (distal)
        self.ax.scatter(
            end_point[0], end_point[1], end_point[2],
            color=color,
            s=100,
            marker='v',
            label='Distal'
        )
    
    def plot_surface_points(
        self,
        points: np.ndarray,
        color: str = 'blue',
        size: float = 10,
        alpha: float = 0.6,
        label: str = 'Puntos de superficie'
    ):
        """
        Grafica puntos de la superficie discretizada.
        
        Parameters
        ----------
        points : np.ndarray
            Puntos de superficie (shape: (N, 3))
        color : str
            Color de los puntos
        size : float
            Tamaño de los puntos
        alpha : float
            Transparencia
        label : str
            Etiqueta para la leyenda
        """
        self.ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=color,
            s=size,
            alpha=alpha,
            label=label
        )
    
    def plot_seeds(
        self,
        seeds: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        color_valid: str = 'green',
        color_invalid: str = 'orange',
        size: float = 50,
        label: str = 'Semillas'
    ):
        """
        Grafica puntos semilla.
        
        Parameters
        ----------
        seeds : np.ndarray
            Puntos semilla (shape: (K, 3))
        valid_mask : np.ndarray, optional
            Máscara booleana de semillas válidas (shape: (K,))
        color_valid : str
            Color de semillas válidas
        color_invalid : str
            Color de semillas inválidas
        size : float
            Tamaño de los puntos
        label : str
            Etiqueta para leyenda
        """
        if valid_mask is None:
            valid_mask = np.ones(len(seeds), dtype=bool)
        
        # Semillas válidas
        valid_seeds = seeds[valid_mask]
        if len(valid_seeds) > 0:
            self.ax.scatter(
                valid_seeds[:, 0],
                valid_seeds[:, 1],
                valid_seeds[:, 2],
                c=color_valid,
                s=size,
                marker='*',
                label='Semillas válidas'
            )
        
        # Semillas inválidas
        invalid_seeds = seeds[~valid_mask]
        if len(invalid_seeds) > 0:
            self.ax.scatter(
                invalid_seeds[:, 0],
                invalid_seeds[:, 1],
                invalid_seeds[:, 2],
                c=color_invalid,
                s=size,
                marker='x',
                label='Semillas inválidas'
            )
    
    def plot_approximations(
        self,
        approximations: List[Dict],
        color: str = 'red',
        alpha: float = 0.2,
        n_points: int = 20
    ):
        """
        Grafica múltiples esferas aproximadas en ROJO.
        
        Parameters
        ----------
        approximations : List[Dict]
            Lista de aproximaciones con 'center', 'radius', 'valid'
        color : str
            Color de las esferas (default: 'red')
        alpha : float
            Transparencia
        n_points : int
            Puntos de parametrización
        
        Examples
        --------
        >>> approximations = [
        ...     {'center': np.array([10, 20, 30]), 'radius': 25.0, 'valid': True},
        ...     {'center': np.array([11, 21, 31]), 'radius': 24.5, 'valid': True}
        ... ]
        >>> viz.plot_approximations(approximations)
        """
        for approx in approximations:
            if approx.get('valid', False):
                self.plot_sphere(
                    approx['center'],
                    approx['radius'],
                    color=color,
                    alpha=alpha,
                    n_points=n_points,
                    wireframe=True
                )
    
    def add_legend(self, loc: str = 'upper left'):
        """Añade leyenda al gráfico."""
        self.ax.legend(loc=loc, fontsize=8)
    
    def set_equal_aspect(self):
        """Establece escala igual en los tres ejes."""
        # Obtener límites actuales
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()
        
        # Calcular el rango máximo
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        z_range = zlim[1] - zlim[0]
        max_range = max(x_range, y_range, z_range) / 2.0
        
        # Centros
        x_mid = (xlim[0] + xlim[1]) * 0.5
        y_mid = (ylim[0] + ylim[1]) * 0.5
        z_mid = (zlim[0] + zlim[1]) * 0.5
        
        # Establecer nuevos límites
        self.ax.set_xlim(x_mid - max_range, x_mid + max_range)
        self.ax.set_ylim(y_mid - max_range, y_mid + max_range)
        self.ax.set_zlim(z_mid - max_range, z_mid + max_range)
    
    def show(self):
        """Muestra el gráfico."""
        self.add_legend()
        self.set_equal_aspect()
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath: str, dpi: int = 150):
        """
        Guarda el gráfico en archivo.
        
        Parameters
        ----------
        filepath : str
            Ruta del archivo (ej: 'visualization.png')
        dpi : int
            Resolución (puntos por pulgada)
        """
        self.add_legend()
        self.set_equal_aspect()
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Visualización guardada en: {filepath}")


    
    def show_interactive(self):
        """Abre la visualización en navegador web (Wayland compatible)."""
        import webbrowser
        import tempfile
        
        if self.fig is None:
            raise ValueError("Debes llamar a create_figure() primero")
        
        # Guardar en archivo temporal
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        self.fig.savefig(temp_file.name, dpi=100, bbox_inches='tight')
        temp_file.close()
        
        # Crear HTML para mostrar en navegador
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Visualización 3D - Húmero</title>
    <style>
        body {{ 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            height: 100vh; 
            margin: 0; 
            background-color: #f0f0f0;
            font-family: Arial, sans-serif;
        }}
        .container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        img {{ max-width: 100%; height: auto; }}
        h1 {{ text-align: center; color: #333; }}
        .info {{ color: #666; text-align: center; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔴 Visualización 3D - Esfera y Eje del Húmero</h1>
        <img src="file://{temp_file.name}" alt="Visualización 3D">
        <div class="info">
            <p>Esfera en <strong style="color: red;">ROJO</strong> | Eje en <strong style="color: red;">ROJO</strong></p>
            <p>Archivo: {temp_file.name}</p>
        </div>
    </div>
</body>
</html>"""
        
        html_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
        html_file.write(html_content)
        html_file.close()
        
        # Abrir en navegador
        print(f"✓ Abriendo en navegador: {html_file.name}")
        webbrowser.open(f'file://{html_file.name}')
        print("✓ ¡Visualización abierta!")




class InteractiveVisualizer:
    """Visualizador interactivo con vistas comparativas."""
    
    def __init__(self, figsize: Tuple[int, int] = (18, 6)):
        """
        Inicializa visualizador interactivo.
        
        Parameters
        ----------
        figsize : Tuple[int, int]
            Tamaño de figura en pulgadas
        """
        self.figsize = figsize
        self.fig = None
        self.axes = None
    
    def create_comparison_view(self, meshes: Union[Dict, List[Dict]], approximations: List[Dict], axis_data: Optional[Dict] = None):
        """
        Crea vista comparativa de múltiples aproximaciones.
        
        Parameters
        ----------
        meshes : Union[Dict, List[Dict]]
            Malla con 'vertices' y 'faces', o lista de mallas
        approximations : List[Dict]
            Lista de aproximaciones con 'center', 'radius', 'error'
        axis_data : Optional[Dict]
            Datos del eje con 'origin', 'direction', 'length'
        """
        # Convertir mesh único a lista
        if isinstance(meshes, dict):
            meshes_list = [meshes] * len(approximations)
        else:
            meshes_list = meshes
        
        n_approx = len(approximations)
        self.fig = plt.figure(figsize=self.figsize)
        self.axes = []
        
        for i, approx in enumerate(approximations):
            ax = self.fig.add_subplot(1, n_approx, i + 1, projection='3d')
            self.axes.append(ax)
            
            # Plot mesh
            if i < len(meshes_list):
                mesh = meshes_list[i]
                vertices = mesh['vertices']
                faces = mesh['faces']
                
                # Plotear mesh directamente
                for face in faces:
                    if len(face) >= 3:
                        # Cerrar el triangulo
                        indices = list(face[:3]) + [face[0]]
                        points = vertices[indices]
                        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'gray', linewidth=0.1)
            
            # Plot sphere in RED (wireframe)
            center = approx['center']
            radius = approx['radius']
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
            ax.plot_wireframe(x, y, z, color='red', alpha=0.6, linewidth=0.5)
            ax.scatter(*center, color='red', s=50)
            
            # Plot axis if provided
            if axis_data:
                origin = axis_data['origin']
                direction = axis_data['direction'] / np.linalg.norm(axis_data['direction'])
                length = axis_data['length']
                end_point = origin + direction * length
                ax.plot(
                    [origin[0], end_point[0]],
                    [origin[1], end_point[1]],
                    [origin[2], end_point[2]],
                    color='red',
                    linewidth=2
                )
                ax.scatter(*origin, color='red', s=50, marker='^')
                ax.scatter(*end_point, color='red', s=50, marker='v')
            
            error = approx.get('error', 0)
            ax.set_title(f'Aprox {i+1}\nError: {error:.3f}mm')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        return self.fig
    
    def save(self, filepath: str):
        """Guarda la figura."""
        if self.fig is None:
            raise ValueError("Debes llamar a create_comparison_view() primero")
        self.fig.savefig(filepath, dpi=100, bbox_inches='tight')
        print(f"✓ Figura guardada en: {filepath}")
    
    def show(self):
        """Muestra la figura."""
        if self.fig is None:
            raise ValueError("Debes llamar a create_comparison_view() primero")
        plt.show()
