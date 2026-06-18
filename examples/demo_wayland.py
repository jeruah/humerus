"""Demostración de visualización 3D en Wayland/Hyprland con navegador.

Este script muestra cómo visualizar una esfera y un eje en ROJO usando
el método show_interactive() que abre en el navegador predeterminado.
"""

import sys
from pathlib import Path
import numpy as np

# Añadir directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.visualizer import Visualizer3D, InteractiveVisualizer


def demo_single_visualization():
    """Demo 1: Visualizar una única esfera y eje."""
    print("🖥️  Demo 1: Visualización simple en navegador")
    print("-" * 50)
    
    viz = Visualizer3D(figsize=(12, 9))
    viz.create_figure()
    
    # Esfera en ROJO
    center = np.array([15.0, 20.0, 30.0])
    radius = 20.0
    viz.plot_sphere(center, radius, color='red')
    
    # Eje en ROJO
    origin = np.array([10.0, 15.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    viz.plot_axis(origin, direction, 60, color='red')
    
    print("✓ Abriendo visualización en navegador...")
    viz.show_interactive()
    print("✓ ¡Abierto en tu navegador predeterminado!\n")


def demo_multiple_approximations():
    """Demo 2: Comparar múltiples aproximaciones."""
    print("🖥️  Demo 2: Comparación de múltiples aproximaciones")
    print("-" * 50)
    
    # Datos de aproximaciones
    approximations = [
        {
            'center': np.array([10.0, 20.0, 30.0]),
            'radius': 25.0,
            'error': 0.45
        },
        {
            'center': np.array([11.0, 21.0, 31.0]),
            'radius': 24.5,
            'error': 0.38
        },
        {
            'center': np.array([10.5, 20.5, 30.5]),
            'radius': 25.5,
            'error': 0.52
        }
    ]
    
    # Mesh simulado
    mesh_data = {
        'vertices': np.random.randn(100, 3) * 10 + np.array([10, 20, 30]),
        'faces': np.random.randint(0, 100, (50, 3))
    }
    
    # Eje longitudinal
    axis_data = {
        'origin': np.array([10.0, 20.0, 0.0]),
        'direction': np.array([0.0, 0.0, 1.0]),
        'length': 80.0
    }
    
    viz = InteractiveVisualizer(figsize=(18, 6))
    viz.create_comparison_view(mesh_data, approximations, axis_data)
    viz.save('/tmp/comparison_view.png')
    print("✓ Comparación guardada en /tmp/comparison_view.png\n")


def demo_with_seeds():
    """Demo 3: Visualizar semillas y aproximaciones."""
    print("🖥️  Demo 3: Visualización con semillas")
    print("-" * 50)
    
    viz = Visualizer3D(figsize=(14, 10))
    viz.create_figure()
    
    # Mesh simulado
    vertices = np.random.randn(200, 3) * 15 + np.array([10, 20, 30])
    faces = np.random.randint(0, 200, (100, 3))
    viz.plot_mesh(vertices, faces, color='lightblue', alpha=0.3)
    
    # Semillas (puntos de inicio)
    seeds = np.array([
        [10.0, 20.0, 30.0],
        [11.0, 21.0, 31.0],
        [12.0, 22.0, 32.0],
    ])
    viz.plot_seeds(seeds, color_valid="red", color_invalid="gray", size=100)
    
    # Aproximaciones desde semillas
    approximations = [
        {'center': np.array([10.0, 20.0, 30.0]), 'radius': 25.0, 'valid': True},
        {'center': np.array([11.0, 21.0, 31.0]), 'radius': 24.5, 'valid': True},
        {'center': np.array([12.0, 22.0, 32.0]), 'radius': 26.0, 'valid': True},
    ]
    viz.plot_approximations(approximations, color='red', alpha=0.5)
    
    # Eje longitudinal
    origin = np.array([10.0, 20.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    viz.plot_axis(origin, direction, 80, color='red')
    
    print("✓ Abriendo visualización con semillas...")
    viz.show_interactive()
    print("✓ ¡Abierto en tu navegador predeterminado!\n")


def demo_save_and_share():
    """Demo 4: Guardar visualización para compartir."""
    print("🖥️  Demo 4: Guardar visualización")
    print("-" * 50)
    
    viz = Visualizer3D(figsize=(16, 12))
    viz.create_figure()
    
    # Crear visualización
    center = np.array([0.0, 0.0, 0.0])
    radius = 30.0
    
    viz.plot_sphere(center, radius, color='red')
    
    origin = np.array([-20.0, -20.0, -50.0])
    direction = np.array([0.0, 0.0, 1.0])
    viz.plot_axis(origin, direction, 100, color='red', label='Eje longitudinal')
    
    viz.add_legend()
    
    # Guardar en formato PNG
    output_file = '/tmp/humerus_approximation.png'
    viz.save(output_file, dpi=150)
    print(f"✓ Visualización guardada en: {output_file}")
    print(f"  Tamaño: {output_file}\n")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔴 DEMOSTRACIONES: Visualización 3D con Wayland/Hyprland")
    print("="*60 + "\n")
    
    demo_single_visualization()
    demo_multiple_approximations()
    demo_with_seeds()
    demo_save_and_share()
    
    print("=" * 60)
    print("✅ Todas las demostraciones completadas!")
    print("="*60 + "\n")
