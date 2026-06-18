"""
Script de demostración: visualización de esfera y eje en rojo.

Este script muestra cómo usar el visualizador para graficar:
- Esfera aproximada en ROJO
- Eje longitudinal en ROJO
- Múltiples aproximaciones comparativas
"""

import numpy as np
import sys
from pathlib import Path

# Añadir directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.visualizer import Visualizer3D, InteractiveVisualizer
from src.audit.trail import AuditTrail, AuditManager


def demo_single_sphere_and_axis():
    """
    Demostración 1: Una esfera y un eje, ambos en ROJO.
    """
    print("=" * 60)
    print("DEMOSTRACIÓN 1: Esfera y Eje en ROJO")
    print("=" * 60)
    
    # Crear visualizador
    viz = Visualizer3D(figsize=(12, 10))
    viz.create_figure()
    
    # 1. Graficar esfera en ROJO
    center = np.array([10.0, 20.0, 30.0])
    radius = 25.0
    
    print(f"\n✓ Graficando esfera:")
    print(f"  Centro: {center}")
    print(f"  Radio: {radius} mm")
    print(f"  Color: ROJO")
    
    viz.plot_sphere(center, radius, color='red', alpha=0.3)
    
    # 2. Graficar eje en ROJO
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    length = 100.0
    
    print(f"\n✓ Graficando eje longitudinal:")
    print(f"  Origen (cabeza): {origin}")
    print(f"  Dirección: {direction}")
    print(f"  Longitud: {length} mm")
    print(f"  Color: ROJO")
    
    viz.plot_axis(origin, direction, length, color='red', linewidth=3)
    
    # 3. Guardar
    output_file = "demo_single_sphere_axis.png"
    viz.save(output_file)
    print(f"\n✓ Figura guardada en: {output_file}")


def demo_multiple_approximations():
    """
    Demostración 2: Múltiples esferas aproximadas en ROJO.
    """
    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN 2: Múltiples Aproximaciones en ROJO")
    print("=" * 60)
    
    # Crear visualizador
    viz = Visualizer3D(figsize=(12, 10))
    viz.create_figure()
    
    # Simular múltiples aproximaciones
    approximations = []
    print(f"\n✓ Generando 5 aproximaciones de esferas:")
    
    for i in range(5):
        center = np.array([10.0, 20.0, 30.0]) + np.random.randn(3) * 0.5
        radius = 25.0 + np.random.randn() * 0.3
        
        approximations.append({
            'center': center,
            'radius': radius,
            'valid': True
        })
        
        print(f"  {i+1}. Centro: {center}, Radio: {radius:.2f} mm")
    
    # Graficar todas en ROJO
    print(f"\n✓ Graficando esferas en COLOR ROJO")
    viz.plot_approximations(approximations, color='red', alpha=0.2)
    
    # Graficar eje
    print(f"✓ Graficando eje en COLOR ROJO")
    viz.plot_axis(
        np.array([10.0, 20.0, 30.0]),
        np.array([0.0, 0.0, 1.0]),
        80.0,
        color='red'
    )
    
    # Guardar
    output_file = "demo_multiple_approximations.png"
    viz.save(output_file)
    print(f"\n✓ Figura guardada en: {output_file}")


def demo_with_auditing():
    """
    Demostración 3: Auditoría + Visualización.
    """
    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN 3: Auditoría + Visualización")
    print("=" * 60)
    
    # Crear gestor de auditorías
    manager = AuditManager()
    
    # Simular 3 semillas
    region = np.array([
        [10.0, 20.0, 30.0],
        [11.0, 21.0, 31.0],
        [9.0, 19.0, 29.0]
    ])
    
    approximations = []
    
    print(f"\n✓ Procesando 3 semillas:")
    
    for i in range(3):
        audit = manager.create_audit(f"seed_{i:03d}")
        
        # Generar semilla
        seed = region[i]
        print(f"\n  Semilla {i+1}: {seed}")
        
        # Validar
        audit.log_step("generate_seed", {"seed": seed.tolist()})
        is_valid = audit.validate_seed(seed, region)
        
        print(f"    Validación: {'✓ VÁLIDA' if is_valid else '✗ INVÁLIDA'}")
        
        if is_valid:
            # Aproximar
            sphere = {
                'center': seed + np.random.randn(3) * 0.05,
                'radius': 25.0 + np.random.randn() * 0.2,
                'error': 0.3 + np.random.rand() * 0.3,
                'valid': True
            }
            
            audit.is_valid_approximation(sphere)
            approximations.append(sphere)
            
            print(f"    Radio: {sphere['radius']:.2f} mm")
            print(f"    Error: {sphere['error']:.3f} mm")
    
    # Visualizar
    print(f"\n✓ Visualizando {len(approximations)} aproximaciones:")
    
    viz = Visualizer3D(figsize=(12, 10))
    viz.create_figure()
    
    # Graficar en ROJO
    viz.plot_approximations(approximations, color='red', alpha=0.25)
    
    # Graficar eje
    axis_center = np.mean([a['center'] for a in approximations], axis=0)
    viz.plot_axis(
        axis_center,
        np.array([0, 0, 1]),
        80.0,
        color='red'
    )
    
    # Resumen
    summary = manager.get_summary()
    print(f"\n✓ RESUMEN DE AUDITORÍA:")
    print(f"  Total de semillas: {summary['total_audits']}")
    print(f"  Aproximaciones válidas: {summary['valid_approximations']}")
    print(f"  Tasa de éxito: {summary['success_rate']*100:.1f}%")
    
    # Guardar
    output_file = "demo_with_auditing.png"
    viz.save(output_file)
    print(f"\n✓ Figura guardada en: {output_file}")


def demo_comparison_view():
    """
    Demostración 4: Vista comparativa de múltiples aproximaciones.
    """
    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN 4: Vista Comparativa de Aproximaciones")
    print("=" * 60)
    
    viz = InteractiveVisualizer()
    
    # Datos de malla (simplificados)
    mesh_data = {
        'vertices': np.random.randn(20, 3) * 10,
        'faces': np.array([[0, 1, 2]])
    }
    
    # Aproximaciones
    approximations = [
        {
            'center': np.array([10.0, 20.0, 30.0]),
            'radius': 25.0
        },
        {
            'center': np.array([10.5, 20.5, 30.5]),
            'radius': 24.8
        },
        {
            'center': np.array([9.8, 19.8, 29.8]),
            'radius': 25.2
        }
    ]
    
    # Eje
    axis_data = {
        'origin': np.array([10.0, 20.0, 30.0]),
        'direction': np.array([0, 0, 1]),
        'length': 80.0
    }
    
    print(f"\n✓ Generando vista comparativa de 3 aproximaciones en ROJO")
    
    fig = viz.create_comparison_view(
        mesh_data,
        approximations,
        axis_data
    )
    
    output_file = "demo_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Figura guardada en: {output_file}")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DEMOSTRACIONES: Visualización 3D - Esfera y Eje en ROJO")
    print("=" * 60)
    
    try:
        # Ejecutar demostraciones
        demo_single_sphere_and_axis()
        demo_multiple_approximations()
        demo_with_auditing()
        demo_comparison_view()
        
        print("\n" + "=" * 60)
        print("✓ TODAS LAS DEMOSTRACIONES COMPLETADAS EXITOSAMENTE")
        print("=" * 60)
        print("\nArchivos generados:")
        print("  - demo_single_sphere_axis.png")
        print("  - demo_multiple_approximations.png")
        print("  - demo_with_auditing.png")
        print("  - demo_comparison.png")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
