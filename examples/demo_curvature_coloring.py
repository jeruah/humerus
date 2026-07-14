"""Demo: colorear la superficie del húmero según cercanía a una esfera.

Muestrea puntos de la superficie, calcula su curvatura local y asigna un
score continuo de "sphericity" (0 = coincide con la esfera esperada,
1 = se aleja). Verde = cerca de la esfera, rojo = lejos.

Uso básico:

    python3 examples/demo_curvature_coloring.py --stl data/sample_humeri/HumeroFinal1.stl
    python3 examples/demo_curvature_coloring.py --synthetic-demo
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.curvature import CurvatureCalculator
from src.mesh.discretizer import MeshDiscretizer
from src.mesh.loader import STLLoader
from src.visualization.interactive_web import InteractiveWeb3D


def synthetic_humerus_points() -> tuple:
    """Cabeza semi-esférica con diáfisis cilíndrica para pruebas rápidas."""
    center = np.array([12.0, 3.0, 80.0])
    radius = 22.0

    theta = np.linspace(0.0, np.pi / 2.0, 24)
    phi = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    head = np.column_stack((
        center[0] + radius * np.sin(theta_grid).ravel() * np.cos(phi_grid).ravel(),
        center[1] + radius * np.sin(theta_grid).ravel() * np.sin(phi_grid).ravel(),
        center[2] + radius * np.cos(theta_grid).ravel(),
    ))
    head_normals = head - center
    head_normals = head_normals / np.linalg.norm(head_normals, axis=1)[:, None]

    z = np.linspace(-217.6, 68.0, 160)
    a = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    z_grid, a_grid = np.meshgrid(z, a)
    shaft_radius = 8.0
    shaft = np.column_stack((
        shaft_radius * np.cos(a_grid).ravel(),
        shaft_radius * np.sin(a_grid).ravel(),
        z_grid.ravel(),
    ))
    shaft_normals = np.column_stack((
        np.cos(a_grid).ravel(),
        np.sin(a_grid).ravel(),
        np.zeros(a_grid.size),
    ))

    points = np.vstack((head, shaft))
    normals = np.vstack((head_normals, shaft_normals))
    return points, normals


def load_surface(args: argparse.Namespace) -> tuple:
    """Carga superficie desde STL o usa datos sintéticos."""
    if args.stl:
        mesh = STLLoader.load(args.stl)
        discretizer = MeshDiscretizer()
        return discretizer.discretize_uniform(mesh.vertices, mesh.faces, args.samples, random_seed=42)
    return synthetic_humerus_points()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", type=str, default=None, help="Ruta a archivo STL")
    parser.add_argument("--synthetic-demo", action="store_true", help="Usar superficie sintética")
    parser.add_argument("--samples", type=int, default=5000, help="Puntos a muestrear del STL")
    parser.add_argument("--neighbors", type=int, default=16, help="Vecinos para estimar curvatura por punto")
    parser.add_argument("--radius-estimate", type=float, default=22.0, help="Radio esperado de la esfera (mm)")
    parser.add_argument("--tolerance", type=float, default=0.15, help="Tolerancia relativa (score=1.0 en el límite)")
    parser.add_argument("--output", type=str, default=None, help="Guardar HTML en vez de abrir navegador")
    args = parser.parse_args()

    if not args.stl and not args.synthetic_demo:
        parser.error("Usa --stl <archivo> o --synthetic-demo")

    print("Cargando y muestreando superficie...")
    points, normals = load_surface(args)
    print(f"  {len(points)} puntos muestreados")

    print("Calculando curvatura local por punto...")
    curvatures = CurvatureCalculator.compute_point_cloud_curvatures(points, normals, k=args.neighbors)

    print(f"Calculando score de sphericity (R esperado={args.radius_estimate}mm, tol={args.tolerance})...")
    scores = CurvatureCalculator.sphericity_score(curvatures, args.radius_estimate, args.tolerance)
    print(f"  Score promedio: {np.nanmean(scores):.3f} (0=esférico, 1=alejado)")
    print(f"  Puntos cerca de la esfera (score<0.3): {int(np.sum(scores < 0.3))}/{len(scores)}")

    hover_text = [
        f"score={s:.3f}<br>k1={c[0]:.4f}<br>k2={c[1]:.4f}<br>H={c[2]:.4f}"
        for s, c in zip(scores, curvatures)
    ]

    viz = InteractiveWeb3D(title="Curvatura del Húmero: Verde=Esférico, Rojo=Alejado")
    viz.plot_points_colored(points, scores, name="Sphericity", colorbar_title="score", hover_text=hover_text)

    if args.output:
        viz.save(args.output)
    else:
        viz.show()


if __name__ == "__main__":
    main()
