"""Demo: mapa de riesgo de falsos positivos para la búsqueda de cabeza humeral.

En vez de curvatura diferencial punto a punto (sensible al ruido de mallas
STL decimadas), ajusta una esfera local en cada punto muestreado -usando el
mismo mecanismo que `SphericalApproximator`/`HumeralHeadBestFitSearch`- y
marca en verde los puntos donde ese ajuste sería aceptado como candidato
válido por el buscador real. Zonas verdes fuera de la cabeza anatómica son
posibles falsos positivos para el algoritmo de búsqueda.

Por defecto los hiperparámetros son los mismos que usa
`HumeralHeadBestFitSearch` en producción (initial_radius=22.5,
max_error=2.0, search_radius=initial_radius*1.15), pero son todos
ajustables por CLI.

Uso básico:

    python3 examples/demo_head_risk_map.py --stl data/sample_humeri/HumeroFinal1.stl
    python3 examples/demo_head_risk_map.py --synthetic-demo
    python3 examples/demo_head_risk_map.py --stl mi_humero.stl --radius-estimate 28 --max-error 3.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.differential import DifferentialAnalyzer
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

    z = np.linspace(-217.6, 68.0, 160)
    a = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    z_grid, a_grid = np.meshgrid(z, a)
    shaft_radius = 8.0
    shaft = np.column_stack((
        shaft_radius * np.cos(a_grid).ravel(),
        shaft_radius * np.sin(a_grid).ravel(),
        z_grid.ravel(),
    ))

    return np.vstack((head, shaft))


def load_points(args: argparse.Namespace) -> np.ndarray:
    """Carga superficie desde STL o usa datos sintéticos (solo posiciones)."""
    if args.stl:
        mesh = STLLoader.load(args.stl)
        discretizer = MeshDiscretizer()
        points, _ = discretizer.discretize_uniform(mesh.vertices, mesh.faces, args.samples, random_seed=42)
        return points
    return synthetic_humerus_points()


def local_sphere_fit_scan(
    points: np.ndarray,
    search_radius: float,
    initial_radius: float,
    min_neighbors: int,
) -> tuple:
    """
    Ajusta una esfera local en cada punto usando sus vecinos dentro de search_radius.

    Misma lógica de vecindario que `SphericalApproximator._get_local_points`,
    reusando `DifferentialAnalyzer.fit_sphere_to_neighbors` para el ajuste.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (rmse, fitted_radius) por punto; NaN donde no hay suficientes vecinos.
    """
    tree = cKDTree(points)
    rmse = np.full(len(points), np.nan)
    fitted_radius = np.full(len(points), np.nan)

    for i, point in enumerate(points):
        neighbor_idx = tree.query_ball_point(point, search_radius)
        neighbor_idx = [j for j in neighbor_idx if j != i]
        if len(neighbor_idx) < min_neighbors:
            continue
        try:
            _, radius, error = DifferentialAnalyzer.fit_sphere_to_neighbors(
                point, points[neighbor_idx], initial_radius=initial_radius
            )
            rmse[i] = error
            fitted_radius[i] = radius
        except (ValueError, np.linalg.LinAlgError):
            continue

    return rmse, fitted_radius


def risk_score(
    rmse: np.ndarray,
    fitted_radius: np.ndarray,
    max_error: float,
    radius_min: float,
    radius_max: float,
) -> np.ndarray:
    """
    Score de riesgo por punto en [0, 1]: 0 = candidato válido para el buscador
    (buen ajuste Y radio dentro de [radius_min, radius_max]), 1 = el buscador
    lo descartaría (RMSE en o sobre max_error, o radio fuera del rango).
    """
    finite = np.isfinite(rmse) & np.isfinite(fitted_radius)
    rmse_ratio = np.where(finite, rmse / max_error, np.inf)

    radius_half = 0.5 * (radius_max - radius_min)
    outside_by = np.maximum(radius_min - fitted_radius, fitted_radius - radius_max)
    outside_by = np.maximum(outside_by, 0.0)
    radius_ratio = np.where(finite, outside_by / radius_half, np.inf)

    score = np.maximum(rmse_ratio, radius_ratio)
    return np.clip(score, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stl", type=str, default=None, help="Ruta a archivo STL")
    parser.add_argument("--synthetic-demo", action="store_true", help="Usar superficie sintética")
    parser.add_argument("--samples", type=int, default=4000, help="Puntos a muestrear del STL")
    parser.add_argument("--radius-estimate", type=float, default=22.5,
                         help="Radio esperado de la cabeza, mm (default: igual a HumeralHeadBestFitSearch.initial_radius)")
    parser.add_argument("--search-radius", type=float, default=None,
                         help="Radio de vecindario local, mm (default: radius-estimate * 1.15, igual que SphericalApproximator)")
    parser.add_argument("--max-error", type=float, default=2.0,
                         help="RMSE máximo aceptado, mm (default: igual a HumeralHeadBestFitSearch.max_error)")
    parser.add_argument("--radius-min", type=float, default=20.0, help="Radio fisiológico mínimo aceptado, mm")
    parser.add_argument("--radius-max", type=float, default=40.0, help="Radio fisiológico máximo aceptado, mm")
    parser.add_argument("--min-neighbors", type=int, default=15, help="Vecinos mínimos para ajustar la esfera local")
    parser.add_argument("--output", type=str, default=None, help="Guardar HTML en vez de abrir navegador")
    args = parser.parse_args()

    if not args.stl and not args.synthetic_demo:
        parser.error("Usa --stl <archivo> o --synthetic-demo")

    search_radius = args.search_radius if args.search_radius is not None else args.radius_estimate * 1.15

    print("Cargando y muestreando superficie...")
    points = load_points(args)
    print(f"  {len(points)} puntos muestreados")

    print(f"Ajustando esfera local por punto (search_radius={search_radius:.2f}mm, "
          f"initial_radius={args.radius_estimate}mm, min_neighbors={args.min_neighbors})...")
    rmse, fitted_radius = local_sphere_fit_scan(points, search_radius, args.radius_estimate, args.min_neighbors)

    print(f"Calculando score de riesgo (max_error={args.max_error}mm, "
          f"radio_fisiologico=[{args.radius_min}, {args.radius_max}]mm)...")
    scores = risk_score(rmse, fitted_radius, args.max_error, args.radius_min, args.radius_max)

    candidate_mask = scores < 1.0
    print(f"  Puntos con ajuste sin datos suficientes: {int(np.sum(~np.isfinite(rmse)))}/{len(points)}")
    print(f"  Puntos candidatos (score<1.0, el buscador los aceptaria): {int(np.sum(candidate_mask))}/{len(points)}")
    print(f"  Puntos de bajo riesgo (score<0.3): {int(np.sum(scores < 0.3))}/{len(points)}")

    hover_text = [
        f"score={s:.3f}<br>rmse={r:.3f}mm<br>radio_ajustado={fr:.2f}mm"
        if np.isfinite(r) else f"score={s:.3f}<br>sin vecinos suficientes"
        for s, r, fr in zip(scores, rmse, fitted_radius)
    ]

    viz = InteractiveWeb3D(title="Mapa de riesgo: Verde=candidato valido para el buscador, Rojo=descartado")
    viz.plot_points_colored(points, scores, name="Riesgo", colorbar_title="score", hover_text=hover_text)

    if args.output:
        viz.save(args.output)
    else:
        viz.show()


if __name__ == "__main__":
    main()
