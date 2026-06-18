"""Tests del pipeline cientifico minimo."""

import struct
from pathlib import Path

import numpy as np

from examples.demo_interactive_web import compute_from_seed, result_to_response, synthetic_humerus_points
from src.approximation.sphere import SphericalApproximator
from src.audit.trail import AuditTrail
from src.axis.longitudinal import AxisApproximator
from src.geometry.curvature import CurvatureCalculator, CurvatureData
from src.mesh.discretizer import MeshDiscretizer
from src.mesh.loader import STLLoader
from src.optimization.refinement import SphereOptimizer
from src.validation.viability import SeedValidator
from src.visualization.interactive_web import InteractiveWeb3D


def test_deterministic_seed_estimates_known_sphere():
    points, normals, seeds = synthetic_humerus_points()
    seed = seeds[10]
    audit = AuditTrail("seed_deterministic")

    sphere = SphericalApproximator(max_iterations=20, convergence_threshold=1e-6).approximate_from_seed(
        seed,
        points,
        normals,
        audit_trail=audit,
        initial_radius=25.0,
    )

    np.testing.assert_allclose(sphere["center"], np.array([0.0, 0.0, 80.0]), atol=1e-5)
    assert abs(sphere["radius"] - 25.0) < 1e-5
    assert sphere["error"] < 1e-5
    assert sphere["converged"]


def test_axis_estimation_on_synthetic_humerus():
    points, _, _ = synthetic_humerus_points()
    axis = AxisApproximator.compute_longitudinal_axis(points)

    assert axis["length"] > 100.0
    assert axis["validation"]["overall_valid"]
    assert abs(abs(axis["direction"][2]) - 1.0) < 1e-6
    assert axis["method"] == "shaft_pca"
    assert axis["axis_fit_point_count"] < axis["total_point_count"]


def test_shaft_pca_ignores_biased_irregular_ends():
    rng = np.random.default_rng(7)
    z = np.linspace(-120.0, 120.0, 90)
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    z_grid, angle_grid = np.meshgrid(z, angles)
    shaft = np.column_stack((
        8.0 * np.cos(angle_grid).ravel(),
        5.0 * np.sin(angle_grid).ravel(),
        z_grid.ravel(),
    ))

    proximal = rng.normal(loc=[45.0, 0.0, 165.0], scale=[16.0, 7.0, 9.0], size=(900, 3))
    distal = rng.normal(loc=[-35.0, 0.0, -160.0], scale=[13.0, 8.0, 8.0], size=(900, 3))
    points = np.vstack((shaft, proximal, distal))

    global_axis = AxisApproximator.compute_longitudinal_axis(points, method="pca")
    shaft_axis = AxisApproximator.compute_longitudinal_axis(points, method="shaft_pca", shaft_trim_fraction=0.28)
    true_axis = np.array([0.0, 0.0, 1.0])

    global_alignment = abs(np.dot(global_axis["direction"], true_axis))
    shaft_alignment = abs(np.dot(shaft_axis["direction"], true_axis))

    assert shaft_alignment > global_alignment
    assert shaft_alignment > 0.98


def test_seed_validator_and_random_seed_selection():
    points, _, seeds = synthetic_humerus_points()
    validator = SeedValidator()

    assert validator.is_in_articulation_region(seeds[0], points, tolerance=0.1)
    selected = SphereOptimizer().select_random_seeds(seeds, n_seeds=5, random_seed=123)

    assert selected.shape == (5, 3)
    np.testing.assert_allclose(
        selected,
        SphereOptimizer().select_random_seeds(seeds, n_seeds=5, random_seed=123),
    )


def test_visualizer_has_selected_seed_trace():
    _, _, seeds = synthetic_humerus_points()
    viz = InteractiveWeb3D()
    viz.plot_selected_seed(seeds[3])

    assert len(viz.fig.data) == 1
    assert viz.fig.data[0].name == "Semilla Seleccionada"


def test_clicked_seed_response_contains_visual_traces():
    points, normals, seeds = synthetic_humerus_points()
    result = compute_from_seed(seeds[10], points, normals, initial_radius=25.0, max_error=2.0)
    response = result_to_response(result)

    np.testing.assert_allclose(response["center"], [0.0, 0.0, 80.0], atol=1e-5)
    assert abs(response["radius"] - 25.0) < 1e-5
    assert response["axis_length"] > 0
    assert abs(response["diameter_length_percentage"] - 15.5) < 0.2
    assert response["sphere_drawn"]
    assert response["valid"]
    assert len(response["traces"]) == 4


def test_large_visual_sphere_is_not_drawn():
    points, normals, seeds = synthetic_humerus_points()
    result = compute_from_seed(seeds[10], points, normals, initial_radius=25.0, max_error=2.0)
    response = result_to_response(result, max_visual_diameter_length_ratio=0.10)

    assert not response["sphere_drawn"]
    assert "exceeds" in response["sphere_visual_reason"]
    assert len(response["traces"]) == 3


def test_sphere_validation_uses_bone_length_ratio():
    points, _, _ = synthetic_humerus_points()
    audit = AuditTrail("ratio_valid")
    valid_sphere = {
        "center": np.array([0.0, 0.0, 80.0]),
        "radius": 25.0,
        "error": 0.1,
    }
    assert audit.is_valid_approximation(valid_sphere, surface_points=points)

    audit = AuditTrail("ratio_invalid")
    invalid_sphere = {
        "center": np.array([0.0, 0.0, 80.0]),
        "radius": 35.0,
        "error": 0.1,
    }
    assert not audit.is_valid_approximation(invalid_sphere, surface_points=points)


def test_ascii_stl_loader_and_uniform_discretizer(tmp_path: Path):
    stl = tmp_path / "triangle.stl"
    stl.write_text(
        """solid tri
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 1 0 0
  vertex 0 1 0
 endloop
endfacet
endsolid tri
""",
        encoding="utf-8",
    )

    mesh = STLLoader.load(str(stl))
    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.shape == (1, 3)
    np.testing.assert_allclose(mesh.normals[0], np.array([0.0, 0.0, 1.0]))

    points, normals = MeshDiscretizer().discretize_uniform(mesh.vertices, mesh.faces, n_samples=10, random_seed=1)
    assert points.shape == (10, 3)
    assert normals.shape == (10, 3)
    assert np.all(points[:, 0] >= 0.0)
    assert np.all(points[:, 1] >= 0.0)
    assert np.all(points[:, 0] + points[:, 1] <= 1.0 + 1e-12)


def test_binary_stl_loader(tmp_path: Path):
    stl = tmp_path / "triangle_binary.stl"
    normal = (0.0, 0.0, 1.0)
    vertices = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    with stl.open("wb") as fh:
        fh.write(b"binary test".ljust(80, b" "))
        fh.write(struct.pack("<I", 1))
        fh.write(struct.pack("<12fH", *(normal + vertices + (0,))))

    mesh = STLLoader.load(str(stl))
    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.shape == (1, 3)


def test_curvature_spherical_region_filter():
    curvature = CurvatureData(
        principal_k1=1.0 / 25.0,
        principal_k2=1.0 / 25.0,
        mean_curvature=1.0 / 25.0,
        gaussian_curvature=1.0 / (25.0 * 25.0),
        normal=np.array([0.0, 0.0, 1.0]),
    )

    assert CurvatureCalculator.is_local_sphere(curvature, radius_estimate=25.0, tolerance=0.01)
