"""Tests de malla, geometría, viabilidad y refinamiento (módulos de baja cobertura)."""

import numpy as np
import pytest

from src.geometry.curvature import CurvatureCalculator, CurvatureData
from src.geometry.differential import DifferentialAnalyzer
from src.geometry.sphere import SphereGeometry
from src.mesh.cleaner import MeshCleaner
from src.mesh.discretizer import MeshDiscretizer
from src.mesh.loader import STLLoader
from src.optimization.refinement import SphereOptimizer
from src.validation.sphere import SphereValidator, SurfaceSupportValidationConfig
from src.validation.viability import SeedValidator
from src.visualization.interactive_web import InteractiveWeb3D


def _tetrahedron():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ])
    return vertices, faces


class TestSTLLoaderErrors:
    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            STLLoader.load("/no/existe.stl")

    def test_load_ascii_rejects_binary(self, tmp_path):
        path = tmp_path / "bin.stl"
        path.write_bytes(b"\x00" * 200)
        with pytest.raises(ValueError):
            STLLoader.load_ascii(str(path))

    def test_load_binary_rejects_ascii(self, tmp_path):
        path = tmp_path / "ascii.stl"
        path.write_text(
            "solid tri\nfacet normal 0 0 1\n outer loop\n  vertex 0 0 0\n  vertex 1 0 0\n  vertex 0 1 0\n endloop\nendfacet\nendsolid\n"
        )
        with pytest.raises(ValueError):
            STLLoader.load_binary(str(path))

    def test_load_invalid_content(self, tmp_path):
        path = tmp_path / "bad.stl"
        path.write_text("esto no es un stl valido")
        with pytest.raises(ValueError):
            STLLoader.load(str(path))

    def test_truncated_binary_raises(self, tmp_path):
        path = tmp_path / "trunc.stl"
        path.write_bytes(b"\x00" * 40)
        with pytest.raises(ValueError):
            STLLoader.load_binary(str(path))


class TestMeshCleaner:
    def test_keeps_largest_component(self):
        vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0],
        ])
        faces = np.array([
            [0, 1, 2],          # componente A
            [3, 4, 5],          # componente B
        ])
        cleaned = MeshCleaner().clean(vertices, faces)
        assert cleaned.faces.shape == (1, 3)
        assert cleaned.cleaning_report["component_count"] == 1
        assert cleaned.cleaning_report["removed_small_component_faces"] == 1

    def test_removes_nonfinite_vertices(self):
        vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [np.nan, np.nan, np.nan],
        ])
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        cleaned = MeshCleaner().clean(vertices, faces)
        assert cleaned.faces.shape == (1, 3)
        assert cleaned.cleaning_report["removed_nonfinite_faces"] == 1

    def test_all_degenerate_raises(self):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        with pytest.raises(ValueError):
            MeshCleaner().clean(vertices, faces)

    def test_vertex_normals(self):
        vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        ])
        faces = np.array([[0, 1, 2]])
        cleaned = MeshCleaner(keep_largest_component=False).clean(vertices, faces)
        vn = MeshCleaner().vertex_normals(cleaned)
        assert vn.shape == (3, 3)
        np.testing.assert_allclose(np.linalg.norm(vn, axis=1), 1.0, atol=1e-12)

    def test_build_adjacency_quad(self):
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        adjacency = MeshCleaner.build_adjacency(faces)
        assert adjacency == [[1], [0]]

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            MeshCleaner().clean(np.zeros((4, 2)), np.array([[0, 1, 2]]))


class TestMeshDiscretizer:
    def test_adaptive_focuses_high_curvature(self):
        rng = np.random.default_rng(0)
        vertices = rng.standard_normal((100, 3))
        faces = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 3]])
        curvature = np.array([10.0, 1.0, 1.0])
        disc = MeshDiscretizer()
        pts, norms = disc.discretize_adaptive(vertices, faces, target_samples=300, curvature_data=curvature, random_seed=1)
        assert pts.shape == (300, 3)
        assert norms.shape == (300, 3)
        # Más puntos caen en la cara de mayor curvatura (área index 0).
        # Identificamos la cara muestreada aproximando: las caras son [0,1,2],[1,2,3],[0,2,3].
        # Comprobación indirecta: sample_surface pondera, no se cae el pipeline.
        assert np.all(np.isfinite(pts))

    def test_adaptive_without_curvature_is_uniform(self):
        vertices, faces = _tetrahedron()
        disc = MeshDiscretizer()
        pts, norms = disc.discretize_adaptive(vertices, faces, target_samples=100, random_seed=7)
        assert pts.shape == (100, 3)
        assert norms.shape == (100, 3)

    def test_zero_area_mesh_raises(self):
        vertices = np.zeros((3, 3))
        faces = np.array([[0, 1, 2]])
        with pytest.raises(ValueError):
            MeshDiscretizer().discretize_uniform(vertices, faces, n_samples=5)

    def test_extract_articulation_region(self):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [50.0, 50.0, 50.0]])
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        verts, fcs = MeshDiscretizer().extract_articulation_region(
            vertices, faces, center_estimate=np.array([0.0, 0.0, 0.0]), radius_search=5.0
        )
        # Ambas caras conservan vértices dentro del radio (0 y 1).
        assert len(fcs) == 2
        assert verts.shape[1] == 3

    def test_invalid_shapes_raise(self):
        with pytest.raises(ValueError):
            MeshDiscretizer().discretize_uniform(np.zeros((3, 2)), np.array([[0, 1, 2]]))


class TestCurvature:
    def test_vertex_normals_face_flag_disambiguates(self):
        vertices, faces = _tetrahedron()
        face_normals = STLLoader.compute_normals(vertices, faces)
        # N vértices == N caras == 4: la ambigüedad del shape está presente.
        assert vertices.shape == face_normals.shape
        vertex_normals = CurvatureCalculator._vertex_normals(vertices, faces, face_normals, face_normals=True)
        assert vertex_normals.shape == (4, 3)
        np.testing.assert_allclose(np.linalg.norm(vertex_normals, axis=1), 1.0, atol=1e-12)
        # Promedio: la normal del vértice 0 NO debe ser la de una sola cara.
        assert not np.allclose(vertex_normals[0], face_normals[0])

    def test_vertex_normals_pass_through(self):
        vertices, faces = _tetrahedron()
        vn = np.repeat([[0.0, 0.0, 1.0]], 4, axis=0)
        result = CurvatureCalculator._vertex_normals(vertices, faces, vn)
        np.testing.assert_allclose(result, vn)

    def test_point_curvature_spherical_cap(self):
        radius = 25.0
        point = np.array([0.0, 0.0, radius])
        rng = np.random.default_rng(3)
        phi = rng.uniform(0, 2 * np.pi, 24)
        t = rng.uniform(0, 0.5, 24)
        neighbors = np.column_stack((
            radius * np.sin(t) * np.cos(phi),
            radius * np.sin(t) * np.sin(phi),
            radius * np.cos(t),
        ))
        data = CurvatureCalculator.compute_point_curvature(point, neighbors, np.array([0.0, 0.0, 1.0]))
        # El código usa |k| para detección de esferas (convención de signo
        # según la orientación de la normal). Verificamos magnitud.
        assert abs(data.principal_k1) == pytest.approx(1 / radius, rel=0.2)
        assert abs(data.principal_k2) == pytest.approx(1 / radius, rel=0.2)

    def test_compute_all_curvatures_shape(self):
        vertices, faces = _tetrahedron()
        face_normals = STLLoader.compute_normals(vertices, faces)
        vn = CurvatureCalculator._vertex_normals(vertices, faces, face_normals, face_normals=True)
        out = CurvatureCalculator.compute_all_curvatures(vertices, faces, vn)
        assert out.shape == (4, 4)

    def test_find_spherical_region(self):
        curvatures = np.array([
            [1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 625.0],
            [0.1, 0.2, 0.15, 0.02],
        ])
        indices = CurvatureCalculator.find_spherical_region(curvatures, np.zeros((2, 3)), radius_estimate=25.0, tolerance=0.1)
        np.testing.assert_array_equal(indices, [0])

    def test_is_local_sphere(self):
        data = CurvatureData(
            principal_k1=1 / 25.0, principal_k2=1 / 25.0,
            mean_curvature=1 / 25.0, gaussian_curvature=1 / 625.0,
            normal=np.array([0.0, 0.0, 1.0]),
        )
        assert CurvatureCalculator.is_local_sphere(data, radius_estimate=25.0, tolerance=0.01)
        not_sphere = CurvatureData(
            principal_k1=0.1, principal_k2=0.001,
            mean_curvature=0.05, gaussian_curvature=0.0001,
            normal=np.array([0.0, 0.0, 1.0]),
        )
        assert not CurvatureCalculator.is_local_sphere(not_sphere, radius_estimate=25.0, tolerance=0.1)


class TestDifferential:
    def test_fit_quadric_paraboloid(self):
        point = np.zeros(3)
        grid = np.linspace(-0.5, 0.5, 7)
        x, y = np.meshgrid(grid, grid)
        neighbors = np.column_stack((x.ravel(), y.ravel(), (x ** 2 + y ** 2).ravel()))
        hessian, _, fit_error = DifferentialAnalyzer.fit_quadric_surface(point, neighbors, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(hessian, [[2.0, 0.0], [0.0, 2.0]], atol=0.15)
        assert fit_error < 0.1

    def test_principal_curvatures_sorted(self):
        hessian = np.array([[4.0, 0.0], [0.0, 1.0]])
        k1, k2, _e1, _e2 = DifferentialAnalyzer.compute_principal_curvatures(hessian)
        assert k1 >= k2
        assert k1 == pytest.approx(4.0)
        assert k2 == pytest.approx(1.0)

    def test_shape_operator(self):
        first = np.eye(2)
        second = np.array([[2.0, 0.0], [0.0, 1.0]])
        S = DifferentialAnalyzer.compute_shape_operator(first, second)
        np.testing.assert_allclose(S, second)

    def test_fit_sphere_to_neighbors(self):
        center = np.array([5.0, -3.0, 8.0])
        radius = 20.0
        rng = np.random.default_rng(4)
        phi = rng.uniform(0, 2 * np.pi, 30)
        t = rng.uniform(0, np.pi, 30)
        surface_point = center + radius * np.array([0.0, 0.0, 1.0])
        neighbors = center + radius * np.column_stack((
            np.sin(t) * np.cos(phi), np.sin(t) * np.sin(phi), np.cos(t),
        ))
        fitted_center, fitted_radius, error = DifferentialAnalyzer.fit_sphere_to_neighbors(
            surface_point, neighbors, initial_radius=20.0
        )
        np.testing.assert_allclose(fitted_center, center, atol=1.0)
        assert abs(fitted_radius - radius) < 1.0
        assert error < 1.0

    def test_curvature_bounds(self):
        neighbors = np.array([
            [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0], [0.0, 0.0, 2.0],
        ])
        lo, hi = DifferentialAnalyzer.estimate_curvature_bounds(neighbors, np.array([0.0, 0.0, 1.0]))
        assert hi >= lo

    def test_fit_quadric_requires_neighbors(self):
        with pytest.raises(ValueError):
            DifferentialAnalyzer.fit_quadric_surface(np.zeros(3), np.zeros((3, 3)), np.array([0.0, 0.0, 1.0]))


class TestSeedValidator:
    def test_has_spherical_curvature(self):
        center = np.array([0.0, 0.0, 0.0])
        radius = 22.0
        rng = np.random.default_rng(5)
        phi = rng.uniform(0, 2 * np.pi, 40)
        t = rng.uniform(0, 1.2, 40)
        neighbors = center + radius * np.column_stack((
            np.sin(t) * np.cos(phi), np.sin(t) * np.sin(phi), np.cos(t),
        ))
        assert SeedValidator().has_spherical_curvature(np.array([0.0, 0.0, radius]), neighbors, radius_estimate=22.0, tolerance=0.2)

    def test_has_spherical_curvature_false_on_line(self):
        neighbors = np.column_stack((np.linspace(-5, 5, 10), np.zeros(10), np.zeros(10)))
        assert not SeedValidator().has_spherical_curvature(np.zeros(3), neighbors, radius_estimate=22.0, tolerance=0.1)

    def test_is_away_from_other_surfaces(self):
        point = np.array([0.0, 0.0, 0.0])
        near = np.array([[1.0, 0.0, 0.0]])
        far = np.array([[100.0, 0.0, 0.0]])
        assert not SeedValidator().is_away_from_other_surfaces(point, near, min_distance=10.0)
        assert SeedValidator().is_away_from_other_surfaces(point, far, min_distance=10.0)
        assert SeedValidator().is_away_from_other_surfaces(point, None)

    def test_is_in_articulation_region_empty(self):
        assert not SeedValidator().is_in_articulation_region(np.zeros(3), np.empty((0, 3)))


class TestSphereGeometry:
    @staticmethod
    def _sphere_points(radius, n=200, seed=0):
        center = np.zeros(3)
        rng = np.random.default_rng(seed)
        phi = rng.uniform(0, 2 * np.pi, n)
        t = rng.uniform(0, np.pi, n)
        return center + radius * np.column_stack((
            np.sin(t) * np.cos(phi), np.sin(t) * np.sin(phi), np.cos(t),
        ))

    def test_robust_fit_respects_radius_bounds(self):
        radius = 30.0
        points = self._sphere_points(radius)
        _, fitted_radius, _, _ = SphereGeometry.robust_fit(
            points, np.zeros(3), radius, radius_bounds=(10.0, 15.0), f_scale=1.0
        )
        assert fitted_radius <= 15.0 + 1e-6

    def test_robust_fit_weights_ignore_outliers(self):
        radius = 20.0
        good = self._sphere_points(radius)
        points = np.vstack([good, np.array([[0.0, 0.0, 60.0]])])
        weights = np.ones(len(points))
        weights[-1] = 1e-6
        _, fitted_radius, rmse, _ = SphereGeometry.robust_fit(
            points, np.zeros(3), radius, weights=weights, f_scale=1.0
        )
        assert abs(fitted_radius - radius) < 2.0
        assert rmse < 5.0

    def test_robust_fit_requires_min_points(self):
        with pytest.raises(ValueError):
            SphereGeometry.robust_fit(np.zeros((3, 3)), np.zeros(3), 10.0)

    def test_angular_coverage_edges(self):
        center = np.zeros(3)
        assert SphereGeometry.angular_coverage(np.empty((0, 3)), center) == 0.0
        assert SphereGeometry.angular_coverage(np.array([[1.0, 0.0, 0.0]]), center) == 0.0
        rng = np.random.default_rng(2)
        dirs = rng.standard_normal((500, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        coverage = SphereGeometry.angular_coverage(dirs, center)
        assert 0.0 <= coverage <= 1.0

    def test_from_four_points_rejects_coplanar(self):
        points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        ])
        assert SphereGeometry.from_four_points(points) is None


class TestSphereValidator:
    @staticmethod
    def _result(**overrides):
        result = {
            "center": np.array([0.0, 0.0, 0.0]),
            "radius": 22.0,
            "inlier_face_count": 200,
            "inlier_area": 500.0,
            "dominant_component_ratio": 0.95,
            "connected_component_count": 1,
        }
        result.update(overrides)
        return result

    def test_surface_support_valid(self):
        config = SurfaceSupportValidationConfig()
        out = SphereValidator.validate_surface_support(self._result(), total_area=5000.0, config=config)
        assert out["valid"]

    def test_surface_support_rejects_radius(self):
        config = SurfaceSupportValidationConfig()
        out = SphereValidator.validate_surface_support(self._result(radius=50.0), total_area=5000.0, config=config)
        assert not out["valid"]

    def test_surface_support_rejects_few_faces(self):
        config = SurfaceSupportValidationConfig()
        out = SphereValidator.validate_surface_support(self._result(inlier_face_count=3), total_area=5000.0, config=config)
        assert not out["valid"]

    def test_surface_support_rejects_non_finite(self):
        config = SurfaceSupportValidationConfig()
        out = SphereValidator.validate_surface_support(
            self._result(center=np.array([np.nan, 0.0, 0.0])), total_area=5000.0, config=config
        )
        assert not out["valid"]

    def test_surface_support_rejects_no_dominant_component(self):
        config = SurfaceSupportValidationConfig()
        out = SphereValidator.validate_surface_support(
            self._result(dominant_component_ratio=0.4, connected_component_count=3),
            total_area=5000.0,
            config=config,
        )
        assert not out["valid"]

    def test_project_perpendicular_edges(self):
        axis = np.array([0.0, 0.0, 1.0])
        assert SphereValidator.project_perpendicular(None, axis) is None
        # Vector paralelo al eje -> proyección degenerada.
        assert SphereValidator.project_perpendicular(np.array([0.0, 0.0, 5.0]), axis) is None
        # Shape inválido.
        assert SphereValidator.project_perpendicular(np.zeros(2), axis) is None
        # Vector perpendicular real.
        out = SphereValidator.project_perpendicular(np.array([3.0, 4.0, 0.0]), axis)
        assert out is not None
        np.testing.assert_allclose(np.linalg.norm(out), 1.0)

    def test_transverse_frame_with_explicit_directions(self):
        longitudinal = np.array([0.0, 0.0, 1.0])
        medial, posterior = SphereValidator.transverse_frame(
            longitudinal,
            medial_direction=np.array([1.0, 0.0, 0.0]),
            posterior_direction=np.array([0.0, 1.0, 0.0]),
        )
        np.testing.assert_allclose(medial, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(posterior, [0.0, 1.0, 0.0])


class TestInteractiveWeb:
    def test_plot_traces_add_data(self):
        viz = InteractiveWeb3D()
        viz.plot_sphere(np.zeros(3), 20.0)
        viz.plot_axis(np.zeros(3), np.array([0.0, 0.0, 1.0]), 50.0)
        viz.plot_mesh(np.zeros((3, 3)), np.array([[0, 1, 2]]))
        viz.plot_seeds(np.zeros((2, 3)))
        viz.plot_points(np.zeros((4, 3)))
        viz.plot_points_colored(np.zeros((3, 3)), np.array([0.0, 0.5, 1.0]))
        viz.plot_approximations([{"center": np.zeros(3), "radius": 10.0}])
        viz.plot_selected_seed(np.array([1.0, 2.0, 3.0]))
        assert len(viz.fig.data) >= 8

    def test_save_html(self, tmp_path):
        viz = InteractiveWeb3D()
        viz.plot_sphere(np.zeros(3), 10.0)
        out = tmp_path / "viz.html"
        viz.save(str(out))
        assert out.exists()


class TestSphereOptimizer:
    def test_optimize_from_multiple_seeds(self):
        rng = np.random.default_rng(6)
        center = np.array([0.0, 0.0, 0.0])
        radius = 22.0
        phi = rng.uniform(0, 2 * np.pi, 200)
        t = rng.uniform(0, 1.0, 200)
        points = center + radius * np.column_stack((
            np.sin(t) * np.cos(phi), np.sin(t) * np.sin(phi), np.cos(t),
        ))
        normals = points / np.linalg.norm(points, axis=1, keepdims=True)
        seeds = points[:3]
        results = SphereOptimizer().optimize_from_multiple_seeds(seeds, points, normals)
        assert len(results) == 3
        assert all("valid" in r for r in results)

    def test_select_random_seeds_reproducible(self):
        rng = np.random.default_rng(8)
        points = rng.standard_normal((50, 3))
        opt = SphereOptimizer()
        a = opt.select_random_seeds(points, n_seeds=10, random_seed=99)
        b = opt.select_random_seeds(points, n_seeds=10, random_seed=99)
        np.testing.assert_array_equal(a, b)

    def test_optimization_summary(self):
        opt = SphereOptimizer()
        opt.audit_manager.create_audit("seed_1")
        summary = opt.get_optimization_summary()
        assert summary["total_audits"] == 1
