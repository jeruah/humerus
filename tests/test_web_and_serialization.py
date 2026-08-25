"""Tests del servidor web local y de la serialización JSON de la demo."""

import argparse
import http.client
import json
import threading

import numpy as np
import pytest

from examples.demo_interactive_web import (
    best_fit_to_response,
    candidate_to_response,
    compute_best_fit_search,
    create_selection_server,
    is_point_inside_surface_volume,
)
from humero.mesh.cleaner import MeshCleaner
from humero.mesh.discretizer import MeshDiscretizer
from tests._synthetic import synthetic_humerus_mesh


def _base_args(**overrides):
    args = argparse.Namespace(
        stl=None,
        synthetic_demo=False,
        samples=2000,
        best_fit_seeds=200,
        best_fit_top=3,
        initial_radius=22.0,
        max_error=2.0,
        host="127.0.0.1",
        port=0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture()
def running_server():
    servers = []

    def _start(args):
        server = create_selection_server(args)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        host, port = server.server_address
        return f"http://{host}:{port}"

    yield _start

    for server in servers:
        server.shutdown()
        server.server_close()


class TestHTTPServer:
    def test_get_index(self, running_server):
        base = running_server(_base_args())
        conn = http.client.HTTPConnection(base.replace("http://", ""))
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        assert b"<html" in resp.read()
        conn.close()

    def test_get_unknown_404(self, running_server):
        base = running_server(_base_args())
        conn = http.client.HTTPConnection(base.replace("http://", ""))
        conn.request("GET", "/nope")
        resp = conn.getresponse()
        assert resp.status == 404
        conn.close()

    def test_approximate_without_stl_400(self, running_server):
        base = running_server(_base_args())
        conn = http.client.HTTPConnection(base.replace("http://", ""))
        conn.request("POST", "/approximate", body=json.dumps({"point_index": 0}), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 400
        data = json.loads(resp.read())
        assert "error" in data
        conn.close()

    def test_approximate_with_synthetic_stl_ok(self, running_server):
        base = running_server(_base_args(synthetic_demo=True))
        conn = http.client.HTTPConnection(base.replace("http://", ""))
        conn.request("POST", "/approximate", body=json.dumps({"point_index": 10}), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "center" in data
        np.testing.assert_allclose(data["center"], [12.0, 3.0, 80.0], atol=1e-5)
        assert data["valid"]
        conn.close()

    def test_upload_garbage_stl_400(self, running_server):
        base = running_server(_base_args())
        conn = http.client.HTTPConnection(base.replace("http://", ""))
        conn.request("POST", "/upload-stl", body=b"\x00\x01\x02", headers={"Content-Type": "application/octet-stream", "X-Filename": "x.stl"})
        resp = conn.getresponse()
        assert resp.status == 400
        conn.close()

    def test_approximate_index_out_of_range_400(self, running_server):
        base = running_server(_base_args(synthetic_demo=True))
        conn = http.client.HTTPConnection(base.replace("http://", ""))
        conn.request("POST", "/approximate", body=json.dumps({"point_index": 999999}), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 400
        conn.close()


class TestInsideSurfaceVolume:
    def test_point_inside(self):
        points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0], [1.0, 1.0, 0.0],
        ])
        assert is_point_inside_surface_volume(np.array([0.5, 0.5, 0.5]), points)

    def test_point_outside(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert not is_point_inside_surface_volume(np.array([10.0, 10.0, 10.0]), points)

    def test_invalid_shapes(self):
        assert not is_point_inside_surface_volume(np.zeros(3), np.empty((0, 3)))
        assert not is_point_inside_surface_volume(np.zeros(2), np.zeros((4, 3)))


class TestSerialization:
    def test_candidate_to_response_keeps_articular_points(self):
        vertices, faces = synthetic_humerus_mesh()
        cleaned = MeshCleaner(keep_largest_component=False).clean(vertices, faces)
        pts, nrm = MeshDiscretizer().discretize_uniform(cleaned.vertices, cleaned.faces, n_samples=800, random_seed=2)
        best_fit = compute_best_fit_search(
            pts, nrm, n_seeds=200, top_k=1, initial_radius=22.0, max_error=1.5, cleaned_mesh=cleaned
        )
        response = best_fit_to_response(best_fit)
        best = response["best"]
        assert best["method"] == "sphere_ransac"
        assert "articular_points" in best
        assert len(best["articular_points"]) > 100

    def test_candidate_to_response_none(self):
        assert candidate_to_response(None) is None
