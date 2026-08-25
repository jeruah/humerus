"""Tests de las funciones puras del mapa de riesgo (prior académico)."""

import numpy as np
import pytest

from humero.web.risk_map import load_points, local_sphere_fit_scan, risk_score, synthetic_humerus_points
from humero.web.risk_map_interactive import build_html, to_json_array
from humero.web.risk_map_interactive import load_points as interactive_load_points


def test_risk_score_mapping():
    rmse = np.array([0.5, 2.0, 4.0])
    fitted_radius = np.array([22.0, 30.0, 50.0])
    scores = risk_score(rmse, fitted_radius, max_error=2.0, radius_min=20.0, radius_max=40.0)
    assert scores[0] < 1.0          # buen ajuste, radio en rango
    assert scores[1] == 1.0         # RMSE en el umbral
    assert scores[2] == 1.0         # radio fuera de rango
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_risk_score_handles_nan():
    rmse = np.array([np.nan, 0.5])
    fitted_radius = np.array([np.nan, 22.0])
    scores = risk_score(rmse, fitted_radius, max_error=2.0, radius_min=20.0, radius_max=40.0)
    assert scores[0] == 1.0
    assert scores[1] < 1.0


def test_local_sphere_fit_scan_on_synthetic():
    points = synthetic_humerus_points()
    rmse, radius = local_sphere_fit_scan(points, search_radius=25.0, initial_radius=22.5, min_neighbors=15)
    assert rmse.shape == (len(points),)
    assert radius.shape == (len(points),)
    # En la cabeza (z alto) hay ajustes esféricos válidos.
    head = points[:, 2] > 95.0
    assert np.isfinite(rmse[head]).sum() > 100
    head_radius = radius[head]
    head_radius = head_radius[np.isfinite(head_radius)]
    assert np.median(head_radius) == pytest.approx(22.0, abs=1.0)
    assert np.nanmedian(rmse[head]) < 0.01


def test_risk_map_load_points_synthetic():
    import argparse
    args = argparse.Namespace(stl=None, synthetic_demo=True, samples=500)
    points = load_points(args)
    assert points.ndim == 2 and points.shape[1] == 3


def test_to_json_array_nan_to_none():
    values = np.array([1.0, np.nan, np.inf, -2.0])
    out = to_json_array(values)
    assert out[0] == 1.0
    assert out[1] is None
    assert out[2] is None
    assert out[3] == -2.0


def test_interactive_load_points_and_html():
    import argparse
    args = argparse.Namespace(stl=None, synthetic_demo=True, samples=500,
                              radius_estimate=22.5, search_radius=None,
                              min_neighbors=15, max_error=2.0, radius_min=20.0, radius_max=40.0)
    points = interactive_load_points(args)
    rmse = np.zeros(len(points))
    radius = np.full(len(points), 22.0)
    html = build_html(points, rmse, radius, args)
    assert "plotly" in html
    assert "Recalcular geometria" in html
