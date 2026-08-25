"""Datos sintéticos compartidos por los tests (húmero sintético).

Se centralizan aquí los generadores de nube de puntos y de malla sintéticos
para que los tests no dependan de ``examples/demo_interactive_web`` ni se
dupliquen entre archivos de test.
"""

import numpy as np

HUMERUS_CENTER = np.array([12.0, 3.0, 80.0])
HUMERUS_RADIUS = 22.0


def synthetic_humerus_points() -> tuple:
    """Cabeza semi-esférica con diáfisis cilíndrica (puntos + normales + semillas)."""
    center = HUMERUS_CENTER
    radius = HUMERUS_RADIUS

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
    seed_candidates = head[::37]
    return points, normals, seed_candidates


def synthetic_humerus_mesh() -> tuple:
    """Malla triangular sintética: casquete humeral + tallo cilíndrico."""
    center = HUMERUS_CENTER
    radius = HUMERUS_RADIUS
    theta = np.linspace(0.05, np.pi / 2.0, 18)
    phi = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    vertices = []
    for t in theta:
        for p in phi:
            vertices.append(center + radius * np.array([
                np.sin(t) * np.cos(p),
                np.sin(t) * np.sin(p),
                np.cos(t),
            ]))

    faces = []
    n_phi = len(phi)
    for i in range(len(theta) - 1):
        for j in range(n_phi):
            a = i * n_phi + j
            b = i * n_phi + (j + 1) % n_phi
            c = (i + 1) * n_phi + j
            d = (i + 1) * n_phi + (j + 1) % n_phi
            faces.append([a, c, b])
            faces.append([b, c, d])

    offset = len(vertices)
    z = np.linspace(-180.0, 68.0, 60)
    shaft_phi = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    for zz in z:
        for p in shaft_phi:
            vertices.append([8.0 * np.cos(p), 8.0 * np.sin(p), zz])
    for i in range(len(z) - 1):
        for j in range(len(shaft_phi)):
            a = offset + i * len(shaft_phi) + j
            b = offset + i * len(shaft_phi) + (j + 1) % len(shaft_phi)
            c = offset + (i + 1) * len(shaft_phi) + j
            d = offset + (i + 1) * len(shaft_phi) + (j + 1) % len(shaft_phi)
            faces.append([a, b, c])
            faces.append([b, d, c])

    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)
