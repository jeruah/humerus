"""Tests para visualización 3D."""

import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile

from src.visualization.visualizer import Visualizer3D, InteractiveVisualizer


class TestVisualizer3D(unittest.TestCase):
    """Tests para Visualizer3D."""
    
    def setUp(self):
        """Preparación."""
        matplotlib.use('Agg')  # Ensure Agg backend
        self.viz = Visualizer3D()
    
    def tearDown(self):
        """Limpieza."""
        plt.close('all')
    
    def test_creation(self):
        """Test creación del visualizador."""
        self.assertIsNotNone(self.viz)
        self.assertEqual(self.viz.figsize, (14, 10))
    
    def test_create_figure(self):
        """Test creación de figura."""
        self.viz.create_figure()
        self.assertIsNotNone(self.viz.fig)
        self.assertIsNotNone(self.viz.ax)
    
    def test_plot_mesh(self):
        """Test graficar malla."""
        self.viz.create_figure()
        
        # Crear tetraedro simple
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=float)
        
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ])
        
        # No debe lanzar excepción
        self.viz.plot_mesh(vertices, faces)
    
    def test_plot_sphere_red(self):
        """Test graficar esfera en ROJO."""
        self.viz.create_figure()
        
        center = np.array([10.0, 20.0, 30.0])
        radius = 25.0
        
        self.viz.plot_sphere(center, radius, color='red')
        
        # Verificar que se añadieron elementos al gráfico
        self.assertGreater(len(self.viz.ax.collections), 0)
    
    def test_plot_axis_red(self):
        """Test graficar eje en ROJO."""
        self.viz.create_figure()
        
        origin = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])
        length = 100.0
        
        self.viz.plot_axis(origin, direction, length, color='red')
        
        # Verificar que se añadieron líneas
        self.assertGreater(len(self.viz.ax.lines), 0)
    
    def test_plot_surface_points(self):
        """Test graficar puntos de superficie."""
        self.viz.create_figure()
        
        points = np.random.randn(100, 3)
        self.viz.plot_surface_points(points)
        
        self.assertGreater(len(self.viz.ax.collections), 0)
    
    def test_plot_seeds(self):
        """Test graficar semillas."""
        self.viz.create_figure()
        
        seeds = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0]
        ])
        
        valid_mask = np.array([True, False, True])
        self.viz.plot_seeds(seeds, valid_mask)
    
    def test_plot_approximations(self):
        """Test graficar múltiples aproximaciones."""
        self.viz.create_figure()
        
        approximations = [
            {
                'center': np.array([10.0, 20.0, 30.0]),
                'radius': 25.0,
                'valid': True
            },
            {
                'center': np.array([11.0, 21.0, 31.0]),
                'radius': 24.5,
                'valid': False  # No se graficará
            }
        ]
        
        self.viz.plot_approximations(approximations, color='red')
    
    def test_add_legend(self):
        """Test añadir leyenda."""
        self.viz.create_figure()
        self.viz.plot_sphere(np.array([0, 0, 0]), 10.0)
        self.viz.add_legend()
        
        # No debe lanzar excepción
    
    def test_set_equal_aspect(self):
        """Test establecer escala igual."""
        self.viz.create_figure()
        self.viz.plot_sphere(np.array([0, 0, 0]), 10.0)
        self.viz.set_equal_aspect()
        
        # No debe lanzar excepción
    
    def test_save_figure(self):
        """Test guardar figura."""
        self.viz.create_figure()
        self.viz.plot_sphere(np.array([0, 0, 0]), 10.0)
        
        with NamedTemporaryFile(suffix='.png', delete=False) as f:
            filepath = f.name
        
        try:
            self.viz.save(filepath)
            # Verificar que el archivo se creó
            import os
            self.assertTrue(os.path.exists(filepath))
        finally:
            import os
            if os.path.exists(filepath):
                os.remove(filepath)


class TestInteractiveVisualizer(unittest.TestCase):
    """Tests para InteractiveVisualizer."""
    
    def setUp(self):
        """Preparación."""
        matplotlib.use('Agg')  # Ensure Agg backend
        self.viz = InteractiveVisualizer()
    
    def tearDown(self):
        """Limpieza."""
        plt.close('all')
    
    def test_create_comparison_view(self):
        """Test crear vista comparativa."""
        mesh_data = {
            'vertices': np.random.randn(10, 3),
            'faces': np.array([[0, 1, 2]])
        }
        
        approximations = [
            {
                'center': np.array([10.0, 20.0, 30.0]),
                'radius': 25.0
            },
            {
                'center': np.array([11.0, 21.0, 31.0]),
                'radius': 24.5
            },
            {
                'center': np.array([12.0, 22.0, 32.0]),
                'radius': 26.0
            }
        ]
        
        axis_data = {
            'origin': np.array([0.0, 0.0, 0.0]),
            'direction': np.array([0.0, 0.0, 1.0]),
            'length': 100.0
        }
        
        fig = self.viz.create_comparison_view(
            mesh_data,
            approximations,
            axis_data
        )
        
        self.assertIsNotNone(fig)


class TestVisualizationIntegration(unittest.TestCase):
    """Tests de integración de visualización."""
    
    def setUp(self):
        """Preparación."""
        matplotlib.use('Agg')  # Ensure Agg backend
    
    def test_complete_visualization(self):
        """Test visualización completa."""
        viz = Visualizer3D()
        viz.create_figure()
        
        # Graficar todo
        vertices = np.random.randn(50, 3)
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        
        viz.plot_mesh(vertices, faces)
        viz.plot_sphere(np.array([0, 0, 0]), 10.0, color='red')
        viz.plot_axis(
            np.array([0, 0, 0]),
            np.array([0, 0, 1]),
            50.0,
            color='red'
        )
        
        # No debe lanzar excepción
        self.assertIsNotNone(viz.ax)
    
    def tearDown(self):
        """Limpieza."""
        plt.close('all')


if __name__ == '__main__':
    unittest.main()



class TestVisualizer3D(unittest.TestCase):
    """Tests para Visualizer3D."""
    
    def setUp(self):
        """Preparación."""
        self.viz = Visualizer3D()
    
    def tearDown(self):
        """Limpieza."""
        plt.close('all')
    
    def test_creation(self):
        """Test creación del visualizador."""
        self.assertIsNotNone(self.viz)
        self.assertEqual(self.viz.figsize, (14, 10))
    
    def test_create_figure(self):
        """Test creación de figura."""
        self.viz.create_figure()
        self.assertIsNotNone(self.viz.fig)
        self.assertIsNotNone(self.viz.ax)
    
    def test_plot_mesh(self):
        """Test graficar malla."""
        self.viz.create_figure()
        
        # Crear tetraedro simple
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=float)
        
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ])
        
        # No debe lanzar excepción
        self.viz.plot_mesh(vertices, faces)
    
    def test_plot_sphere_red(self):
        """Test graficar esfera en ROJO."""
        self.viz.create_figure()
        
        center = np.array([10.0, 20.0, 30.0])
        radius = 25.0
        
        self.viz.plot_sphere(center, radius, color='red')
        
        # Verificar que se añadieron elementos al gráfico
        self.assertGreater(len(self.viz.ax.collections), 0)
    
    def test_plot_axis_red(self):
        """Test graficar eje en ROJO."""
        self.viz.create_figure()
        
        origin = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])
        length = 100.0
        
        self.viz.plot_axis(origin, direction, length, color='red')
        
        # Verificar que se añadieron líneas
        self.assertGreater(len(self.viz.ax.lines), 0)
    
    def test_plot_surface_points(self):
        """Test graficar puntos de superficie."""
        self.viz.create_figure()
        
        points = np.random.randn(100, 3)
        self.viz.plot_surface_points(points)
        
        self.assertGreater(len(self.viz.ax.collections), 0)
    
    def test_plot_seeds(self):
        """Test graficar semillas."""
        self.viz.create_figure()
        
        seeds = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0]
        ])
        
        valid_mask = np.array([True, False, True])
        self.viz.plot_seeds(seeds, valid_mask)
    
    def test_plot_approximations(self):
        """Test graficar múltiples aproximaciones."""
        self.viz.create_figure()
        
        approximations = [
            {
                'center': np.array([10.0, 20.0, 30.0]),
                'radius': 25.0,
                'valid': True
            },
            {
                'center': np.array([11.0, 21.0, 31.0]),
                'radius': 24.5,
                'valid': False  # No se graficará
            }
        ]
        
        self.viz.plot_approximations(approximations, color='red')
    
    def test_add_legend(self):
        """Test añadir leyenda."""
        self.viz.create_figure()
        self.viz.plot_sphere(np.array([0, 0, 0]), 10.0)
        self.viz.add_legend()
        
        # No debe lanzar excepción
    
    def test_set_equal_aspect(self):
        """Test establecer escala igual."""
        self.viz.create_figure()
        self.viz.plot_sphere(np.array([0, 0, 0]), 10.0)
        self.viz.set_equal_aspect()
        
        # No debe lanzar excepción
    
    def test_save_figure(self):
        """Test guardar figura."""
        self.viz.create_figure()
        self.viz.plot_sphere(np.array([0, 0, 0]), 10.0)
        
        with NamedTemporaryFile(suffix='.png', delete=False) as f:
            filepath = f.name
        
        try:
            self.viz.save(filepath)
            # Verificar que el archivo se creó
            import os
            self.assertTrue(os.path.exists(filepath))
        finally:
            import os
            if os.path.exists(filepath):
                os.remove(filepath)


class TestInteractiveVisualizer(unittest.TestCase):
    """Tests para InteractiveVisualizer."""
    
    def setUp(self):
        """Preparación."""
        self.viz = InteractiveVisualizer()
    
    def tearDown(self):
        """Limpieza."""
        plt.close('all')
    
    def test_create_comparison_view(self):
        """Test crear vista comparativa."""
        mesh_data = {
            'vertices': np.random.randn(10, 3),
            'faces': np.array([[0, 1, 2]])
        }
        
        approximations = [
            {
                'center': np.array([10.0, 20.0, 30.0]),
                'radius': 25.0
            },
            {
                'center': np.array([11.0, 21.0, 31.0]),
                'radius': 24.5
            },
            {
                'center': np.array([12.0, 22.0, 32.0]),
                'radius': 26.0
            }
        ]
        
        axis_data = {
            'origin': np.array([0.0, 0.0, 0.0]),
            'direction': np.array([0.0, 0.0, 1.0]),
            'length': 100.0
        }
        
        fig = self.viz.create_comparison_view(
            mesh_data,
            approximations,
            axis_data
        )
        
        self.assertIsNotNone(fig)


class TestVisualizationIntegration(unittest.TestCase):
    """Tests de integración de visualización."""
    
    def test_complete_visualization(self):
        """Test visualización completa."""
        viz = Visualizer3D()
        viz.create_figure()
        
        # Graficar todo
        vertices = np.random.randn(50, 3)
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        
        viz.plot_mesh(vertices, faces)
        viz.plot_sphere(np.array([0, 0, 0]), 10.0, color='red')
        viz.plot_axis(
            np.array([0, 0, 0]),
            np.array([0, 0, 1]),
            50.0,
            color='red'
        )
        
        # No debe lanzar excepción
        self.assertIsNotNone(viz.ax)
    
    def tearDown(self):
        """Limpieza."""
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
