"""Tests de integración end-to-end."""

import unittest
import numpy as np
from tempfile import TemporaryDirectory

from src.audit.trail import AuditTrail, AuditManager
from src.visualization.visualizer import Visualizer3D, InteractiveVisualizer


class TestEndToEndWorkflow(unittest.TestCase):
    """Tests de flujo completo del sistema."""
    
    def test_single_seed_workflow(self):
        """Test flujo de una sola semilla."""
        # 1. Crear auditoría
        audit = AuditTrail("test_seed_001")
        
        # 2. Registrar inicialización
        seed_point = np.array([10.0, 20.0, 30.0])
        audit.log_step("initialize", {"seed": seed_point.tolist()})
        
        # 3. Validar semilla
        region = np.array([
            [9.0, 19.0, 29.0],
            [11.0, 21.0, 31.0],
            [10.5, 20.5, 30.5]
        ])
        
        is_seed_valid = audit.validate_seed(seed_point, region)
        self.assertTrue(is_seed_valid)
        
        # 4. Registrar aproximación
        sphere = {
            'center': np.array([10.1, 20.1, 30.1]),
            'radius': 25.0,
            'error': 0.8
        }
        
        audit.log_step("approximation_computed", {
            "center": sphere['center'].tolist(),
            "radius": sphere['radius'],
            "error": sphere['error']
        })
        
        # 5. Validar aproximación
        is_approx_valid = audit.is_valid_approximation(sphere)
        self.assertTrue(is_approx_valid)
        
        # 6. Obtener reporte
        report = audit.get_report()
        self.assertEqual(report['seed_id'], "test_seed_001")
        self.assertTrue(report['final_valid'])
    
    def test_multiple_seeds_workflow(self):
        """Test flujo con múltiples semillas."""
        manager = AuditManager()
        
        # Simular 10 semillas
        region = np.array([
            [9.0, 19.0, 29.0],
            [11.0, 21.0, 31.0],
            [10.5, 20.5, 30.5]
        ])
        
        valid_count = 0
        
        for i in range(10):
            audit = manager.create_audit(f"seed_{i:03d}")
            
            # Semilla aleatoria cerca de la región
            seed = region[0] + np.random.randn(3) * 0.5
            
            # Validación
            is_valid = audit.validate_seed(seed, region)
            
            if is_valid:
                valid_count += 1
                
                # Simular aproximación
                sphere = {
                    'center': seed + np.random.randn(3) * 0.1,
                    'radius': 25.0 + np.random.randn() * 0.3,
                    'error': 0.5 + np.random.rand() * 0.5
                }
                
                audit.is_valid_approximation(sphere)
        
        # Resumen
        summary = manager.get_summary()
        self.assertEqual(summary['total_audits'], 10)
        self.assertGreater(summary['valid_approximations'], 0)
    
    def test_visualization_with_audited_results(self):
        """Test visualización con resultados auditados."""
        # Crear auditorías
        manager = AuditManager()
        approximations = []
        
        for i in range(3):
            audit = manager.create_audit(f"seed_{i:03d}")
            
            sphere = {
                'center': np.array([10.0 + i, 20.0, 30.0]),
                'radius': 25.0 + i * 0.5,
                'error': 0.5 + i * 0.1,
                'valid': True
            }
            
            audit.log_step("approximation", {
                "center": sphere['center'].tolist(),
                "radius": sphere['radius']
            })
            approximations.append(sphere)
        
        # Visualizar
        viz = Visualizer3D(figsize=(12, 10))
        viz.create_figure()
        
        # Graficar esferas en ROJO
        for approx in approximations:
            viz.plot_sphere(
                approx['center'],
                approx['radius'],
                color='red',
                alpha=0.2
            )
        
        # Graficar eje en ROJO
        axis_data = {
            'origin': np.array([0.0, 0.0, 0.0]),
            'direction': np.array([0.0, 0.0, 1.0]),
            'length': 100.0
        }
        
        viz.plot_axis(
            axis_data['origin'],
            axis_data['direction'],
            axis_data['length'],
            color='red'
        )
        
        # Verificar que se creó correctamente
        self.assertIsNotNone(viz.ax)
        self.assertGreater(len(viz.ax.collections), 0)
    
    def test_full_integration(self):
        """Test integración completa: auditoría + visualización."""
        # Fase 1: Auditoría de múltiples semillas
        manager = AuditManager()
        approximations = []
        
        n_seeds = 5
        region = np.array([
            [10.0, 20.0, 30.0],
            [11.0, 21.0, 31.0],
            [9.0, 19.0, 29.0]
        ])
        
        for i in range(n_seeds):
            audit = manager.create_audit(f"seed_{i:03d}")
            
            # Generar semilla
            seed = region[i % len(region)] + np.random.randn(3) * 0.1
            
            # Validar
            audit.log_step("generate_seed", {"seed": seed.tolist()})
            is_valid = audit.validate_seed(seed, region)
            
            if is_valid:
                # Aproximar
                sphere = {
                    'center': seed,
                    'radius': 25.0,
                    'error': 0.5
                }
                
                audit.is_valid_approximation(sphere)
                approximations.append(sphere)
        
        # Fase 2: Visualización
        viz = Visualizer3D()
        viz.create_figure()
        
        # Graficar todas las aproximaciones en ROJO
        viz.plot_approximations(approximations, color='red')
        
        # Graficar eje en ROJO
        axis_origin = np.mean([a['center'] for a in approximations], axis=0)
        axis_direction = np.array([0, 0, 1])
        axis_length = 50.0
        
        viz.plot_axis(
            axis_origin,
            axis_direction,
            axis_length,
            color='red'
        )
        
        # Fase 3: Verificar resumen
        summary = manager.get_summary()
        
        self.assertEqual(summary['total_audits'], n_seeds)
        self.assertGreater(summary['valid_approximations'], 0)
        self.assertGreater(len(approximations), 0)
        
        # Limpiar
        import matplotlib.pyplot as plt
        plt.close('all')


class TestDataIntegrity(unittest.TestCase):
    """Tests de integridad de datos."""
    
    def test_audit_preserves_data(self):
        """Test que auditoría preserva datos correctamente."""
        audit = AuditTrail("test_seed")
        
        # Datos con arrays de numpy
        data = {
            "point": np.array([1.0, 2.0, 3.0]),
            "matrix": np.array([[1, 2], [3, 4]]),
            "scalar": 5.5,
            "list": [1, 2, 3]
        }
        
        audit.log_step("test_step", data)
        report = audit.get_report()
        
        # Verificar que se preservó todo
        step_data = report['steps'][0]['data']
        np.testing.assert_array_equal(
            step_data['point'],
            [1.0, 2.0, 3.0]
        )
    
    def test_manager_audit_isolation(self):
        """Test que auditorías no interfieren entre sí."""
        manager = AuditManager()
        
        audit1 = manager.create_audit("seed_1")
        audit2 = manager.create_audit("seed_2")
        
        audit1.log_step("step1", {"value": 1})
        audit2.log_step("step2", {"value": 2})
        
        # Verificar que son independientes
        self.assertEqual(len(audit1.steps), 1)
        self.assertEqual(len(audit2.steps), 1)
        self.assertEqual(audit1.steps[0].data['value'], 1)
        self.assertEqual(audit2.steps[0].data['value'], 2)


if __name__ == '__main__':
    unittest.main()
