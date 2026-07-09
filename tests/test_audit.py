"""Tests para el módulo de auditoría."""

import unittest
import numpy as np
import tempfile
import json
from pathlib import Path

from src.audit.trail import AuditTrail, AuditManager, StepRecord


class TestStepRecord(unittest.TestCase):
    """Tests para StepRecord."""
    
    def test_creation(self):
        """Test creación de registro."""
        data = {"value": 1.5, "array": np.array([1, 2, 3])}
        record = StepRecord(
            step_name="test",
            timestamp="2024-01-01T00:00:00",
            data=data
        )
        self.assertEqual(record.step_name, "test")
    
    def test_to_dict(self):
        """Test conversión a diccionario."""
        record = StepRecord(
            step_name="test",
            timestamp="2024-01-01T00:00:00",
            data={"array": np.array([1, 2, 3])}
        )
        d = record.to_dict()
        self.assertIsInstance(d['data']['array'], list)
        self.assertEqual(d['data']['array'], [1, 2, 3])


class TestAuditTrail(unittest.TestCase):
    """Tests para AuditTrail."""
    
    def setUp(self):
        """Preparación de tests."""
        self.audit = AuditTrail("test_seed_001")
    
    def test_initialization(self):
        """Test inicialización."""
        self.assertEqual(self.audit.seed_id, "test_seed_001")
        self.assertEqual(len(self.audit.steps), 0)
    
    def test_log_step(self):
        """Test registro de pasos."""
        self.audit.log_step("test_step", {"value": 1.5})
        self.assertEqual(len(self.audit.steps), 1)
        self.assertEqual(self.audit.steps[0].step_name, "test_step")
    
    def test_log_multiple_steps(self):
        """Test registro de múltiples pasos."""
        for i in range(5):
            self.audit.log_step(f"step_{i}", {"index": i})
        self.assertEqual(len(self.audit.steps), 5)
    
    def test_validate_seed_valid(self):
        """Test validación de semilla válida."""
        point = np.array([10.0, 20.0, 30.0])
        region = np.array([
            [9.0, 19.0, 29.0],
            [11.0, 21.0, 31.0],
            [10.5, 20.5, 30.5]
        ])
        
        result = self.audit.validate_seed(point, region)
        self.assertTrue(result)
        self.assertTrue(self.audit.validations['seed_viable'])
    
    def test_validate_seed_invalid_shape(self):
        """Test validación de semilla con forma inválida."""
        point = np.array([10.0, 20.0])  # Solo 2D
        result = self.audit.validate_seed(point)
        self.assertFalse(result)
    
    def test_validate_seed_far_from_region(self):
        """Test validación de semilla lejana."""
        point = np.array([100.0, 100.0, 100.0])
        region = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5]
        ])
        
        result = self.audit.validate_seed(point, region)
        self.assertFalse(result)
    
    def test_is_valid_approximation_good(self):
        """Test validación de aproximación buena."""
        sphere = {
            'center': np.array([10.0, 20.0, 30.0]),
            'radius': 22.0,
            'error': 0.5
        }
        
        result = self.audit.is_valid_approximation(sphere)
        self.assertTrue(result)
    
    def test_is_valid_approximation_high_error(self):
        """Test validación de aproximación con error alto."""
        sphere = {
            'center': np.array([10.0, 20.0, 30.0]),
            'radius': 22.0,
            'error': 5.0  # Mayor que 2.0
        }
        
        result = self.audit.is_valid_approximation(sphere)
        self.assertFalse(result)
    
    def test_is_valid_approximation_bad_radius(self):
        """Test validación de aproximación con radio malo."""
        sphere = {
            'center': np.array([10.0, 20.0, 30.0]),
            'radius': 50.0,  # Mayor que 40.0
            'error': 0.5
        }
        
        result = self.audit.is_valid_approximation(sphere)
        self.assertFalse(result)
    
    def test_get_report(self):
        """Test generación de reporte."""
        self.audit.log_step("step1", {"value": 1})
        self.audit.log_step("step2", {"value": 2})
        self.audit.validate_seed(np.array([1, 2, 3]))
        
        report = self.audit.get_report()
        
        self.assertIn('seed_id', report)
        self.assertIn('steps', report)
        self.assertIn('validations', report)
        self.assertEqual(report['total_steps'], 3)  # 2 + 1 de validate
    
    def test_to_json(self):
        """Test serialización a JSON."""
        self.audit.log_step("test", {"value": 1})
        json_str = self.audit.to_json()
        
        data = json.loads(json_str)
        self.assertIn('seed_id', data)
        self.assertIn('steps', data)


class TestAuditManager(unittest.TestCase):
    """Tests para AuditManager."""
    
    def setUp(self):
        """Preparación de tests."""
        self.manager = AuditManager()
    
    def test_create_audit(self):
        """Test creación de auditorías."""
        audit1 = self.manager.create_audit("seed_001")
        audit2 = self.manager.create_audit("seed_002")
        
        self.assertEqual(len(self.manager.audits), 2)
        self.assertIn("seed_001", self.manager.audits)
        self.assertIn("seed_002", self.manager.audits)
    
    def test_get_summary_empty(self):
        """Test resumen con auditorías vacías."""
        summary = self.manager.get_summary()
        
        self.assertEqual(summary['total_audits'], 0)
        self.assertEqual(summary['valid_approximations'], 0)
    
    def test_get_summary_with_audits(self):
        """Test resumen con auditorías."""
        audit1 = self.manager.create_audit("seed_001")
        audit2 = self.manager.create_audit("seed_002")
        
        audit1.validations['approximation_valid'] = True
        audit2.validations['approximation_valid'] = False
        
        summary = self.manager.get_summary()
        
        self.assertEqual(summary['total_audits'], 2)
        self.assertEqual(summary['valid_approximations'], 1)
        self.assertAlmostEqual(summary['success_rate'], 0.5)


class TestAuditIntegration(unittest.TestCase):
    """Tests de integración de auditoría."""
    
    def test_full_workflow(self):
        """Test flujo completo de auditoría."""
        manager = AuditManager()
        
        # Simular múltiples semillas
        for i in range(5):
            audit = manager.create_audit(f"seed_{i:03d}")
            
            # Simular pasos
            point = np.array([10.0 + i, 20.0, 30.0])
            audit.log_step("initialize", {"seed": point.tolist()})
            
            # Simular validación
            region = np.array([
                [9.0, 19.0, 29.0],
                [11.0, 21.0, 31.0]
            ])
            is_valid = audit.validate_seed(point, region)
            
            # Simular aproximación
            if is_valid:
                sphere = {
                    'center': point + np.random.randn(3) * 0.1,
                    'radius': 22.0 + np.random.randn() * 0.5,
                    'error': 0.5 + np.random.rand() * 1.0
                }
                audit.is_valid_approximation(sphere)
        
        # Verificar resumen
        summary = manager.get_summary()
        self.assertEqual(summary['total_audits'], 5)
        self.assertGreater(summary['success_rate'], 0)


if __name__ == '__main__':
    unittest.main()
