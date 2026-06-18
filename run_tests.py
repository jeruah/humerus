#!/usr/bin/env python3
"""
Script para ejecutar TODOS LOS TESTS del proyecto.

Este script:
1. Instala dependencias
2. Ejecuta tests unitarios
3. Ejecuta tests de integración
4. Genera reporte de cobertura
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Ejecuta comando y reporta resultado."""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    print(f"Ejecutando: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    
    if result.returncode == 0:
        print(f"✓ {description} - EXITOSO")
    else:
        print(f"✗ {description} - FALLÓ")
    
    return result.returncode == 0


def main():
    """Ejecuta suite de tests."""
    print("\n" + "="*60)
    print("SUITE DE TESTS - Proyecto de Aproximación de Esfera en Húmero")
    print("="*60)
    
    project_root = Path(__file__).parent
    
    # 1. Instalar dependencias
    success = run_command(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        "Instalando dependencias"
    )
    
    if not success:
        print("\n✗ No se pudieron instalar dependencias")
        return 1
    
    # 2. Tests de auditoría
    success = run_command(
        [sys.executable, "-m", "pytest", "tests/test_audit.py", "-v", "--tb=short"],
        "Tests de Auditoría"
    )
    
    # 3. Tests de visualización
    success = success and run_command(
        [sys.executable, "-m", "pytest", "tests/test_visualization.py", "-v", "--tb=short"],
        "Tests de Visualización 3D"
    )
    
    # 4. Tests de integración
    success = success and run_command(
        [sys.executable, "-m", "pytest", "tests/test_integration.py", "-v", "--tb=short"],
        "Tests de Integración"
    )
    
    # 5. Todos los tests con cobertura
    success = success and run_command(
        [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html"
        ],
        "Suite Completa con Reporte de Cobertura"
    )
    
    # Resumen final
    print(f"\n{'='*60}")
    if success:
        print("✓ TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("✓ Reporte de cobertura disponible en: htmlcov/index.html")
    else:
        print("✗ ALGUNOS TESTS FALLARON")
    print(f"{'='*60}\n")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
