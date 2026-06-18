# Instrucciones del Copilot para el Proyecto de Aproximación de Esfera en Húmero

## Contexto del Proyecto

Este es un proyecto de investigación para aproximar esferas a superficies articulares de húmeros mediante:
- Discretización de modelos 3D (STL)
- Análisis de geometría diferencial
- Algoritmos iterativos de optimización
- Sistemas de auditoría y validación

**Lenguaje**: Python únicamente
**Dominio**: Geometría computacional, biomecánica

## Cómo Trabajar con Este Proyecto

### 1. Entender la Arquitectura
- Lee `.agent.md` para entender los 5 agentes especializados
- Lee `AGENTS.md` para detalles de responsabilidades
- Lee `.instructions.md` para requisitos técnicos

### 2. Usar Agentes Especializados
Cuando pidas código, especifica cuál agente debería implementarlo:

```
@geometry-specialist: Implementar función para calcular curvatura principal

@mesh-processor: Cargar un archivo STL y retornar vértices y facetas

@validation-auditor: Crear clase AuditTrail que registre cada paso

@sphere-optimizer: Implementar algoritmo iterativo de refinamiento

@axis-approximator: Calcular eje longitudinal del húmero usando PCA
```

### 3. Estándares de Código

#### Clases Requeridas
```python
class AuditTrail:
    """Registra cada paso de cada aproximación"""
    
    def log_step(self, step_name: str, data: dict) -> None:
        """Registra un paso con datos estructurados"""
    
    def validate_seed(self, point: np.ndarray) -> bool:
        """Valida que una semilla está en región articular viable"""
    
    def is_valid_approximation(self, sphere: dict) -> bool:
        """Valida que la aproximación es aceptable"""
    
    def get_report(self) -> dict:
        """Retorna reporte completo de auditoría"""

class SphericalApproximator:
    """Aproxima esfera a superficie articular"""
    
    def approximate_from_seed(
        self, 
        seed_point: np.ndarray, 
        audit_trail: AuditTrail
    ) -> dict:
        """
        Aproxima esfera desde una semilla
        
        Returns: {'center': np.ndarray(3,), 'radius': float, 'error': float}
        """
        
class MeshProcessor:
    """Carga y procesa mallas 3D (STL)"""
    
    def load_stl(self, filepath: str) -> dict:
        """Carga STL, retorna {'vertices', 'faces', 'normals'}"""
    
    def discretize_surface(self, vertices, faces, n_samples=5000) -> np.ndarray:
        """Samplea N puntos sobre la superficie"""

class GeometricAnalyzer:
    """Análisis de geometría diferencial"""
    
    def compute_curvature(self, point: np.ndarray, neighbors: np.ndarray) -> dict:
        """
        Calcula κ₁, κ₂, H, K en un punto
        
        Returns: {'k1': float, 'k2': float, 'mean': float, 'gaussian': float}
        """
```

#### Type Hints y Documentación
```python
def example_function(
    parameter1: np.ndarray,
    parameter2: float,
    audit: Optional[AuditTrail] = None
) -> dict[str, Union[np.ndarray, float]]:
    """
    Descripción breve.
    
    Parameters
    ----------
    parameter1 : np.ndarray
        Descripción del parámetro
    parameter2 : float
        Descripción del parámetro
    audit : Optional[AuditTrail]
        Objeto de auditoría para registrar
    
    Returns
    -------
    dict
        Descripción del resultado con claves
    
    Notes
    -----
    Notas matemáticas o técnicas importantes
    
    Examples
    --------
    >>> result = example_function(np.array([1,2,3]), 0.5)
    >>> result['key']
    value
    """
    if audit:
        audit.log_step("example_function", {"param1": parameter1.tolist()})
    # ... implementación
```

### 4. Estructura de Directorio Esperada

```
humero/
├── src/
│   ├── __init__.py
│   ├── mesh/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── discretizer.py
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── curvature.py
│   │   └── differential.py
│   ├── approximation/
│   │   ├── __init__.py
│   │   └── sphere.py
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── refinement.py
│   ├── axis/
│   │   ├── __init__.py
│   │   └── longitudinal.py
│   ├── validation/
│   │   ├── __init__.py
│   │   └── viability.py
│   └── audit/
│       ├── __init__.py
│       └── trail.py
├── tests/
│   ├── test_mesh.py
│   ├── test_geometry.py
│   ├── test_approximation.py
│   ├── test_validation.py
│   └── test_integration.py
├── data/
│   └── sample_humeri/
├── .agent.md
├── .instructions.md
├── AGENTS.md
├── copilot-instructions.md
└── README.md
```

### 5. Validación y Testing

- Crear tests para cada módulo
- Test de integración: carga STL → aproxima esfera → audita
- Verificar que múltiples semillas no divergen
- Comparar aproximaciones con curvatura esperada

### 6. Puntos Críticos a Evitar

❌ **NO**: Usar librerías que no sean Python estándar sin justificación
❌ **NO**: Crear aproximaciones sin auditoría
❌ **NO**: Omitir validación de semillas
❌ **NO**: Dejar código sin documentación
❌ **NO**: Usar globals o estado compartido entre semillas

✅ **SI**: Documentar cada algoritmo
✅ **SI**: Auditar cada aproximación
✅ **SI**: Validar viabilidad antes de aproximar
✅ **SI**: Manejar excepciones explícitamente
✅ **SI**: Usar type hints completos

## Dependencias Esperadas

```
numpy>=1.20
scipy>=1.7
scikit-learn>=1.0  # Para PCA
```

Otras librerías (trimesh, etc.) deben ser justificadas.

## Próximos Pasos

1. Crear estructura de directorios
2. Implementar carga STL (mesh-processor)
3. Implementar cálculo de curvatura (geometry-specialist)
4. Crear sistema AuditTrail (validation-auditor)
5. Implementar algoritmo de aproximación (sphere-optimizer)
6. Implementar eje longitudinal (axis-approximator)
7. Tests de integración
