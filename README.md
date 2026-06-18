# Aproximación de Esfera en Húmero mediante Geometría Diferencial

## Descripción del Proyecto

Este proyecto aproxima una esfera a la semi-esfera articular de la cabeza del húmero (donde toca con la clavícula) usando:

- **Discretización** de modelos 3D (archivos STL)
- **Geometría diferencial** para identificar superficies esféricas
- **Algoritmos iterativos** de optimización
- **Sistemas de auditoría** para validación y trazabilidad

### Objetivos Principales

1. ✅ Aproximar con máxima fidelidad la esfera articular del húmero
2. ✅ Validar que las aproximaciones convergen en la región correcta
3. ✅ Aproximar el eje longitudinal del húmero (cabeza a distal)
4. ✅ Auditar cada paso del proceso para reproducibilidad
5. ✅ Manejar robustamente múltiples semillas aleatorias

## Características

### Metodología

#### 1. Carga y Discretización (STL)
```python
mesh = MeshProcessor()
surface_points = mesh.discretize_surface("humerus.stl", n_samples=5000)
```

#### 2. Análisis de Curvatura
```python
analyzer = GeometricAnalyzer()
curvature = analyzer.compute_curvature(point, neighbors)
# Identifica puntos donde κ₁ ≈ κ₂ ≈ 1/R (superficie esférica)
```

#### 3. Aproximación con Auditoría
```python
audit = AuditTrail()
approximator = SphericalApproximator()

for seed in random_seeds_on_articulation:
    if audit.validate_seed(seed):  # ¿Está en región viable?
        sphere = approximator.approximate_from_seed(seed, audit)
        if audit.is_valid_approximation(sphere):
            spheres.append(sphere)

report = audit.get_report()  # Auditoría completa
```

#### 4. Aproximación del Eje
```python
axis_calc = AxisApproximator()
axis = axis_calc.compute_longitudinal_axis(surface_points)
# Retorna: línea parametrizada desde cabeza a distal
```

## Arquitectura de Agentes

El proyecto usa 5 agentes especializados:

| Agente | Especialidad | Archivos |
|--------|--------------|----------|
| **geometry-specialist** | Geometría diferencial, curvatura | `src/geometry/` |
| **mesh-processor** | STL, discretización de mallas | `src/mesh/` |
| **validation-auditor** | Auditoría, validación | `src/validation/`, `src/audit/` |
| **sphere-optimizer** | Optimización iterativa | `src/optimization/` |
| **axis-approximator** | Eje longitudinal, PCA | `src/axis/` |

Ver `AGENTS.md` para detalles.

## Estructura del Proyecto

```
humero/
├── src/
│   ├── __init__.py
│   ├── mesh/
│   │   ├── loader.py          # Cargador de STL
│   │   └── discretizer.py     # Muestreo de superficie
│   ├── geometry/
│   │   ├── curvature.py       # Cálculos de curvatura principal/media
│   │   └── differential.py    # Análisis diferencial
│   ├── approximation/
│   │   └── sphere.py          # Algoritmo de aproximación de esfera
│   ├── optimization/
│   │   └── refinement.py      # Refinamiento iterativo
│   ├── axis/
│   │   └── longitudinal.py    # Eje longitudinal del húmero
│   ├── validation/
│   │   └── viability.py       # Validación de semillas
│   └── audit/
│       └── trail.py           # Sistema de auditoría
├── tests/
│   ├── test_mesh.py
│   ├── test_geometry.py
│   ├── test_approximation.py
│   ├── test_validation.py
│   └── test_integration.py
├── data/
│   └── sample_humeri/         # Modelos STL de ejemplo
├── .agent.md
├── .instructions.md
├── AGENTS.md
├── copilot-instructions.md
└── README.md
```

## Dependencias

```
numpy>=1.20
scipy>=1.7
scikit-learn>=1.0
```

Instalar:
```bash
pip install -r requirements.txt
```

## Uso Básico

```python
from src.mesh.loader import STLLoader
from src.geometry.curvature import CurvatureCalculator
from src.approximation.sphere import SphericalApproximator
from src.audit.trail import AuditTrail

# 1. Cargar modelo
loader = STLLoader()
mesh = loader.load("humerus.stl")

# 2. Discretizar
discretizer = MeshDiscretizer()
surface_points = discretizer.discretize(mesh, n_samples=5000)

# 3. Analizar curvatura
analyzer = CurvatureCalculator()
curvatures = analyzer.compute_all(surface_points)

# 4. Encontrar región articular
articulation_region = analyzer.find_spherical_region(curvatures)

# 5. Aproximar esfera desde múltiples semillas
audit = AuditTrail()
approximator = SphericalApproximator()
spheres = []

for seed in select_random_seeds(articulation_region, n=20):
    if audit.validate_seed(seed, articulation_region):
        sphere = approximator.approximate_from_seed(seed, audit)
        if audit.is_valid_approximation(sphere):
            spheres.append(sphere)

# 6. Generar reporte
report = audit.get_report()
print(f"Esferas aproximadas: {len(spheres)}")
print(f"Centro promedio: {average_centers(spheres)}")
print(f"Radio promedio: {average_radius(spheres)}")
```

## Criterios de Validación

### Semilla Válida
- Punto debe estar en región con curvatura compatible con esfera
- κ₁ ≈ κ₂ ≈ 1/R (donde R = radio esperado ~25-35mm)
- Debe estar dentro de la región articular identificada

### Aproximación Válida
- Algoritmo converge en máximo 100 iteraciones
- Error cuadrático medio (RMSE) < 2mm
- Centro calculado dentro de región articular esperada
- Radio dentro de rango fisiológico (20-40mm)

### Auditoría Completa
- Cada semilla registrada
- Validación de cada semilla auditada
- Convergencia documentada
- Error final registrado
- Reproducibilidad garantizada

## Puntos Críticos

⚠️ **Evitar Divergencias**: Usar restricciones de curvatura para que aproximaciones no diverjan a otras superficies del húmero

⚠️ **Validación de Semillas**: NUNCA aproximar sin validar primero que la semilla está en región viable

⚠️ **Auditoría Obligatoria**: Cada aproximación debe ser auditada; sin auditoría, no se puede validar

## Próximos Pasos

1. [ ] Crear estructura de directorios
2. [ ] Implementar carga STL
3. [ ] Implementar cálculo de curvatura
4. [ ] Crear sistema AuditTrail
5. [ ] Implementar algoritmo de aproximación
6. [ ] Implementar eje longitudinal
7. [ ] Tests de integración
8. [ ] Documentación de resultados

## Referencias

- **Geometría Diferencial**: Do Carmo, M. P. (1976). Differential Geometry of Curves and Surfaces
- **Curvatura en Mallas**: Meyer et al. (2003). Discrete Differential-Geometry Operators for Triangulated 2-Manifolds
- **Optimización Esférica**: Umeyama (1991). Least-Squares Estimation of Transformation Parameters Between Two Point Patterns

## Licencia

Proyecto académico - Universidad

---

**Última actualización**: 2024
**Estado**: En desarrollo
