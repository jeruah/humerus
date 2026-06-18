# Guía de Agentes Especializados

## Invocar Agentes

Cada agente es un especialista en su dominio. Invócalos como:

```bash
# Ejemplo: pedirle al geometry-specialist que implemente curvatura
@geometry-specialist Implementar el cálculo de curvatura principal y media en una malla triangular

# Ejemplo: pedirle al validation-auditor que defina el sistema de auditoría
@validation-auditor Diseñar e implementar el sistema AuditTrail para registrar cada paso
```

## Agentes Disponibles

### 1️⃣ **geometry-specialist**
**Dominio**: Geometría diferencial, matemáticas computacionales, cálculos de curvatura

**Responsabilidades**:
- Implementar cálculo de curvatura principal (κ₁, κ₂)
- Implementar cálculo de curvatura media (H = (κ₁ + κ₂)/2)
- Implementar cálculo de curvatura Gaussiana (K = κ₁ × κ₂)
- Implementar algoritmos de ajuste esférico (least-squares)
- Implementar métodos de optimización local para encontrar centro y radio

**Archivos a crear/modificar**:
- `src/geometry/curvature.py`
- `src/geometry/differential.py`
- `src/approximation/sphere.py`

**Criterios de éxito**:
- Curvatura calculada con precisión ≥ 6 decimales
- Algoritmo convergente en máx 100 iteraciones
- Documentación matemática completa

---

### 2️⃣ **mesh-processor**
**Dominio**: Procesamiento de modelos 3D, STL, discretización, muestreo

**Responsabilidades**:
- Cargar archivos STL (ASCII y binarios)
- Extraer vértices y facetas
- Implementar muestreo uniforme de superficie
- Implementar muestreo adaptativo (más denso en regiones de curvatura alta)
- Calcular normales de superficie
- Identificar región articular de cabeza del húmero

**Archivos a crear/modificar**:
- `src/mesh/loader.py`
- `src/mesh/discretizer.py`

**Criterios de éxito**:
- Carga correcta de STL ASCII y binarios
- Muestreo sin perdida de características articulares
- Normales calculadas correctamente
- Manejo de mallas con millones de triángulos

---

### 3️⃣ **validation-auditor**
**Dominio**: Validación, auditoría, trazabilidad, logging

**Responsabilidades**:
- Diseñar clase `AuditTrail` para registrar cada paso
- Implementar validación de semillas (¿está en región articular?)
- Implementar validación de convergencia (¿el algoritmo convergió?)
- Implementar validación de esfera aproximada (¿error aceptable?)
- Generar reportes de auditoría
- Detectar aproximaciones divergentes

**Archivos a crear/modificar**:
- `src/validation/viability.py`
- `src/audit/trail.py`

**Criterios de éxito**:
- Cada paso del algoritmo queda registrado
- Auditoría permite reproducir exactamente qué sucedió
- Reportes generan datos para análisis post-hoc
- Validación previene divergencias

---

### 4️⃣ **sphere-optimizer**
**Dominio**: Optimización, refinamiento iterativo, algoritmos de convergencia

**Responsabilidades**:
- Implementar algoritmo iterativo de refinamiento de esfera
- Usar múltiples semillas aleatorias
- Implementar criterios de convergencia
- Manejar casos de no-convergencia
- Implementar early-stopping
- Generar estadísticas de convergencia

**Archivos a crear/modificar**:
- `src/optimization/refinement.py`
- `src/approximation/sphere.py`

**Criterios de éxito**:
- Convergencia garantizada con semillas válidas
- Manejo robusto de casos patológicos
- Estadísticas de convergencia disponibles
- Capaz de procesar 100+ semillas sin fallos

---

### 5️⃣ **axis-approximator**
**Dominio**: Análisis de componentes principales, geometría 3D, eje longitudinal

**Responsabilidades**:
- Implementar PCA para encontrar eje principal del húmero
- Validar que eje va desde cabeza a distal
- Parametrizar eje como línea 3D
- Calcular distancia de puntos al eje
- Generar métricas de alineación

**Archivos a crear/modificar**:
- `src/axis/longitudinal.py`

**Criterios de éxito**:
- Eje identificado correctamente
- Parametrización robusta
- Métricas de validación disponibles

---

## Flujo de Trabajo con Agentes

```
START
  ↓
[mesh-processor] 
  → Carga STL
  → Discretiza superficie
  → Identifica región articular
  ↓
[geometry-specialist] + [axis-approximator] (en paralelo)
  → Calcula curvatura (geometry)
  → Calcula eje longitudinal (axis)
  ↓
[validation-auditor] 
  → Prepara sistema de auditoría
  ↓
FOR CADA SEMILLA ALEATORIA:
  ↓
  [validation-auditor] 
    → ¿Semilla en región viable?
    → SI → siguiente paso, NO → rechazar
  ↓
  [sphere-optimizer]
    → Aproximar esfera desde semilla
    → Registrar convergencia
  ↓
  [validation-auditor]
    → ¿Aproximación válida?
    → ¿Error aceptable?
    → Registrar en auditoría
  ↓
END FOR
  ↓
[validation-auditor]
  → Generar reporte final
  → Análisis comparativo de todas las semillas
  ↓
END
```

## Criterios de Integración

- Los agentes deben usar la misma clase `AuditTrail` de validation-auditor
- Todos los módulos deben ser 100% documentados
- Usar type hints en Python 3.8+
- Excepciones siempre deben ser auditadas
- Logs estructurados (JSON-friendly)

## Preguntas Frecuentes

**P: ¿Un agente puede invocar a otro?**
R: Sí, pero coordina a través de tus instrucciones. Ejemplo: mesh-processor entrega datos a geometry-specialist.

**P: ¿Qué pasa si un agente falla?**
R: Usa validation-auditor para detectar y registrar fallos. Reasigna trabajo manualmente si es necesario.

**P: ¿Cómo validar que todo funciona junto?**
R: Crea tests de integración que usen todos los agentes en secuencia.
