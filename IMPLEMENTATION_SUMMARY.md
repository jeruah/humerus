# Resumen de Implementación

Última actualización: 2026-07-14

Este documento resume el estado real del proyecto después de las iteraciones recientes. La documentación de agentes IA, Copilot y quickstart fue retirada porque ya no representa el flujo de trabajo ni ayuda a mejorar la precisión geométrica.

## Limpieza de Documentación

Se eliminaron:

- `.agent.md`
- `.instructions.md`
- `AGENTS.md`
- `copilot-instructions.md`
- `QUICK_START.md`
- `STATUS.md`

La documentación viva queda concentrada en:

- `README.md`: guía principal de uso, arquitectura, métricas y limitaciones.
- `IMPLEMENTATION_SUMMARY.md`: estado de implementación y decisiones técnicas.

## Capacidades Implementadas

### Carga y Muestreo STL

Archivos:

- `src/mesh/loader.py`
- `src/mesh/discretizer.py`

Estado:

- Carga STL ASCII y binario.
- Extrae vértices, caras y normales.
- Discretiza superficie con muestreo uniforme.
- Entrega `surface_points` y `surface_normals` para el pipeline geométrico.

### Ajuste de Esfera Desde Semilla

Archivo:

- `src/approximation/sphere.py`

Estado:

- Implementa `SphericalApproximator`.
- Ajusta esfera por least-squares local.
- Itera desde una semilla de superficie.
- Registra inicialización, iteraciones y resultado en `AuditTrail`.
- Controla convergencia por cambio de centro/radio.

### Best-Fit Automático Poblacional

Archivo:

- `src/optimization/best_fit.py`

Clase:

- `HumeralHeadBestFitSearch`

Estado:

- Toma `n_seeds` puntos candidatos.
- Detecta el extremo probable de cabeza comparando expansión transversal en ambos extremos del eje.
- Prioriza semillas alejadas del eje para evitar el tallo.
- Ajusta una esfera por semilla.
- Evalúa cada candidata con una función de costo.
- Retorna `best`, `top_candidates`, conteo de candidatos válidos, cobertura y auditoría.

Componentes del score:

- RMSE normalizado.
- Penalización morfológica por z-scores.
- Penalización por baja cobertura de superficie proximal.
- Penalización por falta de convergencia.
- Penalización por estar fuera de rangos de referencia.

### Eje Longitudinal Robusto

Archivo:

- `src/axis/longitudinal.py`

Método por defecto:

- `diaphyseal_slice_axis`

Estado:

- No usa la esfera para estimar el eje.
- Usa PCA global solo para orientación inicial.
- Neutraliza densidad mediante voxel downsample.
- Detecta si el húmero parece completo por longitud proyectada.
- En modelo completo descarta cabeza y cola.
- En modelo incompleto descarta solo la porción proximal.
- Divide la región de interés en slices.
- Elimina slices con spikes de área o perímetro.
- Ajusta el eje final con RANSAC sobre centros de slices.
- Devuelve diagnósticos: completitud, crop, slices retenidos, inliers/outliers RANSAC.

### Auditoría y Validación Morfológica

Archivo:

- `src/audit/trail.py`

Estado:

- Registra pasos en formato JSON-friendly.
- Valida semillas.
- Valida aproximaciones por RMSE y ROC plausible.
- Calcula métricas morfológicas:
  - ROC.
  - Medial offset.
  - Posterior offset.
  - Total offset transversal.
- Reporta rangos de referencia, medias, desviaciones estándar y z-scores.

Referencias configuradas:

| Métrica | Rango | Media | SD |
|---|---:|---:|---:|
| ROC | 17-30 mm | 22.5 mm | 2.8 mm |
| Medial offset | 1-14 mm | 6.8 mm | 2.5 mm |
| Posterior offset | 0-10 mm | 2.0 mm | 2.0 mm |

Decisión de validación:

- Los rangos de referencia morfológica son indicadores por defecto.
- No invalidan una esfera salvo que se use `enforce_morphology_reference=True`.
- La validación dura mantiene RMSE y ROC plausible como criterios principales.

### Demo Web Interactiva

Archivo:

- `examples/demo_interactive_web.py`

Estado:

- Permite cargar STL desde navegador.
- Discretiza la superficie.
- Ejecuta best-fit automático al cargar.
- Dibuja superficie, mejor esfera, semilla automática y eje.
- Permite hacer clic en una semilla manual para comparar.
- Muestra:
  - Score best-fit.
  - ROC.
  - RMSE.
  - Cobertura.
  - Candidatos válidos.
  - Longitud de eje.
  - Completo/incompleto.
  - Modo de crop.
  - RANSAC inliers.
  - Medial offset.
  - Posterior offset.
  - Estado y z-scores de referencia.

Parámetros CLI relevantes:

```bash
--stl
--samples
--synthetic-demo
--initial-radius
--max-error
--best-fit-seeds
--best-fit-top
--host
--port
--no-browser
```

## Estado de Tests

Comando:

```bash
pytest -q
```

Resultado actual:

```text
49 passed
```

Cobertura funcional en tests:

- Auditoría y serialización.
- Visualización 3D.
- Carga STL ASCII/binaria.
- Discretización uniforme.
- Curvatura en región esférica.
- Estimación de eje robusta en húmero completo e incompleto.
- Validación de offsets morfológicos.
- Respuesta JSON de la demo web.
- Best-fit automático recuperando una esfera sintética conocida.

## Resultados de Smoke Test en STL de Muestra

Con 20 semillas y 3000 puntos:

```text
Human_humerus_2_reduced.stl
  candidatos válidos: 4/20
  ROC: 31.161 mm
  RMSE: 1.825 mm
  cobertura: 131 puntos
  score: 1.873

HumeroFinal1.stl
  candidatos válidos: 14/20
  ROC: 22.048 mm
  RMSE: 0.287 mm
  cobertura: 723 puntos
  score: 0.571

Right_humerus_bone_one-piece.stl
  candidatos válidos: 13/20
  ROC: 23.389 mm
  RMSE: 1.347 mm
  cobertura: 341 puntos
  score: 1.084
```

Estos smoke tests verifican comportamiento del pipeline; no deben interpretarse como validación clínica.

## Decisiones Técnicas Relevantes

### La esfera ya no define el eje

El eje depende solo de la nube de puntos y del análisis diafisario. Esto evita el ciclo lógico donde la validez de la esfera depende del eje y el eje depende de una esfera potencialmente inválida.

### Modelos completos e incompletos

El eje intenta adaptarse a ambos:

- Completo: descarta extremos proximal y distal.
- Incompleto: descarta la porción proximal, conserva más tallo distal disponible.

### Morfología como referencia, no bloqueo automático

Los rangos de ROC/MO/PO ayudan a ordenar y auditar candidatos, pero no invalidan por sí solos salvo modo estricto. Esto evita rechazar geometrías plausibles por variabilidad anatómica o por errores de marco anatómico.

### Best-fit por ranking, no por primera convergencia

El resultado automático no es "la primera esfera que funciona"; es la mejor candidata según score combinado. Esto permite comparar semillas y detectar candidatos razonables aunque algunas semillas fallen.

## Limitaciones Pendientes

- Falta segmentación explícita de la superficie articular de la cabeza.
- El marco medial/posterior sigue siendo aproximado.
- La detección de cabeza por expansión transversal puede fallar en STL muy parciales.
- La cobertura depende de densidad de muestreo y calidad de normales.
- El best-fit es heurístico; necesita validación sistemática con casos anotados.

## Próximos Pasos Recomendados

1. Crear un módulo de segmentación de cabeza humeral antes del ajuste esférico.
2. Estimar marco anatómico local para medial/posterior en lugar de usar ejes globales.
3. Guardar reportes comparables por STL en JSON/CSV.
4. Evaluar sensibilidad de resultados a `samples`, `best_fit_seeds` y tolerancia de cobertura.
5. Incorporar casos con ground truth o anotaciones manuales para medir precisión real.
