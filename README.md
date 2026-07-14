# Aproximación de Esfera en Cabeza Humeral

Proyecto académico para cargar modelos STL de húmero, estimar el eje diafisario y aproximar la esfera de la cabeza humeral usando validación geométrica, auditoría y visualización 3D interactiva.

El enfoque actual ya no depende de una sola semilla manual ni de comparar el diámetro de la esfera contra la longitud total del húmero. La validación principal usa métricas morfológicas de artroplastia de hombro y ajuste geométrico a la superficie articular.

## Estado Actual

- Carga de STL ASCII/binario.
- Muestreo uniforme de superficie con normales.
- Aproximación iterativa de esfera desde una semilla.
- Búsqueda automática de best-fit sphere con múltiples semillas.
- Estimación robusta del eje longitudinal desde la diáfisis, sin usar la esfera.
- Auditoría JSON-friendly de semillas, iteraciones, convergencia y métricas.
- Validación morfológica con ROC, medial offset y posterior offset.
- Demo web local para cargar STL, ver la mejor esfera automática y comparar con semillas manuales.
- Suite de tests: `49 passed`.

## Instalación

```bash
cd /home/jeruah/Documentos/University/geometria/humero
pip install -r requirements.txt
```

Dependencias principales:

```text
numpy
scipy
scikit-learn
matplotlib
plotly
pytest
pytest-cov
```

## Uso

Ejecutar tests:

```bash
pytest -q
```

Abrir la demo web sin STL precargado:

```bash
python examples/demo_interactive_web.py
```

Precargar un STL y ejecutar el best-fit automático al iniciar:

```bash
python examples/demo_interactive_web.py \
  --stl data/sample_humeri/HumeroFinal1.stl \
  --samples 8000 \
  --best-fit-seeds 60 \
  --best-fit-top 5
```

Usar datos sintéticos para verificación rápida:

```bash
python examples/demo_interactive_web.py --synthetic-demo
```

La interfaz permite:

- Cargar un STL desde el navegador.
- Discretizar la superficie.
- Calcular automáticamente el best-fit sphere.
- Dibujar la esfera automática, el eje longitudinal y la mejor semilla.
- Hacer clic en un punto real de la cabeza para comparar con una semilla manual.
- Ver score, ROC, RMSE, cobertura, offsets y estado de referencia morfológica.

## Flujo Geométrico Actual

1. `STLLoader` carga la malla desde `src/mesh/loader.py`.
2. `MeshDiscretizer` genera puntos y normales de superficie.
3. `AxisApproximator` estima el eje diafisario:
   - PCA global solo para orientación inicial.
   - Clasificación completo/incompleto por longitud proyectada.
   - Crop proximal/distal si el modelo está completo.
   - Crop proximal si el modelo está incompleto.
   - Slices transversales sobre la región media.
   - Filtro de cortes con spikes de área o perímetro.
   - RANSAC sobre centros de cortes retenidos.
4. `HumeralHeadBestFitSearch` selecciona semillas en la región probable de cabeza:
   - Detecta el extremo de cabeza comparando expansión transversal de ambos extremos.
   - Prioriza puntos alejados del eje para evitar semillas sobre el tallo.
   - Ajusta una esfera por semilla con `SphericalApproximator`.
   - Ordena candidatos por una función de costo.
5. `AuditTrail` registra validación, métricas morfológicas y datos de referencia.

## Best-Fit Automático

La clase principal está en `src/optimization/best_fit.py`:

```python
from src.optimization.best_fit import HumeralHeadBestFitSearch

searcher = HumeralHeadBestFitSearch(
    n_seeds=60,
    top_k=5,
    initial_radius=22.0,
    max_error=2.0,
)
ranking = searcher.search(surface_points, surface_normals)
best = ranking["best"]
```

La puntuación combina:

- RMSE normalizado del ajuste esférico.
- Penalización morfológica por z-scores de ROC, medial offset y posterior offset.
- Cobertura de puntos proximales que caen sobre la superficie de la esfera.
- Penalización por falta de convergencia.
- Penalización si el candidato cae fuera de los rangos de referencia.

La cobertura se calcula contando puntos de la región proximal cuya distancia al centro es compatible con el radio de la esfera dentro de una tolerancia, y cuyas normales no contradicen la dirección radial.

## Métricas Morfológicas

`AuditTrail.compute_morphological_metrics` calcula:

- `roc`: radio de curvatura de la esfera.
- `medial_offset`: componente medial del vector entre eje y centro de esfera.
- `posterior_offset`: componente posterior del mismo vector.
- `total_offset`: magnitud total del desplazamiento transversal.

Rangos de referencia usados como indicadores estadísticos:

| Métrica | Rango de referencia | Media | SD |
|---|---:|---:|---:|
| ROC | 17-30 mm | 22.5 mm | 2.8 mm |
| Medial offset | 1-14 mm | 6.8 mm | 2.5 mm |
| Posterior offset | 0-10 mm | 2.0 mm | 2.0 mm |

Notas importantes:

- La validación dura mantiene ROC plausible en `17-40 mm` y RMSE menor al umbral configurado.
- Los rangos morfológicos no invalidan por defecto; quedan como referencia y z-score.
- Para invalidar por referencia morfológica se debe pasar `enforce_morphology_reference=True`.
- Las direcciones medial/posterior actuales son aproximaciones de marco global; una fase futura debería derivar un marco anatómico específico del lado y orientación del STL.

## Estructura

```text
humero/
├── data/sample_humeri/              # STL de ejemplo
├── examples/
│   ├── demo_interactive_web.py      # Demo principal con carga STL y best-fit
│   ├── demo_visualization.py
│   └── demo_wayland.py
├── src/
│   ├── approximation/sphere.py      # Ajuste iterativo de esfera
│   ├── audit/trail.py               # Auditoría y métricas morfológicas
│   ├── axis/longitudinal.py         # Eje diafisario robusto
│   ├── geometry/                    # Curvatura y análisis diferencial
│   ├── mesh/                        # Carga y discretización STL
│   ├── optimization/
│   │   ├── best_fit.py              # Búsqueda poblacional de esfera
│   │   └── refinement.py
│   ├── validation/viability.py
│   └── visualization/               # Visualización matplotlib/Plotly
└── tests/
    ├── test_audit.py
    ├── test_integration.py
    ├── test_scientific_pipeline.py
    └── test_visualization.py
```

## Resultados de Referencia Rápida

Con 20 semillas y 3000 puntos sobre los STL incluidos:

```text
HumeroFinal1.stl:
  ROC 22.048 mm, RMSE 0.287 mm, cobertura 723 puntos, score 0.571

Right_humerus_bone_one-piece.stl:
  ROC 23.389 mm, RMSE 1.347 mm, cobertura 341 puntos, score 1.084

Human_humerus_2_reduced.stl:
  ROC 31.161 mm, RMSE 1.825 mm, cobertura 131 puntos, score 1.873
```

Estos valores son controles de funcionamiento, no conclusiones clínicas.

## Limitaciones Actuales

- El best-fit es heurístico; mejora la exploración de semillas, pero no sustituye segmentación anatómica validada.
- La detección de cabeza se basa en expansión transversal y puede requerir ajustes en STL muy parciales o mal orientados.
- Las direcciones medial/posterior necesitan un marco anatómico más explícito para estudios bilaterales o poblacionales.
- La cobertura depende de densidad de muestreo y de calidad de normales.

## Próximo Enfoque Recomendado

El cuello de botella ya no es agregar más iteraciones manuales, sino mejorar el modelo geométrico:

1. Segmentar explícitamente región articular proximal antes del ajuste.
2. Estimar un marco anatómico local para medial/posterior.
3. Cambiar el ajuste de esfera por una búsqueda robusta sobre región articular segmentada.
4. Guardar reportes comparables por STL para evaluar sensibilidad a `n_seeds`, tolerancia y muestreo.

## Licencia

Proyecto académico - Universidad.

Última actualización: 2026-07-14
