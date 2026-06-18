# 🌐 Visualizador 3D Interactivo en Navegador

## Qué es

`InteractiveWeb3D` es un visualizador 3D completamente interactivo que funciona en tu navegador web. 

**No requiere**:
- ❌ GTK, Qt, Tk instalados
- ❌ Permisos de root
- ❌ Configuraciones complejas
- ❌ Conexiones display
- ❌ X11 o Wayland especiales

**Solo necesita**:
- ✅ Python 3.7+
- ✅ Plotly (descargado con `pip install plotly`)
- ✅ Un navegador web (ya tienes uno)

## Uso Rápido

```python
from src.visualization.interactive_web import InteractiveWeb3D
import numpy as np

viz = InteractiveWeb3D()
viz.plot_sphere(np.array([0, 0, 0]), 25, name="Esfera")
viz.plot_axis(np.array([0, 0, -50]), np.array([0, 0, 1]), 100, name="Eje")
viz.show()  # 🎯 Abre en navegador automáticamente
```

## Controles Interactivos

| Acción | Control |
|--------|---------|
| **Rotar** | Clic izquierdo + arrastrar |
| **Zoom** | Rueda del ratón o pellizco en trackpad |
| **Pan** | Clic derecho + arrastrar |
| **Reset** | Botón "Home" en esquina superior |
| **Toggle Capas** | Click en leyenda (derecha) |

## Métodos Disponibles

### `plot_sphere(center, radius, name='Esfera')`
Grafica una esfera en ROJO.

```python
viz.plot_sphere(
    center=np.array([10, 20, 30]),  # Centro 3D
    radius=25.0,                     # Radio en mm
    name="Mi Esfera"                 # Nombre en leyenda
)
```

### `plot_axis(origin, direction, length, name='Eje')`
Grafica el eje longitudinal en ROJO.

```python
viz.plot_axis(
    origin=np.array([0, 0, 0]),      # Punto inicial
    direction=np.array([0, 0, 1]),   # Dirección (se normaliza)
    length=100,                       # Longitud en mm
    name="Eje Principal"
)
```

### `plot_mesh(vertices, faces, name='Malla')`
Grafica una malla triangular (azul).

```python
vertices = np.array([[...], [...], ...])  # (N, 3)
faces = np.array([[0, 1, 2], ...])        # (M, 3)

viz.plot_mesh(vertices, faces, name="Húmero")
```

### `plot_seeds(seeds, name='Semillas')`
Grafica puntos semilla.

```python
seeds = np.array([
    [10, 20, 30],
    [11, 21, 31],
    ...
])
viz.plot_seeds(seeds, name="Puntos de inicio")
```

### `plot_approximations(approximations, name='Aproximaciones')`
Grafica múltiples esferas aproximadas.

```python
approximations = [
    {'center': np.array([10, 20, 30]), 'radius': 25.0, 'error': 0.45},
    {'center': np.array([11, 21, 31]), 'radius': 24.5, 'error': 0.38},
]
viz.plot_approximations(approximations)
```

### `show()`
Genera HTML interactivo y abre en navegador.

```python
viz.show()  # Abre automáticamente en tu navegador
```

### `save(filepath)`
Guarda el HTML para compartir.

```python
viz.save('/tmp/resultado.html')
# Envía el archivo a colegas - ¡abre en cualquier navegador!
```

## Ejemplos

### Ejemplo 1: Básico
```python
from src.visualization.interactive_web import InteractiveWeb3D
import numpy as np

viz = InteractiveWeb3D(title="Mi Visualización")
viz.plot_sphere(np.array([0, 0, 0]), 20)
viz.show()
```

### Ejemplo 2: Malla + Esfera + Eje
```python
# Datos
vertices = np.random.randn(500, 3) * 20
faces = np.random.randint(0, 500, (300, 3))

viz = InteractiveWeb3D()
viz.plot_mesh(vertices, faces, name="Hueso")
viz.plot_sphere(np.array([0, 0, 0]), 25, name="Esfera")
viz.plot_axis(np.array([0, 0, -40]), np.array([0, 0, 1]), 80)
viz.show()
```

### Ejemplo 3: Comparar Aproximaciones
```python
viz = InteractiveWeb3D()

for i, approx in enumerate(my_approximations):
    viz.plot_sphere(
        approx['center'],
        approx['radius'],
        name=f"Aprox {i+1} (Error={approx['error']:.2f})"
    )

viz.plot_axis(origin, direction, length)
viz.show()
```

### Ejemplo 4: Guardar para Compartir
```python
viz = InteractiveWeb3D()
# ... agregar datos ...
viz.save('/tmp/mi_resultado.html')
print("Compartir: /tmp/mi_resultado.html")
```

## Por qué Plotly

✅ **Interactividad**
- Pan, zoom, rotación suave
- Leyenda clicable
- Hover info

✅ **Compatibilidad**
- Wayland/Hyprland sin problemas
- Windows, macOS, Linux
- Cualquier navegador moderno

✅ **Sin dependencias pesadas**
- Genera HTML puro
- No necesita servidor
- Funciona offline

✅ **Profesional**
- Render 3D real
- Colores y transparencia
- Exportable

## Comparación con matplotlib

| Característica | Plotly (InteractiveWeb3D) | Matplotlib (Visualizer3D) |
|---|---|---|
| **Interactividad** | ✅ Completa (pan, zoom, rotate) | ❌ Estática |
| **Wayland** | ✅ Perfecto | ⚠️ Problemas con backends |
| **GUI pesada** | ❌ No necesita | ✅ Necesita GTK/Qt/Tk |
| **Compartir** | ✅ HTML (abre en navegador) | ⚠️ PNG estático |
| **Tests** | ⚠️ Requiere navegador | ✅ Headless (Agg) |

## Troubleshooting

**P: No se abre navegador**
```python
# Abre manualmente el archivo generado
viz.save('/tmp/viz.html')
# Luego abre /tmp/viz.html en tu navegador
```

**P: Plotly no importa**
```bash
pip install plotly --break-system-packages
```

**P: Quiero versión offline**
```python
# Plotly ya genera HTML sin dependencias externas
# El HTML funciona completamente offline
viz.save('offline.html')  # Abre en navegador sin internet
```

**P: Controles no responden**
- Asegúrate de mover el ratón dentro del área 3D
- Prueba con navegadores moderno (Chrome, Firefox, Edge)

## Conclusión

`InteractiveWeb3D` es la forma moderna de visualizar datos 3D.
Es perfecta para:
- 🖥️ Desarrollo en Linux con Wayland
- 👥 Compartir resultados con colegas
- 📊 Presentaciones interactivas
- 🔍 Exploración de datos 3D

¡Disfruta la visualización interactiva! 🎉
