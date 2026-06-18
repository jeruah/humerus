# 🚀 QUICK START - Proyecto Aproximación de Esfera en Húmero

## ⚡ INICIO RÁPIDO (3 pasos)

```bash
cd /home/jeruah/Documentos/University/geometria/humero

# 1️⃣ Tests
python3 -m pytest tests/ -q

# 2️⃣ Ver visualización interactiva 3D en navegador
python3 examples/demo_interactive_web.py

# 3️⃣ ¡Listo! Comenzar a desarrollar
```

---

## 🖼️ VER VISUALIZACIÓN 3D EN NAVEGADOR

### Código Mínimo (30 segundos)

```python
from src.visualization.interactive_web import InteractiveWeb3D
import numpy as np

viz = InteractiveWeb3D()
viz.plot_sphere(np.array([10, 20, 30]), 25)
viz.plot_axis(np.array([0, 0, 0]), np.array([0, 0, 1]), 100)
viz.show()  # 🎯 Abre en navegador automáticamente
```

### 🎮 CONTROLES (En tu Navegador)

| Control | Acción |
|---------|--------|
| **🖱️ Clic + Arrastrar** | Rotar en 3D |
| **🔍 Rueda del Ratón** | Zoom in/out |
| **➡️ Clic Derecho + Arrastrar** | Pan (mover vista) |
| **🏠 Botón Superior** | Reset vista |
| **👁️ Leyenda** | Mostrar/ocultar capas |

---

## 📦 AUDITOR DE PASOS

```python
from src.audit.trail import AuditTrail
import numpy as np

audit = AuditTrail('seed_001')
audit.log_step('initialize', {'seed': [10, 20, 30]})
is_valid = audit.validate_seed(np.array([10, 20, 30]), np.array([[9, 19, 29], [11, 21, 31]]))
report = audit.get_report()
print(f"Válida: {report['final_valid']}")
```

---

## 🧪 TESTS

```bash
python3 -m pytest tests/ -v    # 36 tests
python3 -m pytest tests/test_audit.py -v
python3 -m pytest tests/test_visualization.py -v
```

---

## 📂 PROYECTO

```
humero/
├── src/
│   ├── audit/trail.py                 ✅ IMPLEMENTADO
│   ├── visualization/
│   │   ├── visualizer.py              ✅ (matplotlib)
│   │   └── interactive_web.py         ✅ NUEVO (Plotly - INTERACTIVO)
│   ├── mesh/ geometry/ approximation/  📋 Para agentes
├── tests/                             ✅ 36 tests
└── examples/demo_interactive_web.py   ✅ NUEVO
```

---

## 💡 EJEMPLOS

### Esfera + Eje (Lo Básico)

```python
from src.visualization.interactive_web import InteractiveWeb3D
import numpy as np

viz = InteractiveWeb3D()
viz.plot_sphere(np.array([0, 0, 0]), 25)
viz.plot_axis(np.array([0, 0, -50]), np.array([0, 0, 1]), 100)
viz.show()
```

### Múltiples Aproximaciones

```python
viz = InteractiveWeb3D()
for i, center in enumerate([[10,20,30], [11,21,31], [10.5,20.5,30.5]]):
    viz.plot_sphere(np.array(center), 25, name=f"Aprox {i+1}")
viz.show()
```

### Con Malla + Eje

```python
vertices = np.random.randn(500, 3) * 20 + np.array([10, 20, 30])
faces = np.random.randint(0, 500, (300, 3))

viz = InteractiveWeb3D()
viz.plot_mesh(vertices, faces)
viz.plot_sphere(np.array([10, 20, 30]), 25)
viz.plot_axis(np.array([10, 20, 0]), np.array([0, 0, 1]), 80)
viz.show()
```

### Guardar HTML

```python
viz = InteractiveWeb3D()
viz.plot_sphere(np.array([0, 0, 0]), 25)
viz.save('/tmp/resultado.html')  # Envía a colegas
```

---

## ✅ VERIFICACIÓN RÁPIDA

```bash
python3 -m pytest tests/ -q
# ✅ 36 passed
```

---

## 🌐 POR QUÉ PLOTLY

✅ Interactivo en navegador (pan, zoom, rotación)
✅ Compatible con Wayland/Hyprland
✅ Sin dependencias GUI complejas
✅ Funciona en cualquier navegador
✅ HTML puro - puedes compartir

---

**¡Sistema listo! 🚀**
