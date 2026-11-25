# Contador de plantas de plátano y banano con DETR

Taller final de Procesamiento Digital de Imágenes.

La aplicación recibe una foto aérea (tomada con dron, vista desde arriba) de un cultivo de plátano o banano, **encuentra cada planta, dibuja un rectángulo sobre ella y cuenta cuántas hay**. Usa un modelo DETR (*DEtection TRansformer*, ResNet-50 con transformer de 6 capas de encoder y 6 de decoder) entrenado con fotos de cultivos. La interfaz está hecha con Streamlit y corre en el navegador.

**Resultado en el conjunto de prueba** (30 imágenes nunca vistas, 436 plantas reales):

| Métrica | Valor |
|---|---|
| AP@0.5 | 0,924 |
| AP@[.5:.95] | 0,542 |
| Precisión / recall / F1 (umbral 0,9) | 0,886 / 0,858 / 0,872 |
| Error medio de conteo | 1,27 plantas por imagen |
| Conteo total | 422 predichas frente a 436 reales (error de 3,2 %) |

> **Alcance:** el modelo tiene **una sola clase** (`planta`). Detecta y cuenta, pero **no distingue plátano de banano**. Para separarlos habría que reentrenar con dos clases (ver sección 10).

---

## 1. Estructura del proyecto

```
tu_proyecto/
├── Codigo/
│   ├── app.py              Interfaz de Streamlit (subir foto, ver resultado)
│   ├── reconocer.py        Carga el modelo, parte la foto en mosaicos, detecta y dibuja
│   └── detr_modelo.py      Arquitectura DETR (solo para predecir; sin código de entrenamiento)
├── modelos/
│   ├── detr_plantas_mejor.pt   Checkpoint original del cuaderno (entrada de la exportación)
│   └── detr_plantas_front.pt   Modelo que usa la aplicación (lo genera exportar_modelo.py)
├── exportar_modelo.py      Script que se corre una sola vez para generar detr_plantas_front.pt
├── requirements.txt        Dependencias de Python
├── .gitignore              Qué archivos NO se suben al repositorio
├── .gitattributes          Le indica a Git que los .pt de modelos/ van con Git LFS
└── README.md               Este archivo
```

`.gitignore` y `.gitattributes` son archivos ocultos (empiezan por punto) y deben estar **en la raíz del proyecto**, al mismo nivel que este README.

`detr_modelo.py` **debe estar dentro de `Codigo/`**, junto a `app.py` y `reconocer.py`: `reconocer.py` lo importa desde esa misma carpeta.

El entrenamiento se hizo en el cuaderno `DETR_Plantas_UNIBAN.ipynb` (Google Colab, GPU T4). **La aplicación no necesita el cuaderno**, solo los pesos que salieron de él.

---

## 2. Qué se necesita

| Qué | Detalle |
|---|---|
| **Python** | 3.12 recomendado (3.10 o superior) |
| **Bibliotecas** | `torch`, `torchvision`, `streamlit`, `numpy`, `pillow` (las instala `requirements.txt`) |
| **Archivos de código** | `Codigo/app.py`, `Codigo/reconocer.py`, `Codigo/detr_modelo.py` |
| **Modelo entrenado** | `modelos/detr_plantas_front.pt` (se genera a partir de `modelos/detr_plantas_mejor.pt`) |
| **Espacio en disco** | Unos 3 GB con la versión de `torch` con CUDA, o bastante menos con la versión de CPU |
| **Git y Git LFS** | Solo si se obtiene el proyecto con `git clone` (los `.pt` se guardan con Git LFS, ver la sección 11) |
| **Internet** | Solo para instalar las bibliotecas. La aplicación **funciona sin conexión** y no descarga nada al arrancar |
| **GPU** | Opcional. Si hay una con CUDA, la aplicación la usa; si no, corre en CPU (más lento, pero funciona) |

Para **solo usar** la aplicación no se necesita el dataset ni el cuaderno.

---

## 3. Verificar que todo está

Todos los comandos se corren **desde la carpeta del proyecto** (la que contiene este archivo). Los de este documento son para Windows. En Linux o macOS cambie `py` por `python3`, `.venv\Scripts\python` por `.venv/bin/python` y `\` por `/`.

### 3.1 Python

```
py --version
```
Debe mostrar `Python 3.xx.x`. Si dice que no lo encuentra, ver la sección 4.

### 3.2 Entorno virtual y bibliotecas

```
.venv\Scripts\python -c "import torch, torchvision, streamlit, PIL, numpy; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```
Si imprime una línea como `torch 2.x.x | CUDA: False`, todo está instalado (`CUDA: False` solo significa que no hay GPU, y no es un error). Si aparece `ModuleNotFoundError` o no existe `.venv`, ver la sección 4.

### 3.3 Archivos del proyecto

```
.venv\Scripts\python -c "from pathlib import Path; [print(('OK     ' if Path(p).exists() else 'FALTA  ') + p) for p in ['Codigo/app.py','Codigo/reconocer.py','Codigo/detr_modelo.py','modelos/detr_plantas_mejor.pt','modelos/detr_plantas_front.pt']]"
```

Resultado esperado:

```
OK     Codigo/app.py
OK     Codigo/reconocer.py
OK     Codigo/detr_modelo.py
OK     modelos/detr_plantas_mejor.pt      (solo hace falta para generar el siguiente)
OK     modelos/detr_plantas_front.pt      (el imprescindible para usar la aplicación)
```

### 3.4 El modelo carga correctamente

```
.venv\Scripts\python -c "import sys; sys.path.insert(0, 'Codigo'); from reconocer import cargar_modelo; m, i = cargar_modelo(); print(i)"
```
Debe imprimir un diccionario con `clases: ['planta']`, `umbral: 0.9`, `lado: 640`, `consultas: 100`, las métricas y el dispositivo (`cpu` o `cuda`). Si imprime eso, el código y los pesos son compatibles.

---

## 4. Si falta algo

| Qué falta | Qué hacer |
|---|---|
| **Python** | Instalarlo desde https://www.python.org/downloads/ marcando *Add python.exe to PATH* y volver a abrir la terminal |
| **`.venv` o las bibliotecas** | Hacer la instalación de la sección 5, pasos 1 y 2 |
| **`detr_plantas_front.pt` existe pero pesa solo unos bytes (o da `invalid load key`)** | Es el archivito de texto de Git LFS y no el modelo real. Instalar Git LFS y correr `git lfs pull` (sección 11) |
| **Archivos de `Codigo/`** | Copiarlos desde el repositorio. Los tres deben estar en la misma carpeta |
| **`detr_plantas_front.pt`, pero sí está `detr_plantas_mejor.pt`** | Correr `exportar_modelo.py` (sección 5, paso 4) |
| **Ninguno de los dos `.pt`** | Bajar `detr_plantas_mejor.pt` de Google Drive (`MyDrive/DETR_plantas/`, donde lo deja el cuaderno al entrenar) y ponerlo en `modelos/`. Después, correr `exportar_modelo.py` |
| **Tampoco está en Drive** | Hay que volver a entrenar: abrir `DETR_Plantas_UNIBAN.ipynb` en Colab con GPU T4, subir `dataset_plantas_detr.zip` a `MyDrive/DETR_plantas/` y ejecutar todo (unos 40 a 60 minutos). Al terminar se crea `detr_plantas_mejor.pt` |

**Atajo sin exportar:** si ya tiene `detr_plantas_mejor.pt`, puede copiarlo a `modelos/` con el nombre `detr_plantas_front.pt`. La aplicación funciona igual (con umbral 0,9), solo que sin las métricas en el encabezado y con un archivo más grande (unos 166 MB).

---

## 5. Pasos de una sola vez (primera vez en cada computador)

1. **Crear el entorno virtual**
   ```
   py -m venv .venv
   .venv\Scripts\python -m pip install --upgrade pip
   ```

2. **Instalar las bibliotecas**
   ```
   .venv\Scripts\python -m pip install -r requirements.txt
   ```
   *Opcional, si no hay GPU:* para evitar la descarga de CUDA (muy pesada), instale antes la versión de CPU y después el resto:
   ```
   .venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

3. **Poner el checkpoint del cuaderno en `modelos/`**: copiar allí `detr_plantas_mejor.pt` (ver la sección 4 si no lo tiene).

4. **Generar el modelo de la aplicación**
   ```
   .venv\Scripts\python exportar_modelo.py
   ```
   Crea `modelos/detr_plantas_front.pt` (unos 83 MB, pesos en media precisión) con las clases, el umbral y las métricas de prueba. Al final imprime un resumen del archivo generado. No modifica `detr_plantas_mejor.pt`. Con `--fp32` se conserva la precisión completa (unos 166 MB).

   > Las métricas y el umbral (0,9) del script salen del cuaderno actual. **Si se reentrena el modelo**, actualícelos al inicio de `exportar_modelo.py` con los valores nuevos.

5. **Comprobar** con la sección 3.3 y 3.4. Con eso queda listo y **no hay que repetir nada de esto**.

---

## 6. Uso normal (cuando ya está todo instalado)

Solo hay que iniciar la aplicación:

```
.venv\Scripts\streamlit run Codigo\app.py
```

Se abre una página en el navegador (si no, ir a http://localhost:8501). Si la primera vez pregunta por un correo en la terminal, presione Enter. Para cerrarla, `Ctrl + C` en la terminal.

**Cómo usarla:**

1. Suba una foto aérea (JPG o PNG) en «Abrir una imagen aérea».
2. La aplicación muestra la foto con un rectángulo rojo por planta y su confianza, y a la derecha el **número de plantas detectadas**, el tiempo de procesamiento y la confianza media. Abajo hay una tabla con el detalle de cada planta.
3. Ajustes en la barra lateral:
   * **Umbral de confianza** (por defecto 0,90): solo se cuentan las plantas por encima de ese valor. Si faltan plantas, bájelo; si aparecen detecciones falsas, súbalo.
   * **Mosaicos por lado (N × N)**: el modelo mira 640×640 píxeles por vez y detecta como máximo 100 plantas por mosaico. Las fotos grandes (más de ~1000 px) se parten en mosaicos para que las plantas no queden diminutas. La aplicación propone un valor inicial según el tamaño de la foto, y se puede cambiar.
4. El botón **Descargar imagen con detecciones** guarda la foto anotada en PNG.

La aplicación no escribe nada en disco ni guarda las fotos que se suben.

---

## 7. Cómo probar que funciona

**Nivel 1: el modelo carga.** Comando de la sección 3.4.

**Nivel 2: contar en una imagen conocida, sin abrir la aplicación.** Use una imagen del conjunto de prueba del dataset (`dataset_plantas/images/test/`), que tiene su archivo de etiquetas en `labels/test/` con **una línea por planta real**:

```
.venv\Scripts\python -c "import sys; sys.path.insert(0, 'Codigo'); from PIL import Image; from reconocer import cargar_modelo, detectar; m, i = cargar_modelo(); d, ms, _ = detectar(Image.open('RUTA/A/LA/IMAGEN.jpg'), m, i); print(len(d), 'plantas detectadas en', round(ms), 'ms')"
```

El número debe ser parecido a las líneas del `.txt` de esa imagen (error típico de 1 o 2 plantas, según el resultado de prueba de arriba).

**Nivel 3: la aplicación completa.** Abra la aplicación, suba esa misma imagen con 1 mosaico (las imágenes del dataset ya son de 640×640) y confirme que el conteo coincide con el del nivel 2. Después pruebe:

* Subir una foto aérea propia de un cultivo y revisar que los rectángulos caigan sobre las plantas.
* Subir una foto más grande y comparar el resultado con 1 mosaico y con 2×2.
* Subir una foto que no sea de un cultivo: debería detectar muy pocas plantas o ninguna.
* Mover el umbral y ver cómo cambia el conteo.

---

## 8. Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `FileNotFoundError ... detr_plantas_front.pt` | Falta el modelo. Ver la sección 4 |
| `ModuleNotFoundError: No module named 'torch'` (u otra biblioteca) | Se usó otro Python o no se instaló. Instalar con la sección 5, paso 2, y correr con `.venv\Scripts\...` |
| `ModuleNotFoundError: No module named 'detr_modelo'` | `detr_modelo.py` no está en la carpeta `Codigo/` |
| `invalid load key` o `UnpicklingError` al cargar el `.pt` | El archivo es el puntero de Git LFS y no el modelo. Correr `git lfs pull` |
| `Missing key(s)` o `size mismatch` al cargar el modelo | El `.pt` no corresponde a este código (otro modelo o configuración distinta). Regenerarlo con `exportar_modelo.py` a partir del checkpoint del cuaderno |
| `streamlit` no se reconoce como comando | Correrlo como `.venv\Scripts\streamlit run Codigo\app.py`, o `.venv\Scripts\python -m streamlit run Codigo\app.py` |
| El puerto 8501 está ocupado | `.venv\Scripts\streamlit run Codigo\app.py --server.port 8502` |
| Tarda mucho | En CPU puede tardar varios segundos por mosaico, y más si se usan muchos. Es normal sin GPU |
| No detecta casi nada | Bajar el umbral, o la foto no es cenital / las plantas se ven muy pequeñas (usar menos mosaicos) |
| Detecta plantas en cosas que no lo son | Subir el umbral |
| Aviso de «cerca del máximo de plantas» | Un mosaico tiene demasiadas plantas para las 100 consultas del modelo. Aumentar el número de mosaicos |

---

## 9. Cómo funciona por dentro (resumen)

1. `app.py` lee la foto y los ajustes de la barra lateral.
2. `reconocer.py` la divide en N × N mosaicos, redimensiona cada uno a 640 × 640 (igual que en el entrenamiento) y lo normaliza con las medias de ImageNet.
3. `detr_modelo.py` define el modelo: ResNet-50, proyección 1×1, codificación posicional seno/coseno, encoder y decoder de 6 capas con 100 consultas, y las cabezas de clase y caja.
4. Cada consulta con probabilidad de «planta» mayor al umbral cuenta como una planta. **No se usa NMS**: DETR asigna una consulta distinta a cada objeto durante el entrenamiento.
5. Las cajas se convierten a las coordenadas de la foto original y se dibujan.

---

## 10. Limitaciones

* **Una sola clase:** cuenta plantas, pero no separa plátano de banano. Separarlas exige reentrenar el modelo con dos clases y más imágenes de banano.
* **Máximo 100 plantas por mosaico**, y le cuestan las plantas muy pequeñas. Por eso las fotos grandes se parten en mosaicos.
* **Mosaicos:** una planta cortada por la línea de división puede contarse dos veces o no detectarse. El conteo total es aproximado en esas zonas.
* **Pocos datos:** se entrenó con 391 imágenes y se probó con solo 30 (de 143 fotos originales), así que las métricas tienen bastante variación y no se pueden extrapolar a cualquier cultivo, altura de vuelo o iluminación.
* **Tipo de foto:** está pensado para fotos cenitales de dron. Fotos oblicuas o tomadas a otra altura darán peores resultados.
* El umbral de 0,9 es alto: el conteo total sale muy cercano al real porque los falsos positivos y los falsos negativos se compensan, pero el error por imagen (1,27 plantas) es la cifra más honesta.

---

## 11. Subir el proyecto a GitHub (con Git LFS)

GitHub **rechaza archivos de más de 100 MB** y avisa desde los 50 MB. Los modelos (`detr_plantas_mejor.pt`, unos 166 MB, y `detr_plantas_front.pt`, unos 83 MB) se suben con **Git LFS**, que guarda el archivo grande aparte y deja en el repositorio solo un puntero de texto.

### Qué se sube y qué no

| Se sube | No se sube |
|---|---|
| `Codigo/`, `exportar_modelo.py`, `requirements.txt`, `README.md`, `.gitignore`, `.gitattributes` | `.venv/`, `__pycache__/` |
| `modelos/*.pt` (por Git LFS) | El dataset (`dataset_plantas/`, `dataset_plantas_detr.zip`) |
| | Archivos del sistema (`Thumbs.db`, `.DS_Store`, ...) |

### `.gitignore` (en la raíz del proyecto)

```
# Entorno y archivos temporales de Python
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/

# Dataset de entrenamiento: pesa mucho y no hace falta para usar la aplicación
dataset_plantas/
dataset_plantas_detr.zip

# Sistema
Thumbs.db
desktop.ini
.DS_Store
```

Los `.pt` de `modelos/` **no** se ignoran: van por Git LFS.

### `.gitattributes` (en la raíz del proyecto)

```
modelos/*.pt filter=lfs diff=lfs merge=lfs -text
```

Este archivo lo crea solo el comando `git lfs track` del paso 2, pero debe subirse al repositorio.

### Pasos (una sola vez, por quien sube el modelo)

1. **Instalar Git LFS** y activarlo en el usuario de Git:
   ```
   git lfs version
   git lfs install
   ```
   Si el primer comando no se reconoce: `winget install GitHub.GitLFS` (Windows), `brew install git-lfs` (macOS) o `sudo apt install git-lfs` (Linux), y volver a abrir la terminal.

2. **Marcar los modelos como archivos LFS, antes de añadirlos:**
   ```
   git lfs track "modelos/*.pt"
   git add .gitattributes .gitignore
   ```

3. **Subir el proyecto:**
   ```
   git add .
   git commit -m "Aplicación de detección de plantas con DETR"
   git push
   ```

4. **Comprobar** que los modelos quedaron en LFS:
   ```
   git lfs ls-files
   ```
   Deben aparecer `modelos/detr_plantas_mejor.pt` y `modelos/detr_plantas_front.pt`.

### Si el push fue rechazado por el tamaño

El archivo grande quedó en un commit hecho antes de activar LFS. Si es el **último commit y aún no se ha subido**:

```
git reset --soft HEAD~1
git restore --staged modelos/detr_plantas_mejor.pt
git lfs track "modelos/*.pt"
git add .gitattributes modelos/
git commit -m "Aplicación de detección de plantas con DETR"
git push
```

Si el archivo está en varios commits, hay que reescribir el historial con `git lfs migrate import --include="modelos/*.pt"` y luego `git push --force-with-lease`. Esto cambia el historial para todo el equipo, así que hay que avisarles antes.

### Para quien clona el repositorio

1. Instalar Git LFS y correr `git lfs install` (una vez por computador) **antes** de clonar.
2. Clonar con `git clone` (no con «Download ZIP», que puede traer solo los punteros).
3. Si `modelos/detr_plantas_front.pt` pesa solo unos bytes, correr `git lfs pull` dentro del proyecto.
4. Seguir con la sección 3 para verificar y la sección 6 para usar la aplicación.

### Cuotas de Git LFS

* **Tamaño por archivo:** hasta 2 GB en GitHub Free.
* **Almacenamiento y ancho de banda:** la documentación de GitHub indica 10 GiB de cada uno en el plan Free (una página antigua dice 1 GiB; conviene revisar la cuota real en *Settings → Billing and plans*). Cada versión nueva del modelo que se suba suma al almacenamiento, y cada clon o `git lfs pull` suma al ancho de banda del dueño del repositorio.
* **Alternativa sin LFS:** dejar los `.pt` en Google Drive o en GitHub Releases, y poner el enlace aquí para descargarlos a `modelos/`.

---

## 12. Créditos y fuentes

* **Modelo:** DETR (Carion et al., Facebook AI Research). Se parte de los pesos oficiales de DETR-R50 entrenados en COCO (*transfer learning*) y se reentrena con fotos de cultivos.
* **Datos:** muestra del proyecto de aforamiento digital de UNIBAN (`afflorezc/aforamiento-app-dataset`), que combina los datasets de Roboflow *platano-lasuiza* (Daniel F. Tovar) y *Banana Tree Detection* (Elliot Su).
* **Interfaz:** adaptada del clasificador de piezas Lego de la Parte 2 de la Tarea 2 del mismo equipo.
