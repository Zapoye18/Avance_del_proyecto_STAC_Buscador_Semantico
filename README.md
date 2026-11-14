# STAC Buscador Semántico

Proyecto para clasificar documentos y realizar búsquedas semánticas sobre un corpus documental. La clasificación utiliza TF-IDF y modelos supervisados, mientras que la búsqueda utiliza TF-IDF y similitud coseno.

## Requisitos

- Docker Desktop con Docker Compose.
- Git, si se desea clonar o actualizar el repositorio.
- Para ejecutar el proyecto sin Docker: Python 3.11, 3.12 o 3.13.

La forma recomendada de ejecución es Docker, porque instala las dependencias y mantiene consistente el entorno.

## Estructura principal

```text
app/                 Código de la API, extracción, clasificación y búsqueda
data/corpus/         Documentos organizados por departamento
data/incoming/       Documentos nuevos para clasificar
data/training/       Datasets generados para entrenamiento
models/classifier/   Modelos entrenados y métricas
scripts/             Preparación, entrenamiento e inspección
tests/               Pruebas automatizadas
compose.yaml         Configuración de Docker Compose
```

Las categorías esperadas en `data/corpus/` son:

```text
diseno/
ingenieria/
audiovisual/
calidad/
operacion/
negocio/
```

Cada carpeta debe contener documentos `.txt`.

## Inicio rápido con Docker

Desde la raíz del proyecto, construir la imagen:

```powershell
docker compose build api
```

Iniciar la API:

```powershell
docker compose up api
```

La API queda disponible en `http://127.0.0.1:8000` y su documentación interactiva en `http://127.0.0.1:8000/docs`.

Para detener el servicio:

```powershell
docker compose down
```

## Preparar el corpus

Antes de entrenar un modelo, generar los datasets a partir del corpus:

```powershell
docker compose run --rm api python -m scripts.prepare_corpus
```

El script valida las carpetas, extrae los documentos y genera:

```text
data/training/documents_body.csv
data/training/documents_title_body.csv
```

El dataset principal para el entrenamiento actual es `documents_body.csv`. La etiqueta de cada documento se obtiene de la carpeta donde está almacenado.

## Entrenar los modelos

Para comparar Naive Bayes, SVM y Regresión Logística:

```powershell
docker compose run --rm api python -m scripts.train
```

Para entrenar la variante experimental SVM v1:

```powershell
docker compose run --rm api python -m scripts.train_svm_v1
```

Los modelos y sus métricas se guardan en `models/classifier/`.

La API de clasificación utiliza específicamente `models/classifier/svm_v1.joblib`, por lo que este archivo debe existir antes de utilizar `/classify`.

## Clasificar un documento desde la terminal

Colocar el archivo `.txt` en `data/incoming/` y ejecutar:

```powershell
docker compose run --rm api python -m scripts.predict_document --document data/incoming/documento_prueba.txt
```

La ruta debe incluir la extensión `.txt`. El resultado muestra el nombre del archivo, el departamento predicho y la versión del modelo.

El documento se procesa temporalmente y no se agrega al corpus original.

## Buscar documentos desde la terminal

Para realizar una búsqueda sobre el corpus:

```powershell
docker compose run --rm api python -m scripts.search_terminal --query "problemas de latencia del servidor" --top-k 5
```

`--query` contiene la consulta en lenguaje natural y `--top-k` indica el número máximo de resultados. El índice se construye a partir de `data/corpus/` y los resultados se ordenan por similitud.

## Usar la API

### Comprobar el estado

```powershell
curl http://127.0.0.1:8000/health
```

### Listar documentos

```powershell
curl http://127.0.0.1:8000/documents
```

### Consultar un documento

```powershell
curl http://127.0.0.1:8000/documents/nombre_del_documento.txt
```

### Normalizar texto

```powershell
curl -X POST http://127.0.0.1:8000/preprocess `
  -H "Content-Type: application/json" `
  -d '{"text":"Texto   con espacios innecesarios"}'
```

### Clasificar un archivo

```powershell
curl -X POST http://127.0.0.1:8000/classify `
  -F "file=@data/incoming/documento_prueba.txt"
```

Solo se aceptan archivos `.txt`, con un tamaño máximo de 2 MB. El archivo no se almacena en el corpus.

### Buscar documentos

```powershell
curl -X POST http://127.0.0.1:8000/search `
  -H "Content-Type: application/json" `
  -d '{"query":"problemas de latencia del servidor","top_k":5}'
```

`top_k` debe ser un entero entre 1 y 50. La respuesta incluye el identificador, archivo, departamento, título y puntuación de similitud de cada resultado.

## Ejecutar las pruebas

Ejecutar toda la suite:

```powershell
docker compose run --rm api pytest
```

Ejecutar un archivo específico:

```powershell
docker compose run --rm api pytest tests/test_search_api.py
```

Las pruebas cubren extracción, preprocesamiento, preparación del corpus, entrenamiento, clasificación, API y búsqueda.

## Scripts de inspección

Algunos scripts útiles son:

```powershell
docker compose run --rm api python -m scripts.inspect_corpus
docker compose run --rm api python -m scripts.inspect_tfidf
docker compose run --rm api python -m scripts.inspect_model_terms
docker compose run --rm api python -m scripts.inspect_classification_errors
```

Estos comandos permiten revisar la calidad del corpus, los términos TF-IDF, las palabras influyentes de los modelos y los documentos clasificados incorrectamente.

## Notas importantes

- El corpus original se encuentra en `data/corpus/` y se monta como solo lectura dentro del contenedor.
- Los datasets, modelos y resultados se guardan en carpetas del proyecto mediante volúmenes de Docker.
- Si Docker informa errores de permisos, iniciar Docker Desktop y verificar que el usuario tenga acceso al daemon.
- Si la clasificación indica que no existe el modelo, ejecutar primero `prepare_corpus` y `train_svm_v1`.
