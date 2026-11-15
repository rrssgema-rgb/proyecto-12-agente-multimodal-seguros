# Proyecto 12 – Agente multimodal de seguros

**Máster AI Engineer – ThePower**

## 1. Descripción del proyecto

Desarrollo de un agente conversacional multimodal aplicado al sector asegurador, capaz de procesar información en distintos formatos e interactuar con el usuario mediante herramientas especializadas.

El proyecto integra modelos de inteligencia artificial locales y herramientas de procesamiento de imágenes, audio e investigación web.

El agente permite:

- Analizar imágenes de vehículos y documentación de seguros.
- Extraer información de pólizas y generar resultados estructurados.
- Transcribir grabaciones de audio.
- Generar respuestas mediante voz sintética.
- Realizar cálculos.
- Investigar cuestiones relacionadas con seguros mediante Deep Research.
- Mantener el contexto de la conversación mediante memoria.

## 2. Tecnologías utilizadas

| Tecnología | Aplicación |
|---|---|
| Python | Desarrollo del proyecto |
| Ollama | Ejecución local de modelos |
| Qwen2.5-VL 3B | Procesamiento y análisis de imágenes |
| Llama 3.1 8B | Modelo conversacional e investigación |
| LangGraph | Construcción del agente y sus flujos |
| MemorySaver | Memoria conversacional |
| OpenAI Whisper | Transcripción de audio (STT) |
| Piper TTS | Generación de voz |
| DuckDuckGo | Búsqueda de información web |
| Pillow | Procesamiento de imágenes |

## 3. Estructura del proyecto

```text
multimodal-agent/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── 05_multimodal_research.py
├── 06_analizar_siniestro.py
├── 07_extraer_poliza.py
├── 08_respuesta_voz_siniestro.py
├── 09_generar_salida_ejercicio1.py
│
├── vision.py
├── audio.py
├── research.py
│
├── ejemplos/
│   ├── Imágenes y documentos anonimizados del caso
│   ├── Grabaciones y transcripciones
│   └── Capturas de las pruebas realizadas
│
└── resultados/
    ├── ejercicio1_seguros.json
    └── poliza_extraida.json
```

## 4. Ejercicio 1. Procesamiento de imágenes

Se desarrolla un sistema de análisis de imágenes mediante un modelo de visión local.

Las funcionalidades implementadas incluyen:

- Interpretación de imágenes.
- Análisis visual de un vehículo siniestrado.
- Extracción de información de documentación aseguradora.
- Generación de resultados estructurados en JSON.
- Validación de los campos obligatorios.

Los documentos utilizados corresponden a un caso práctico de seguros, con imágenes anonimizadas.

**Resultados principales:**
- `resultados/ejercicio1_seguros.json`
- `resultados/poliza_extraida.json`

## 5. Ejercicio 2. Chatbot multimodal

Se desarrolla un agente conversacional utilizando LangGraph, que integra distintas herramientas y mantiene el contexto de la conversación.

### 5.1. Visión

La herramienta de visión permite analizar imágenes y responder preguntas sobre su contenido.

Se utiliza Qwen2.5-VL 3B mediante Ollama.

### 5.2. Audio: transcripción (STT)

Se integra Whisper para convertir grabaciones de voz en texto.

El sistema permite utilizar el contenido transcrito como información de entrada para el agente.

### 5.3. Audio: generación de voz (TTS)

Se incorpora Piper para generar respuestas habladas a partir del texto producido por el agente.

### 5.4. Herramienta de cálculo

El agente dispone de una herramienta para realizar operaciones matemáticas cuando la consulta lo requiere.

### 5.5. Deep Research

Se integra un grafo de investigación que permite:

1. Planificar una investigación y dividir la pregunta en subpreguntas.
2. Buscar información en la web.
3. Evaluar si la información obtenida es suficiente.
4. Realizar nuevas búsquedas cuando sea necesario.
5. Elaborar un informe con referencias a las fuentes consultadas.

La síntesis incluye citas numeradas y un listado de fuentes.

### 5.6. Memoria conversacional

Se utiliza `MemorySaver` de LangGraph junto con un identificador de conversación (`thread_id`) para conservar el contexto entre interacciones de una misma sesión.

## 6. Instalación y configuración

### Requisitos previos

- Python.
- Ollama.
- Los modelos locales necesarios:
  - `llama3.1:8b`
  - `qwen2.5vl:3b`

### Instalación de dependencias

Crear y activar un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Instalar las dependencias:

```powershell
pip install -r requirements.txt
```

Descargar los modelos de Ollama:

```powershell
ollama pull llama3.1:8b
ollama pull qwen2.5vl:3b
```

La generación de voz con Piper requiere disponer del modelo de voz correspondiente en la carpeta `voces/`.

Whisper descarga el modelo de transcripción necesario durante su primera utilización.

## 7. Archivos principales

| Archivo | Funcionalidad |
|---|---|
| `05_multimodal_research.py` | Agente multimodal principal |
| `06_analizar_siniestro.py` | Análisis visual de siniestros |
| `07_extraer_poliza.py` | Extracción de información de pólizas |
| `08_respuesta_voz_siniestro.py` | Generación de respuestas habladas |
| `09_generar_salida_ejercicio1.py` | Generación y validación de resultados del ejercicio 1 |
| `vision.py` | Funciones de procesamiento visual |
| `audio.py` | Funciones de transcripción y generación de voz |
| `research.py` | Grafo de investigación Deep Research |

## 8. Evidencias y resultados

La carpeta `ejemplos/` contiene las imágenes, documentos y grabaciones utilizados en las pruebas, así como las capturas de pantalla que documentan la implementación y validación de las funcionalidades.

La carpeta `resultados/` contiene los archivos generados durante el procesamiento de la documentación aseguradora.

## 9. Procedencia del código

Los módulos `vision.py`, `audio.py` y `research.py` parten del material didáctico proporcionado durante la formación y se han reutilizado y adaptado para integrarlos en el caso práctico de seguros.

El desarrollo del agente principal y los scripts específicos del caso práctico incorporan la integración de las herramientas y funcionalidades solicitadas en el ejercicio.

## 10. Conclusión

El proyecto integra visión artificial, procesamiento de audio, generación de voz, memoria conversacional, herramientas y búsqueda de información mediante un agente construido con LangGraph.

La aplicación de estas tecnologías a un caso práctico del sector asegurador permite explorar cómo un agente multimodal puede ayudar a interpretar documentación, analizar siniestros y responder consultas de forma conversacional.