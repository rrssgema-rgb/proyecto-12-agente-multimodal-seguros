
"""Paso 5 · Agente multimodal de seguros de automóvil.

El agente selecciona la herramienta adecuada:
- CALCULAR: operaciones matemáticas sencillas.
- VER: análisis de imágenes con Qwen2.5-VL.
- OÍR: transcripción de archivos de audio con Whisper.
- INVESTIGAR: reutiliza el grafo Deep Research del paso 4.
- RESPONDER: consultas que no necesitan herramientas.

Incluye memoria conversacional mediante LangGraph MemorySaver
y generación de voz con Piper cuando la entrada procede de audio.
"""

import ast
import json
import operator
import sys
import time
from pathlib import Path
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from vision import ver
from audio import transcribir, hablar
from research import construir as construir_research


# ----------------------------------------------------------------
# 1. MODELOS Y GRAFO DE INVESTIGACIÓN
# ----------------------------------------------------------------

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

juez = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    format="json"
)

investigador = construir_research()

CARPETA_SALIDAS = Path("salidas")
CARPETA_SALIDAS.mkdir(exist_ok=True)


# ----------------------------------------------------------------
# 2. ESTADO DEL AGENTE
# ----------------------------------------------------------------

class Estado(TypedDict, total=False):
    texto: str
    imagen: str
    audio: str
    audio_texto: str
    entrada_audio: bool

    percibido: str
    ruta: str
    motivo: str
    expresion: str
    pregunta_investigar: str

    resultado_herramienta: str
    respuesta: str
    voz: str

    rastro: list
    historial: list


# ----------------------------------------------------------------
# 3. PERCIBIR: RECOGER LAS ENTRADAS SIN EJECUTAR HERRAMIENTAS
# ----------------------------------------------------------------

def percibir(e: Estado) -> Estado:
    partes = []
    rastro = []

    texto = e.get("texto", "").strip()
    imagen = e.get("imagen", "").strip()
    audio = e.get("audio", "").strip()
    audio_texto = e.get("audio_texto", "").strip()

    if texto:
        partes.append(
            "CONSULTA DEL ASEGURADO:\n" + texto
        )

    if imagen:
        partes.append(
            "El usuario ha adjuntado una imagen.\n"
            "Ruta de la imagen: " + imagen
        )
        rastro.append("ENTRADA · Imagen adjunta")

    if audio:
        partes.append(
            "El usuario ha adjuntado un archivo de audio.\n"
            "Ruta del audio: " + audio
        )
        rastro.append("ENTRADA · Archivo de audio adjunto")

    if audio_texto:
        partes.append(
            "TRANSCRIPCIÓN DE AUDIO YA DISPONIBLE:\n"
            + audio_texto
        )
        rastro.append("ENTRADA · Transcripción de audio disponible")

    if not partes:
        partes.append(
            "El usuario no ha proporcionado información."
        )

    percibido = "\n\n".join(partes)

    print("\n--- INFORMACIÓN RECIBIDA ---")
    print(percibido)

    return {
        "percibido": percibido,
        "rastro": rastro
    }


# ----------------------------------------------------------------
# 4. DECIDIR: EL LLM SELECCIONA LA HERRAMIENTA
# ----------------------------------------------------------------

DECIDIR = """Eres el coordinador de herramientas de un agente especializado
en seguros de automóvil en España.

Debes seleccionar UNA herramienta para atender la consulta.

HERRAMIENTAS DISPONIBLES:

1. "calcular":
   Para operaciones matemáticas, porcentajes, importes, sumas,
   diferencias o cálculos numéricos.

2. "ver":
   Para analizar una imagen adjunta: fotografías de vehículos,
   daños, documentos, pólizas, facturas o recibos.

3. "oir":
   Para escuchar y transcribir un archivo de audio o utilizar
   una transcripción de audio ya disponible.

4. "investigar":
   Para consultas que necesitan información externa, actualizada,
   normativa, legislación o investigación documental.

5. "responder":
   Para consultas que se pueden resolver con la información recibida
   y el historial, sin utilizar otra herramienta.



REGLAS:
- Selecciona la herramienta que mejor responda a la intención principal.
- Distingue claramente entre mencionar un archivo y adjuntarlo realmente.
- Solo considera que hay una imagen adjunta cuando la información
  recibida contiene una ruta de imagen en el campo de imagen.
- Si el usuario dice que tiene fotografías, pero no las adjunta,
  NO selecciones "ver".
- Si hay una imagen realmente adjunta y pregunta por su contenido,
  selecciona "ver".
- Si el usuario adjunta un audio o aporta una transcripción
  y quiere que se analice lo que se ha dicho, selecciona "oir".
- Si solicita un cálculo, selecciona "calcular".
- Si pregunta por normativa, plazos legales o información externa
  que deba comprobarse, selecciona "investigar".
- Si aporta una transcripción y pide analizarla, identificar
  necesidades o explicar documentación, selecciona "responder",
  salvo que solicite expresamente investigar información externa.
- No selecciones "investigar" para cálculos sencillos.
- No selecciones "ver" si no hay una imagen adjunta.
- No selecciones "oir" si no hay audio ni transcripción.
- No inventes datos ni presupongas el contenido de archivos.
- En "calcular", devuelve una expresión matemática sencilla
  utilizando únicamente números y operadores aritméticos.
- En "investigar", formula una pregunta concreta para investigar.
- En las demás herramientas, deja "expresion" y "pregunta"
  como cadenas vacías cuando no sean necesarias.
  
   

Devuelve exclusivamente un JSON con esta estructura:
{{
  "herramienta": "calcular, ver, oir, investigar o responder",
  "expresion": "expresión matemática o cadena vacía",
  "pregunta": "pregunta concreta para investigar o cadena vacía",
  "motivo": "explicación breve de la selección"
}}

INFORMACIÓN RECIBIDA:
{percibido}

HISTORIAL:
{historial}
"""


def extraer_json(texto: str) -> dict:
    """Extrae un objeto JSON de la respuesta del modelo."""
    texto = texto.strip()

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        inicio = texto.find("{")
        final = texto.rfind("}")

        if inicio == -1 or final == -1:
            raise ValueError(
                "El modelo no ha devuelto un JSON válido."
            )

        return json.loads(texto[inicio:final + 1])


def decidir(e: Estado) -> Estado:
    historial = e.get("historial", [])

    resultado = juez.invoke(
        DECIDIR.format(
            percibido=e.get("percibido", ""),
            historial=json.dumps(
                historial[-6:],
                ensure_ascii=False
            )
        )
    ).content

    datos = extraer_json(resultado)

    herramientas_validas = {
        "calcular",
        "ver",
        "oir",
        "investigar",
        "responder"
    }

    herramienta = str(
        datos.get("herramienta", "responder")
    ).lower().strip()

    if herramienta not in herramientas_validas:
        herramienta = "responder"

    # Comprobaciones para evitar seleccionar herramientas
    # cuando no existe el archivo o la información necesaria.
    if herramienta == "ver" and not e.get("imagen"):
        herramienta = "responder"

    if herramienta == "oir":
        if not e.get("audio") and not e.get("audio_texto"):
            herramienta = "responder"

    expresion = str(datos.get("expresion", "")).strip()
    pregunta = str(datos.get("pregunta", "")).strip()
    motivo = str(datos.get("motivo", "")).strip()

    print("\n--- HERRAMIENTA SELECCIONADA ---")
    print(herramienta.upper())
    print("Motivo:", motivo)

    return {
        "ruta": herramienta,
        "expresion": expresion,
        "pregunta_investigar": pregunta,
        "motivo": motivo,
        "rastro": e.get("rastro", []) + [
            f"DECIDIR · {herramienta.upper()}: {motivo}"
        ]
    }


# ----------------------------------------------------------------
# 5. HERRAMIENTA CALCULAR
# ----------------------------------------------------------------

OPERADORES = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}


def evaluar_expresion(expresion: str):
    """Evalúa expresiones aritméticas sin utilizar eval()."""

    arbol = ast.parse(expresion, mode="eval")

    def evaluar(nodo):
        if isinstance(nodo, ast.Expression):
            return evaluar(nodo.body)

        if isinstance(nodo, ast.Constant):
            if isinstance(nodo.value, (int, float)):
                return nodo.value
            raise ValueError("Solo se permiten números.")

        if isinstance(nodo, ast.BinOp):
            tipo = type(nodo.op)

            if tipo not in OPERADORES:
                raise ValueError("Operador no permitido.")

            izquierda = evaluar(nodo.left)
            derecha = evaluar(nodo.right)

            if tipo is ast.Pow and abs(derecha) > 10:
                raise ValueError("Exponente demasiado grande.")

            resultado = OPERADORES[tipo](
                izquierda,
                derecha
            )

            if abs(resultado) > 1e15:
                raise ValueError("Resultado fuera del límite permitido.")

            return resultado

        if isinstance(nodo, ast.UnaryOp):
            tipo = type(nodo.op)

            if tipo not in OPERADORES:
                raise ValueError("Operador no permitido.")

            return OPERADORES[tipo](evaluar(nodo.operand))

        raise ValueError("La expresión contiene elementos no permitidos.")

    return evaluar(arbol)



def calcular(e: Estado) -> Estado:
    expresion = e.get("expresion", "").strip()

    print(f"Expresión recibida del modelo: [{expresion}]")

    if not expresion:
        raise ValueError(
            "El modelo no ha proporcionado una expresión matemática."
        )

    # Convertir porcentajes escritos como 21% en 0.21.
    import re

    expresion = re.sub(
        r"(\d+(?:\.\d+)?)\s*%",
        r"(\1 / 100)",
        expresion
    )

    print(f"Expresión adaptada: [{expresion}]")

    resultado = evaluar_expresion(expresion)

    if isinstance(resultado, float):
        resultado_mostrado = round(resultado, 4)
    else:
        resultado_mostrado = resultado

    informacion = (
        f"Expresión calculada: {expresion}\n"
        f"Resultado: {resultado_mostrado}"
    )

    print("\n--- RESULTADO DEL CÁLCULO ---")
    print(informacion)

    return {
        "resultado_herramienta": informacion,
        "percibido": (
            e.get("percibido", "")
            + "\n\nRESULTADO DE LA HERRAMIENTA CALCULAR:\n"
            + informacion
        ),
        "rastro": e.get("rastro", []) + [
            f"CALCULAR · {expresion} = {resultado_mostrado}"
        ]
    }



# ----------------------------------------------------------------
# 6. HERRAMIENTA VER: ANÁLISIS DE IMÁGENES
# ----------------------------------------------------------------

PROMPT_VISION = """Analiza esta imagen relacionada con un seguro de automóvil.

Si es un vehículo:
- Describe los daños visibles y su localización.
- Indica si parecen leves, moderados o graves.
- No determines culpabilidad ni causa del accidente.

Si es una póliza, factura o documento:
- Extrae los datos que se puedan leer.
- Distingue los datos visibles de los que no se pueden confirmar.
- No inventes importes, fechas ni coberturas.

No presentes una valoración visual como una peritación profesional.
"""


def ver_imagen(e: Estado) -> Estado:
    ruta = e.get("imagen", "")

    if not ruta or not Path(ruta).is_file():
        raise FileNotFoundError(
            f"No se encuentra la imagen: {ruta}"
        )

    inicio = time.time()

    resultado = ver(
        ruta,
        PROMPT_VISION
    )

    analisis = resultado["texto"]
    segundos = resultado.get(
        "segundos",
        round(time.time() - inicio, 1)
    )

    informacion = (
        "ANÁLISIS DE LA IMAGEN:\n" + analisis
    )

    print("\n--- RESULTADO DE VER ---")
    print(informacion)

    return {
        "resultado_herramienta": informacion,
        "percibido": (
            e.get("percibido", "")
            + "\n\n"
            + informacion
        ),
        "rastro": e.get("rastro", []) + [
            f"VER · Qwen2.5-VL ({segundos} s)"
        ]
    }


# ----------------------------------------------------------------
# 7. HERRAMIENTA OÍR: TRANSCRIPCIÓN CON WHISPER
# ----------------------------------------------------------------

def oir(e: Estado) -> Estado:
    audio_texto = e.get("audio_texto", "").strip()

    if audio_texto:
        texto = audio_texto
        segundos = 0
        origen = "Transcripción recibida"
    else:
        ruta = e.get("audio", "")

        if not ruta or not Path(ruta).is_file():
            raise FileNotFoundError(
                f"No se encuentra el archivo de audio: {ruta}"
            )

        resultado = transcribir(ruta)
        texto = resultado["texto"]
        segundos = resultado["segundos"]
        origen = "Whisper"

    informacion = (
        "TRANSCRIPCIÓN DEL AUDIO:\n" + texto
    )

    print("\n--- RESULTADO DE OÍR ---")
    print(informacion)

    return {
        "audio_texto": texto,
        "resultado_herramienta": informacion,
        "percibido": (
            e.get("percibido", "")
            + "\n\n"
            + informacion
        ),
        "entrada_audio": True,
        "rastro": e.get("rastro", []) + [
            f"OÍR · {origen} ({segundos} s)"
        ]
    }


# ----------------------------------------------------------------
# 8. RESPONDER UTILIZANDO LA INFORMACIÓN Y LA MEMORIA
# ----------------------------------------------------------------


RESPONDER = """Eres un asistente virtual de atención al cliente
especializado en seguros de automóvil en España.

Tu función es ayudar al asegurado a comprender la información
recibida y orientar sobre la tramitación de su siniestro.

Responde directamente a la pregunta del usuario.
No rechaces analizar imágenes ni describir los resultados
que proporciona la herramienta de visión.

Si existe un análisis de imagen:
- Resume los daños que describe el análisis.
- Indica que son observaciones visuales, no una peritación.
- No inventes daños ni identifiques datos que no se hayan confirmado.

Si se ha transcrito un audio, utiliza su contenido real.

Si se ha realizado un cálculo, conserva exactamente el resultado
numérico proporcionado por la herramienta.

No inventes coberturas, importes, plazos ni responsabilidades.
No determines quién es culpable basándote únicamente en una fotografía.
Si no puedes confirmar una cobertura concreta, indica que debe
comprobarse con la aseguradora.

Responde con naturalidad, sin disculparte innecesariamente.
Máximo 6 frases.

HISTORIAL:
{historial}

INFORMACIÓN ACTUAL:
{percibido}
"""



def responder(e: Estado) -> Estado:
    historial = e.get("historial", [])

    respuesta = llm.invoke(
        RESPONDER.format(
            historial=json.dumps(
                historial[-6:],
                ensure_ascii=False
            ),
            percibido=e.get("percibido", "")
        )
    ).content.strip()

    return {
        "respuesta": respuesta
    }


# ----------------------------------------------------------------
# 9. INVESTIGAR: REUTILIZAR EL GRAFO DEEP RESEARCH DEL PASO 4
# ----------------------------------------------------------------

def investigar(e: Estado) -> Estado:
    pregunta = (
        e.get("pregunta_investigar")
        or e.get("texto")
        or e.get("percibido", "")
    )

    print("\n--- DEEP RESEARCH ---")
    print("Pregunta:", pregunta)

    resultado = investigador.invoke(
        {"pregunta": pregunta}
    )

    informe = resultado.get(
        "informe",
        "La investigación no ha devuelto un informe."
    )

    fuentes = resultado.get("fuentes", [])
    vueltas = resultado.get("vueltas", 0)

    return {
        "respuesta": informe,
        "resultado_herramienta": informe,
        "rastro": e.get("rastro", []) + [
            f"INVESTIGAR · {len(fuentes)} fuentes, "
            f"{vueltas} vueltas"
        ]
    }


# ----------------------------------------------------------------
# 10. HABLAR: GENERAR VOZ CON PIPER SI LA ENTRADA ERA AUDIO
# ----------------------------------------------------------------

RESUMIR_VOZ = """Resume esta respuesta para decirla en voz alta.
Utiliza dos o tres frases cortas, sin listas, URLs ni referencias
entre corchetes. Escribe los números con palabras.

RESPUESTA:
{respuesta}
"""


def hablar_nodo(e: Estado) -> Estado:
    if not e.get("entrada_audio"):
        return {}

    guion = llm.invoke(
        RESUMIR_VOZ.format(
            respuesta=e.get("respuesta", "")
        )
    ).content.strip()

    resultado = hablar(
        guion,
        str(CARPETA_SALIDAS / "respuesta_agente.wav"),
        motor="piper"
    )

    return {
        "voz": resultado["archivo"],
        "rastro": e.get("rastro", []) + [
            f"HABLAR · Piper: {resultado['duracion']} segundos"
        ]
    }


# ----------------------------------------------------------------
# 11. GUARDAR EL HISTORIAL DE CONVERSACIÓN
# ----------------------------------------------------------------

def guardar(e: Estado) -> Estado:
    historial = list(e.get("historial", []))

    historial.append({
        "usuario": e.get("percibido", ""),
        "asistente": e.get("respuesta", "")
    })

    return {
        "historial": historial
    }


# ----------------------------------------------------------------
# 12. CONSTRUIR EL GRAFO CON MEMORIA
# ----------------------------------------------------------------

def seleccionar_siguiente(e: Estado) -> str:
    return e.get("ruta", "responder")


def construir():
    g = StateGraph(Estado)

    g.add_node("percibir", percibir)
    g.add_node("decidir", decidir)

    g.add_node("calcular", calcular)
    g.add_node("ver", ver_imagen)
    g.add_node("oir", oir)
    g.add_node("investigar", investigar)
    g.add_node("responder", responder)

    g.add_node("hablar", hablar_nodo)
    g.add_node("guardar", guardar)

    g.add_edge(START, "percibir")
    g.add_edge("percibir", "decidir")

    g.add_conditional_edges(
        "decidir",
        seleccionar_siguiente,
        {
            "calcular": "calcular",
            "ver": "ver",
            "oir": "oir",
            "investigar": "investigar",
            "responder": "responder"
        }
    )

    # Las herramientas locales aportan información al nodo responder.
    g.add_edge("calcular", "responder")
    g.add_edge("ver", "responder")
    g.add_edge("oir", "responder")

    # Deep Research ya devuelve su propio informe.
    g.add_edge("investigar", "hablar")

    g.add_edge("responder", "hablar")
    g.add_edge("hablar", "guardar")
    g.add_edge("guardar", END)

    memoria = MemorySaver()

    return g.compile(
        checkpointer=memoria
    )


agente = construir()


# ----------------------------------------------------------------
# 13. CONFIGURACIÓN DE MEMORIA
# ----------------------------------------------------------------

THREAD_ID = "conversacion_seguros_01"

CONFIG = {
    "configurable": {
        "thread_id": THREAD_ID
    }
}


# ----------------------------------------------------------------
# 14. INTERFAZ DE PRUEBA POR CONSOLA
# ----------------------------------------------------------------

def mostrar(entrada: dict):
    inicio = time.time()

    resultado = agente.invoke(
        entrada,
        config=CONFIG
    )

    print("\nRASTRO DEL AGENTE")

    for linea in resultado.get("rastro", []):
        print("  " + linea)

    print("\nRESPUESTA:")
    print(resultado.get("respuesta", ""))

    print(
        f"\nTiempo total: {round(time.time() - inicio, 1)} segundos"
    )

    if resultado.get("voz"):
        print("Audio generado:", resultado["voz"])


def iniciar_chat():
    print("\nASISTENTE MULTIMODAL DE SEGUROS")
    print("--------------------------------")
    print("Escribe tu consulta.")
    print("Imagen: imagen:ruta | pregunta")
    print("Audio: audio:ruta | pregunta")
    print("Transcripción ya disponible: transcripcion:texto")
    print("Para terminar: salir\n")

    while True:
        linea = input("Tú: ").strip()

        if linea.lower() in ("salir", "exit", "quit"):
            break

        if not linea:
            continue

        entrada = {
            "texto": "",
            "imagen": "",
            "audio": "",
            "audio_texto": "",
            "entrada_audio": False
        }

        if linea.lower().startswith("imagen:"):
            contenido = linea[len("imagen:"):].strip()
            ruta, separador, pregunta = contenido.partition("|")

            entrada["imagen"] = ruta.strip()
            entrada["texto"] = pregunta.strip()

        elif linea.lower().startswith("audio:"):
            contenido = linea[len("audio:"):].strip()
            ruta, separador, pregunta = contenido.partition("|")

            entrada["audio"] = ruta.strip()
            entrada["texto"] = pregunta.strip()
            entrada["entrada_audio"] = True

        elif linea.lower().startswith("transcripcion:"):
            entrada["audio_texto"] = (
                linea[len("transcripcion:"):].strip()
            )
            entrada["texto"] = (
                "Analiza esta declaración del asegurado."
            )
            entrada["entrada_audio"] = True

        else:
            entrada["texto"] = linea

        try:
            mostrar(entrada)

        except Exception as error:
            print(f"\nERROR: {error}")


# ----------------------------------------------------------------
# 15. INICIO
# ----------------------------------------------------------------

if __name__ == "__main__":
    if "--chat" in sys.argv:
        iniciar_chat()
    else:
        print("Para iniciar el asistente, ejecuta:")
        print("python 05_multimodal_research.py --chat")
