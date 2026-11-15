
"""Deep Research con LangGraph:
planificar -> buscar -> evaluar -> sintetizar con citas.
"""

import json
import re
import time
from datetime import date
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


# ----------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    num_predict=500,
    keep_alive="10m"
)

juez = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    format="json",
    num_predict=256,
    keep_alive="10m"
)

MAX_VUELTAS = 2


class Estado(TypedDict, total=False):
    pregunta: str
    pendientes: list
    hechas: list
    fuentes: list
    vueltas: int
    informe: str
    rastro: list
    llamadas_llm: int
    suficiente: bool


def hoy() -> str:
    return date.today().strftime("%d/%m/%Y")


# ----------------------------------------------------------------
# 1. PLANIFICAR
# ----------------------------------------------------------------

PLANIFICAR = """Eres un investigador especializado en seguros de automóvil
y normativa española.

Tu tarea es descomponer la pregunta en entre 2 y 4 búsquedas concretas
que permitan responderla correctamente.

INSTRUCCIONES:

1. Identifica exactamente qué quiere saber el usuario.
2. No mezcles cuestiones diferentes, como:
   - Comunicación de un siniestro.
   - Reclamación de daños materiales.
   - Reclamación de lesiones personales.
   - Coberturas y exclusiones de una póliza.
3. Si la pregunta es jurídica o normativa, prioriza búsquedas
   en fuentes oficiales, especialmente BOE y organismos públicos.
4. Incluye España y el año actual cuando la información pueda cambiar.
5. Formula búsquedas específicas que permitan localizar artículos
   legales, fechas, plazos y condiciones concretas.
6. No propongas búsquedas genéricas ni cuestiones que el usuario
   no haya preguntado.

Responde EXCLUSIVAMENTE con JSON válido:
{{"subpreguntas": ["búsqueda 1", "búsqueda 2", "búsqueda 3"]}}

Hoy es {hoy}.

PREGUNTA:
{pregunta}
"""


def planificar(e: Estado) -> Estado:
    t0 = time.time()

    resultado = juez.invoke(
        PLANIFICAR.format(
            pregunta=e["pregunta"],
            hoy=hoy()
        )
    ).content

    datos = json.loads(resultado)

    subpreguntas = [
        s for s in datos.get("subpreguntas", [])
        if isinstance(s, str) and s.strip()
    ][:4]

    if not subpreguntas:
        subpreguntas = [e["pregunta"]]

    segundos = round(time.time() - t0, 2)

    return {
        "pendientes": subpreguntas,
        "hechas": [],
        "fuentes": [],
        "vueltas": 0,
        "llamadas_llm": 1,
        "rastro": [
            f"PLAN · {len(subpreguntas)} subpreguntas · {segundos} s"
        ] + [
            f"   - {s}" for s in subpreguntas
        ]
    }


# ----------------------------------------------------------------
# 2. BUSCAR EN LA WEB
# ----------------------------------------------------------------



def buscar_web(consulta: str, n: int = 3) -> list:
    """Busca fuentes oficiales y recupera el contenido de las paginas."""

    import requests
    import re
    from bs4 import BeautifulSoup
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException

    resultados = []
    vistas = set()

    # Descargar y extraer el contenido de una pagina.
    def extraer_contenido(url: str, fragmento: str = "") -> str:
        try:
            respuesta = requests.get(
                url,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"
                }
            )

            respuesta.raise_for_status()

            tipo = respuesta.headers.get("Content-Type", "").lower()

            if "html" not in tipo:
                return fragmento

            soup = BeautifulSoup(respuesta.content, "html.parser")

            for elemento in soup([
                "script", "style", "nav", "footer",
                "header", "form", "noscript"
            ]):
                elemento.decompose()

            contenido = (
                soup.find("main")
                or soup.find("article")
                or soup.body
                or soup
            )

            texto = contenido.get_text(
                separator="\n",
                strip=True
            )

            

            if texto:
                coincidencia = re.search(
                    r"artículo\s+(\d+)",
                    consulta,
                    re.IGNORECASE
                )

                if coincidencia:
                    numero = coincidencia.group(1)

                    if numero == "16":
                        encabezado = r"(?:16|dieciséis)"
                    else:
                        encabezado = re.escape(numero)

                    patron = re.compile(
                        rf"Artículo\s+{encabezado}\s*\.?",
                        re.IGNORECASE
                    )

                    inicio = patron.search(texto)

                    if inicio:
                        siguiente = re.search(
                            r"\nArtículo\s+(?:\d+|[a-záéíóú]+)\s*\.?",
                            texto[inicio.end():],
                            re.IGNORECASE
                        )

                        if siguiente:
                            fin = inicio.end() + siguiente.start()
                            articulo = texto[inicio.start():fin]
                        else:
                            articulo = texto[inicio.start():]

                        return articulo[:5000]

                return texto[:15000]

            return fragmento

        except requests.RequestException as error:
            print(f"No se pudo descargar {url}: {error}")
            return fragmento


    # Buscar primero en fuentes oficiales.
    consulta_oficial = (
        f"{consulta} "
        "(site:boe.es OR site:dgsfp.mineco.gob.es "
        "OR site:consorseguros.es)"
    )

    consultas = [
        (consulta_oficial, 3),
        (consulta, n)
    ]

    for texto_busqueda, max_resultados in consultas:
        try:
            encontrados = DDGS().text(
                texto_busqueda,
                region="es-es",
                max_results=max_resultados
            )

            for r in encontrados:
                url = r.get("href", "")

                if not url or url in vistas:
                    continue

                vistas.add(url)

                fragmento = r.get("body", "")
                contenido = extraer_contenido(url, fragmento)

                resultados.append({
                    "title": r.get("title", "Sin titulo"),
                    "href": url,
                    "body": contenido
                })

        except DDGSException as error:
            print(f"Error en DuckDuckGo: {error}")

    return resultados

    # Descargar y extraer el contenido de una página web.
    
# Descargar y extraer el contenido de una página web.
def extraer_contenido(url: str, fragmento: str = "") -> str:
    try:
        respuesta = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"
            }
        )

        respuesta.raise_for_status()

        # Solo procesamos páginas HTML.
        tipo = respuesta.headers.get("Content-Type", "").lower()

        if "html" not in tipo:
            return fragmento

        soup = BeautifulSoup(respuesta.content, "html.parser")

        # Eliminar elementos que no aportan contenido informativo.
        for elemento in soup([
            "script", "style", "nav", "footer",
            "header", "form", "noscript"
        ]):
            elemento.decompose()

        # Priorizar el contenido principal de la página.
        contenido = (
            soup.find("main")
            or soup.find("article")
            or soup.body
            or soup
        )

        texto = contenido.get_text(
            separator="\n",
            strip=True
        )

        if texto:
            # Buscar si la consulta solicita un artículo concreto.
            import re

            #coincidencia = re.search(
            #    r"artículo\s+(\d+)",
            #    consulta,
            #    re.IGNORECASE
            #)
            coincidencia = re.search(
                r"art[i\u00ed]culo\s+(\d+|diecis[e\u00e9]is)",
                consulta,
                re.IGNORECASE
)

            if coincidencia:
                numero_articulo = coincidencia.group(1)

                # Localizar el encabezado del artículo.
                patron = re.compile(
                    rf"Artículo\s+{numero_articulo}\s*\.?",
                    re.IGNORECASE
                )

                inicio = patron.search(texto)

                if inicio:
                    # Extraer el artículo hasta el siguiente encabezado.
                    siguiente = re.search(
                        r"\nArtículo\s+(?:\d+|[a-záéíóú]+)\s*\.?",
                        texto[inicio.end():],
                        re.IGNORECASE
                    )

                    if siguiente:
                        fin = inicio.end() + siguiente.start()
                        articulo = texto[inicio.start():fin]
                    else:
                        articulo = texto[inicio.start():]

                    return articulo[:5000]

            # Si no se encuentra un artículo concreto,
            # conservar la extracción general.
            return texto[:15000]

        return fragmento

    except requests.RequestException as error:
        print(f"  No se pudo descargar {url}: {error}")
        return fragmento




def buscar(e: Estado) -> Estado:
    t0 = time.time()

    fuentes = list(e["fuentes"])
    vistas = {f["url"] for f in fuentes}
    rastro = []

    for sub in e["pendientes"]:
        inicio = time.time()
        nuevos = 0

        resultados = buscar_web(sub)

        for r in resultados:
            url = r.get("href", "")

            if not url or url in vistas:
                continue

            vistas.add(url)

            fuentes.append({
                "id": len(fuentes) + 1,
                "url": url,
                "titulo": r.get("title", "Sin título"),
                "texto": r.get("body", ""),
                "subpregunta": sub
            })

            nuevos += 1

        segundos = round(time.time() - inicio, 2)

        rastro.append(
            f"BUSCAR · '{sub}' -> "
            f"{nuevos} fuentes nuevas · {segundos} s"
        )

        # Pausa para reducir el riesgo de restricciones de DuckDuckGo.
        time.sleep(1)

    total = round(time.time() - t0, 2)

    rastro.append(f"BUSCAR · tiempo total: {total} s")

    return {
        "fuentes": fuentes,
        "hechas": e["hechas"] + e["pendientes"],
        "pendientes": [],
        "vueltas": e["vueltas"] + 1,
        "rastro": e["rastro"] + rastro
    }


# ----------------------------------------------------------------
# 3. EVALUAR: ¿FALTA INFORMACIÓN?
# ----------------------------------------------------------------

EVALUAR = """Eres un investigador especializado en normativa y seguros
de automóvil en España.

Tu tarea es comprobar rigurosamente si las fuentes permiten responder
a la pregunta planteada.

Analiza cada parte de la pregunta y aplica estas reglas:

1. COMPROBACIÓN DE INFORMACIÓN:
- Comprueba si las fuentes aportan datos concretos y suficientes.
- Identifica artículos legales, fechas, cifras y condiciones.
- Comprueba que la información responde exactamente a la pregunta.
- No confundas plazos de comunicación del siniestro con plazos
  de reclamación de indemnizaciones.

2. FIABILIDAD DE LAS FUENTES:
- Da prioridad a fuentes oficiales: BOE, DGSFP y organismos públicos.
- Comprueba si las fuentes oficiales aportan información suficiente.
- Los blogs, páginas comerciales y despachos privados son fuentes
  secundarias y no deben prevalecer sobre la normativa oficial.
- No consideres confirmada una afirmación únicamente porque
  aparezca repetida en varias páginas secundarias.

3. VIGENCIA:
- Comprueba si la información encontrada corresponde a normativa
  vigente en España.
- Si no se puede comprobar su vigencia, indica que falta verificarla.

4. DECISIÓN:
- Marca "suficiente": true únicamente cuando las fuentes permitan
  responder con información concreta, coherente y suficientemente
  fiable.
- Marca "suficiente": false cuando falten datos, existan contradicciones
  relevantes o no haya fuentes adecuadas para verificar la respuesta.
- Si falta información, propone como máximo 2 búsquedas nuevas.
- Las búsquedas nuevas deben ser concretas y diferentes de las anteriores.
- No inventes datos ni des por confirmada información no verificada.

Responde EXCLUSIVAMENTE con JSON válido:
{{"suficiente": true/false,
  "falta": "información que falta o motivo por el que no es suficiente",
  "nuevas": ["búsqueda nueva 1", "búsqueda nueva 2"]}}

PREGUNTA:
{pregunta}

BUSQUEDAS YA HECHAS:
{hechas}

FUENTES:
{fuentes}
"""


# ----------------------------------------------------------------
# FORMATEAR FUENTES
# ----------------------------------------------------------------

def formatear(fuentes: list, max_chars: int = 800) -> str:
    return "\n\n".join(
        f"[{f['id']}] {f['titulo']}\n"
        f"URL: {f['url']}\n"
        f"Contenido: {f['texto'][:max_chars]}"
        for f in fuentes
    )


# ----------------------------------------------------------------
# EVALUAR
# ----------------------------------------------------------------

def evaluar(e: Estado) -> Estado:
    t0 = time.time()

    resultado = juez.invoke(
        EVALUAR.format(
            pregunta=e["pregunta"],
            hechas="; ".join(e["hechas"]),
            fuentes=formatear(e["fuentes"])
        )
    ).content

    datos = json.loads(resultado)

    nuevas = [
        s for s in datos.get("nuevas", [])
        if isinstance(s, str)
        and s.strip()
        and s not in e["hechas"]
    ][:2]

    # Solo aceptamos el booleano JSON true.
    suficiente = datos.get("suficiente") is True

    segundos = round(time.time() - t0, 2)

    if suficiente:
        linea = "EVALUAR · suficiente"
    else:
        linea = (
            f"EVALUAR · falta: {datos.get('falta', '')} "
            f"-> {nuevas}"
        )

    linea += f" · {segundos} s"

    return {
        "pendientes": [] if suficiente else nuevas,
        "suficiente": suficiente,
        "llamadas_llm": e["llamadas_llm"] + 1,
        "rastro": e["rastro"] + [linea]
    }


# ----------------------------------------------------------------
# DECIDIR: ¿OTRA BÚSQUEDA O SINTETIZAR?
# ----------------------------------------------------------------

def seguir_o_sintetizar(e: Estado) -> str:
    """Continúa buscando si falta información y quedan vueltas."""

    if e["pendientes"] and e["vueltas"] < MAX_VUELTAS:
        return "buscar"

    return "sintetizar"


# ----------------------------------------------------------------
# 4. SINTETIZAR CON CITAS
# ----------------------------------------------------------------

SINTETIZAR = """Eres un investigador especializado en seguros de automóvil
y normativa española.

Redacta un informe breve en español que responda directamente
a la PREGUNTA utilizando exclusivamente la información proporcionada
por las FUENTES.

REGLAS OBLIGATORIAS:

1. FIDELIDAD A LAS FUENTES
- No utilices conocimientos propios.
- No inventes plazos, cifras, artículos legales ni condiciones.
- No afirmes nada que no esté respaldado por las fuentes.
- Si no puedes verificar un dato, indícalo expresamente.

2. PRIORIDAD DE LAS FUENTES
- Da prioridad a la legislación y a los organismos oficiales.
- Identifica claramente cuándo una afirmación procede de una fuente
  secundaria.
- No presentes información de blogs o páginas comerciales como
  si fuera una obligación legal.
- Si una fuente secundaria contradice una fuente oficial,
  prevalece la información oficial, siempre que sea aplicable
  y esté vigente.

3. PRECISIÓN
- Responde exactamente a lo que pregunta el usuario.
- No mezcles plazos de comunicación del siniestro con plazos
  de reclamación de indemnizaciones.
- Diferencia las obligaciones legales de las recomendaciones prácticas.
- No atribuyas consecuencias jurídicas que las fuentes no acrediten.

4. CITAS
- Cada afirmación importante debe incluir su referencia entre corchetes:
  [1], [2], etc.
- Utiliza únicamente identificadores de fuentes existentes.
- No inventes referencias ni URLs.
- Si existen contradicciones relevantes, explícalas y cita las fuentes.
- Antes de redactar, identifica qué fuentes respaldan cada afirmación.
- Incluye obligatoriamente al menos una cita [n] en cada punto
  que contenga información procedente de las fuentes.
- Utiliza los identificadores numéricos que aparecen al principio
  de cada fuente.
- No redactes el apartado FUENTES; lo genera automáticamente
  el programa.
- No incluyas afirmaciones sin respaldo documental.
- Si la información está respaldada por una fuente oficial,
  utiliza preferentemente esa referencia.

5. INFORMACIÓN NO CONFIRMADA
- Si no hay información oficial suficiente, indícalo.
- No conviertas una respuesta incompleta en una conclusión definitiva.

FORMATO DE RESPUESTA:

Un título descriptivo.

Entre 3 y 5 puntos con la información relevante.

Una sección final titulada "Lo que no he podido confirmar",
solo cuando existan datos relevantes que no hayan podido verificarse.

Termina con una conclusión breve y prudente.

PREGUNTA:
{pregunta}

FUENTES:
{fuentes}
"""


def sintetizar(e: Estado) -> Estado:
    t0 = time.time()
    if not e.get("suficiente", False):
        return {
            "informe": (
                "No puedo ofrecer una respuesta verificada porque "
                "la investigación no ha encontrado información "
                "suficiente en fuentes fiables. Es necesario "
                "comprobar la normativa o consultar una fuente oficial."
            ),
            "rastro": e["rastro"] + [
                "SINTETIZAR · bloqueada por falta de información verificada"
            ]
        }

    if not e["fuentes"]:
        return {
            "informe": "No he encontrado fuentes en la web para esta pregunta.",
            "rastro": e["rastro"] + [
                "SINTETIZAR · sin fuentes: no se llama al LLM"
            ]
        }

    texto = llm.invoke(
        SINTETIZAR.format(
            pregunta=e["pregunta"],
            fuentes=formatear(e["fuentes"])
        )
    ).content.strip()

    segundos = round(time.time() - t0, 2)

    # Las citas se extraen del texto generado.
    citadas = sorted({
        int(n) for n in re.findall(r"\[(\d+)\]", texto)
    })

    validas = {f["id"]: f for f in e["fuentes"]}
    fantasma = [n for n in citadas if n not in validas]

    lista = "\n".join(
        f"[{n}] {validas[n]['titulo']} · {validas[n]['url']}"
        for n in citadas
        if n in validas
    )

    informe = f"{texto}\n\nFUENTES\n{lista}"

    rastro = [
        f"SINTETIZAR · cita "
        f"{len(citadas) - len(fantasma)} "
        f"de {len(e['fuentes'])} fuentes · {segundos} s"
    ]

    if fantasma:
        rastro.append(
            f"¡Citas a fuentes inexistentes: {fantasma}"
        )

    return {
        "informe": informe,
        "llamadas_llm": e["llamadas_llm"] + 1,
        "rastro": e["rastro"] + rastro
    }


# ----------------------------------------------------------------
# CONSTRUIR EL GRAFO LANGGRAPH
# ----------------------------------------------------------------

def construir():
    g = StateGraph(Estado)

    g.add_node("planificar", planificar)
    g.add_node("buscar", buscar)
    g.add_node("evaluar", evaluar)
    g.add_node("sintetizar", sintetizar)

    g.add_edge(START, "planificar")
    g.add_edge("planificar", "buscar")
    g.add_edge("buscar", "evaluar")

    g.add_conditional_edges(
        "evaluar",
        seguir_o_sintetizar,
        {
            "buscar": "buscar",
            "sintetizar": "sintetizar"
        }
    )

    g.add_edge("sintetizar", END)

    return g.compile()


# ----------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ----------------------------------------------------------------

def investigar(pregunta: str) -> dict:
    t0 = time.time()

    resultado = construir().invoke({
        "pregunta": pregunta
    })

    return {
        **resultado,
        "segundos": round(time.time() - t0, 2)
    }
