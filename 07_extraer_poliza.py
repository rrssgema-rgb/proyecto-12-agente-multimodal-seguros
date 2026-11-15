"""Extrae información de una póliza de automóvil mediante Qwen2.5-VL."""

import json
from pathlib import Path
from vision import extraer

CARPETA = Path("ejemplos")
SALIDA = Path("resultados")
SALIDA.mkdir(exist_ok=True)

PROMPT_POLIZA = """
Analiza esta página de una póliza de seguro de automóvil.

Extrae exclusivamente los datos que aparezcan en el documento.
No inventes información. Si un dato no aparece en esta página, utiliza null.

Devuelve SOLO JSON válido con esta estructura:
{
  "aseguradora": null,
  "producto": null,
  "vehiculo": {
    "marca": null,
    "modelo": null
  },
  "coberturas": [],
  "exclusiones": [],
  "fecha_inicio": null,
  "fecha_fin": null,
  "prima_neta": null,
  "impuestos": null,
  "prima_total": null
}
"""

paginas = sorted(CARPETA.glob("poliza_pag*_anonimizada_final.png"))

if len(paginas) != 6:
    raise ValueError(f"Se esperaban 6 páginas y se encontraron {len(paginas)}")

resultados = []

for pagina in paginas:
    print(f"Analizando {pagina.name}...")
    resultado = extraer(str(pagina), PROMPT_POLIZA)

    resultados.append({
        "pagina": pagina.name,
        "datos": resultado["datos"],
        "segundos": resultado["segundos"]
    })

documento = {
    "tipo_documento": "Póliza de automóvil",
    "paginas_analizadas": len(resultados),
    "resultados": resultados
}

archivo = SALIDA / "poliza_extraida.json"

with open(archivo, "w", encoding="utf-8") as f:
    json.dump(documento, f, ensure_ascii=False, indent=4)

print(f"\nExtracción terminada. Resultado guardado en: {archivo}")