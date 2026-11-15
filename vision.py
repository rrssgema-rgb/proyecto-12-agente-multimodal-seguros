"""Vision: pasar una imagen a un modelo multimodal local (Ollama) y pedirle texto o datos."""
import json
import time

import ollama

# qwen2.5vl:3b lee bien el texto de documentos en espanol y es ligero (3,2 GB). llava:7b es el clasico
# (el de la teoria), pero en modo JSON se quedo generando sin parar mas de 15 minutos con la factura.
# keep_alive=0: el modelo de vision se descarga de memoria en cuanto contesta (no convive con llama3.1)
MODELO_VISION = "qwen2.5vl:3b"


def ver(imagen: str, pregunta: str, modelo: str = MODELO_VISION) -> dict:
    """Manda la imagen + la pregunta al modelo. La imagen viaja en el mismo mensaje que el texto:
    el encoder de vision la convierte en tokens y el modelo la lee junto a la pregunta."""
    t0 = time.time()
    r = ollama.chat(model=modelo, options={"temperature": 0}, keep_alive=0,
                    messages=[{"role": "user", "content": pregunta, "images": [imagen]}])
    return {"texto": r["message"]["content"].strip(), "segundos": round(time.time() - t0),
            "tokens_entrada": r.get("prompt_eval_count")}


EXTRAER = """Extrae los datos de esta factura. Responde SOLO con JSON con estas claves:
{"proveedor": str, "cif_proveedor": str, "numero": str, "fecha": "dd/mm/aaaa",
 "lineas": [{"concepto": str, "unidades": int, "precio": float, "importe": float}],
 "base_imponible": float, "iva": float, "total": float}
Usa punto como separador decimal. Si un dato no aparece, pon null."""


def extraer(imagen: str, prompt: str = EXTRAER, modelo: str = MODELO_VISION) -> dict:
    """Igual que ver(), pero obligando a contestar en JSON: asi el resultado se puede usar con codigo."""
    t0 = time.time()
    # num_predict: tope de tokens de salida. Sin el, llava:7b en modo JSON no paraba nunca (mas de 15 min)
    r = ollama.chat(model=modelo, format="json", options={"temperature": 0, "num_predict": 700}, keep_alive=0,
                    messages=[{"role": "user", "content": prompt, "images": [imagen]}])
    try:
        datos = json.loads(r["message"]["content"])
    except json.JSONDecodeError:
        datos = {"error": "el modelo no devolvio un JSON valido", "crudo": r["message"]["content"]}
    return {"datos": datos, "segundos": round(time.time() - t0)}


def comprobar_factura(d: dict) -> list:
    """Nunca te fies de los numeros que 'lee' un modelo: las cuentas se comprueban con codigo."""
    avisos = []
    try:
        suma = round(sum(l["importe"] for l in d["lineas"]), 2)
        if abs(suma - d["base_imponible"]) > 0.02:
            avisos.append(f"la suma de las lineas ({suma}) no coincide con la base ({d['base_imponible']})")
        for l in d["lineas"]:
            if abs(l["unidades"] * l["precio"] - l["importe"]) > 0.02:
                avisos.append(f"'{l['concepto']}': {l['unidades']} x {l['precio']} no da {l['importe']}")
        if abs(d["base_imponible"] + d["iva"] - d["total"]) > 0.02:
            avisos.append("base + IVA no da el total")
    except (KeyError, TypeError) as e:
        avisos.append(f"falta un dato para comprobar: {e}")
    return avisos
