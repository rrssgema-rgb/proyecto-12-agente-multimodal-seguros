
"""Valida los resultados existentes del Ejercicio 1 de visión aplicada a seguros.

No ejecuta modelos ni modifica los análisis originales.
Actualiza únicamente los apartados de validación.
"""

import json
import re
from pathlib import Path
from decimal import Decimal, InvalidOperation

ARCHIVO = Path("resultados") / "ejercicio1_seguros.json"

CAMPOS_POLIZA = [
    "aseguradora",
    "producto",
    "vehiculo",
    "coberturas",
    "exclusiones",
    "fecha_inicio",
    "fecha_fin",
    "prima_neta",
    "impuestos",
    "prima_total"
]

CAMPOS_VEHICULO_POLIZA = ["marca", "modelo"]

APARTADOS_SINIESTRO = {
    "vehiculo": ["vehículo", "vehiculo"],
    "danos_visibles": ["daños visibles", "danos visibles"],
    "gravedad_aparente": ["gravedad aparente"],
    "limitaciones": ["limitaciones"]
}


def esta_vacio(valor):
    """Comprueba si un dato está vacío o no informado."""

    return (
        valor is None
        or (isinstance(valor, str) and not valor.strip())
        or (isinstance(valor, (list, dict)) and len(valor) == 0)
    )


def validar_vehiculo(resultado):
    """Valida la presencia del análisis visual y sus apartados."""

    analisis = resultado.get("analisis")

    if not isinstance(analisis, str):
        analisis = ""

    texto = analisis.lower()

    apartados_detectados = []
    apartados_faltantes = []

    for apartado, variantes in APARTADOS_SINIESTRO.items():
        if any(termino in texto for termino in variantes):
            apartados_detectados.append(apartado)
        else:
            apartados_faltantes.append(apartado)

    return {
        "resultado_obtenido": bool(analisis.strip()),
        "analisis_no_vacio": bool(analisis.strip()),
        "apartados_esperados": list(APARTADOS_SINIESTRO.keys()),
        "apartados_detectados": apartados_detectados,
        "apartados_faltantes": apartados_faltantes,
        "estructura_correcta": (
            bool(analisis.strip())
            and len(apartados_faltantes) == 0
        )
    }


def validar_poliza(resultado):
    """Comprueba estructura, campos informados y campos sin datos."""

    datos = resultado.get("analisis")

    if not isinstance(datos, dict):
        return {
            "estructura_correcta": False,
            "campos_faltantes": CAMPOS_POLIZA,
            "campos_nulos": [],
            "campos_vacios": [],
            "datos_completos": False
        }

    campos_faltantes = [
        campo for campo in CAMPOS_POLIZA
        if campo not in datos
    ]

    campos_nulos = [
        campo for campo in CAMPOS_POLIZA
        if campo in datos and datos[campo] is None
    ]

    campos_vacios = [
        campo for campo in CAMPOS_POLIZA
        if campo in datos
        and esta_vacio(datos[campo])
        and datos[campo] is not None
    ]

    # Comprobación adicional de los datos internos del vehículo.
    datos_vehiculo = datos.get("vehiculo")

    if isinstance(datos_vehiculo, dict):
        for campo in CAMPOS_VEHICULO_POLIZA:
            nombre = f"vehiculo.{campo}"

            if campo not in datos_vehiculo:
                campos_faltantes.append(nombre)
            elif datos_vehiculo[campo] is None:
                campos_nulos.append(nombre)
            elif esta_vacio(datos_vehiculo[campo]):
                campos_vacios.append(nombre)

    elif "vehiculo" in datos and datos_vehiculo is not None:
        campos_vacios.append("vehiculo (estructura no válida)")

    estructura_correcta = len(campos_faltantes) == 0

    datos_completos = (
        estructura_correcta
        and len(campos_nulos) == 0
        and len(campos_vacios) == 0
    )

    return {
        "estructura_correcta": estructura_correcta,
        "campos_obligatorios": CAMPOS_POLIZA,
        "campos_faltantes": campos_faltantes,
        "campos_nulos": campos_nulos,
        "campos_vacios": campos_vacios,
        "datos_completos": datos_completos
    }


def convertir_importe(valor):
    """Convierte un importe textual en Decimal sin alterar el original."""

    if valor is None:
        return None

    coincidencias = re.findall(r"\d[\d.,]*", str(valor))

    if not coincidencias:
        return None

    numero = coincidencias[-1]

    if "," in numero and "." in numero:
        if numero.rfind(",") > numero.rfind("."):
            numero = numero.replace(".", "").replace(",", ".")
        else:
            numero = numero.replace(",", "")
    elif "," in numero:
        numero = numero.replace(",", ".")

    try:
        return Decimal(numero)
    except InvalidOperation:
        return None


def comprobar_importes(datos):
    """Comprueba la suma de prima neta e impuestos sin modificar importes."""

    prima_neta = convertir_importe(datos.get("prima_neta"))
    impuestos = convertir_importe(datos.get("impuestos"))
    prima_total = convertir_importe(datos.get("prima_total"))

    if None in (prima_neta, impuestos, prima_total):
        return {
            "comprobacion_realizada": False,
            "motivo": "No se han podido interpretar todos los importes."
        }

    suma = prima_neta + impuestos
    diferencia = prima_total - suma

    return {
        "comprobacion_realizada": True,
        "prima_neta_mas_impuestos": str(suma),
        "prima_total_extraida": str(prima_total),
        "diferencia": str(diferencia),
        "coinciden": diferencia == Decimal("0")
    }


def main():

    if not ARCHIVO.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo: {ARCHIVO}"
        )

    with open(ARCHIVO, "r", encoding="utf-8") as f:
        documento = json.load(f)

    resultados = documento.get("resultados", [])

    if not isinstance(resultados, list) or len(resultados) != 2:
        raise ValueError(
            "Se esperaban dos resultados: fotografía y póliza."
        )

    for resultado in resultados:

        tipo = resultado.get("tipo")

        if tipo == "fotografia_siniestro":
            resultado["validacion"] = validar_vehiculo(resultado)

        elif tipo == "documento_poliza":
            resultado["validacion"] = validar_poliza(resultado)

            datos = resultado.get("analisis", {})

            if isinstance(datos, dict):
                resultado["validacion"]["comprobacion_importes"] = (
                    comprobar_importes(datos)
                )

        else:
            raise ValueError(
                f"Tipo de resultado no reconocido: {tipo}"
            )

    # Se conserva todo el contenido original de los análisis.
    with open(ARCHIVO, "w", encoding="utf-8") as f:
        json.dump(
            documento,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("=" * 60)
    print("VALIDACIÓN DEL EJERCICIO 1")
    print("=" * 60)

    for resultado in resultados:
        print(f"\nImagen: {resultado.get('imagen')}")
        print(f"Tipo: {resultado.get('tipo')}")
        print("Validación:")
        print(json.dumps(
            resultado["validacion"],
            ensure_ascii=False,
            indent=4
        ))

    print("\n" + "=" * 60)
    print(f"Archivo actualizado: {ARCHIVO}")
    print("Los análisis y los importes originales se han conservado.")
    print("No se ha ejecutado ningún modelo.")
    print("=" * 60)


if __name__ == "__main__":
    main()
