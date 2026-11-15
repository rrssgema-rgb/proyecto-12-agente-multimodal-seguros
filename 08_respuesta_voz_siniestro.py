"""Genera una respuesta hablada al cliente de un siniestro usando Piper."""

from pathlib import Path
from audio import hablar

CARPETA = Path("salidas")
CARPETA.mkdir(exist_ok=True)

respuesta = (
    "Buenos días. Hemos recibido su comunicación sobre el accidente "
    "y sentimos lo ocurrido. Para tramitar el siniestro, necesitaremos "
    "las fotografías de los daños y el atestado policial, si está disponible. "
    "También puede enviarnos la factura médica para comprobar si los gastos "
    "están cubiertos por su póliza. Una vez recibida la documentación, "
    "podremos gestionar la valoración del vehículo y orientarle sobre "
    "su reparación. Muchas gracias por contactar con nosotros."
)

resultado = hablar(
    respuesta,
    str(CARPETA / "respuesta_voz_siniestro.wav"),
    motor="piper"
)

print("Respuesta generada correctamente.")
print("Archivo:", resultado["archivo"])
print("Tiempo de generación:", resultado["segundos_generar"], "segundos")
print("Duración del audio:", resultado["duracion"], "segundos")