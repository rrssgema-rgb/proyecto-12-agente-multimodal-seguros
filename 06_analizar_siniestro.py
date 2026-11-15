"""Análisis visual de vehículos implicados en siniestros."""

import sys
from vision import ver

PROMPT_SINIESTRO = """
Analiza la fotografía de este vehículo relacionada con un siniestro
de automóvil.

Devuelve la información organizada en estos apartados:

1. VEHÍCULO:
   Identifica marca, modelo, color y tipo de vehículo si son visibles.

2. DAÑOS VISIBLES:
   Describe las zonas dañadas, los desperfectos y su localización.

3. GRAVEDAD APARENTE:
   Indica si los daños parecen leves, moderados o graves.
   Es una valoración exclusivamente visual, no una peritación.

4. LIMITACIONES:
   Indica qué información no puede determinarse con certeza
   a partir de la fotografía.

No inventes información. Si un dato no es visible, indícalo.
No identifiques personas ni intentes leer matrículas.
"""

if len(sys.argv) < 2:
    print("Uso: python 06_analizar_siniestro.py ruta_imagen")
    sys.exit(1)

imagen = sys.argv[1]

print("Analizando fotografía del siniestro...")

resultado = ver(imagen, PROMPT_SINIESTRO)

print("\n--- RESULTADO DEL ANÁLISIS ---\n")
print(resultado["texto"])
print(f"\nTiempo de respuesta: {resultado['segundos']} segundos")