import os
from datasets import load_dataset

# Creamos la carpeta data si no existe
os.makedirs("data", exist_ok=True)
archivo_salida = "data/input.txt"

# Objetivo: 30 MB de texto (puedes subirlo a 50 o 100 si luego quieres más)
TARGET_SIZE_MB = 30
TARGET_SIZE_BYTES = TARGET_SIZE_MB * 1024 * 1024 

print(f"Conectando a Hugging Face para descargar Wikipedia en español...")

# streaming=True es la magia: no descarga los 20 GB de Wikipedia, 
# sino que nos va dando artículos uno a uno sobre la marcha.
dataset = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)

current_size = 0

with open(archivo_salida, "w", encoding="utf-8") as f:
    for i, article in enumerate(dataset):
        # Extraemos el texto del artículo y le damos un par de saltos de línea
        text = article['text'] + "\n\n"
        f.write(text)
        
        # Calculamos cuánto ocupa este texto en bytes
        current_size += len(text.encode('utf-8'))
        
        # Un pequeño print cada 100 artículos para ver el progreso
        if i % 100 == 0:
            print(f"Descargados {current_size / (1024*1024):.2f} MB / {TARGET_SIZE_MB} MB...")
            
        # Si llegamos a nuestro objetivo, cortamos la descarga
        if current_size >= TARGET_SIZE_BYTES:
            break

print(f"\n¡Éxito! Se han guardado {current_size / (1024*1024):.2f} MB de Wikipedia limpia en {archivo_salida}")