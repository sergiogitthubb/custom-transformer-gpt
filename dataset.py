import os
import requests
import torch

# Dataset path
DATA_PATH = "data/input.txt"
# Loading data, in case that the data dir doesn't exists it creates it, then if the file .txt doesn't exists it downloads it
# making use of requests.get, then opens it and returns it
def load_data():
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(DATA_PATH):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write(requests.get(url).text)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    return text

# 2. Vocabulario y funciones de codificación/decodificación
raw_text = load_data()
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str) -> list[int]:
    """Convierte texto en una lista de números enteros"""
    return [stoi[c] for c in s]

def decode(l: list[int]) -> str:
    """Convierte una lista de números enteros de vuelta a texto"""
    return "".join([itos[i] for i in l])
# 3. Convertir todo el texto a un único tensor 1D de enteros
data = torch.tensor(encode(raw_text), dtype=torch.long)

# 4. Separación en Train (90%) y Validación (10%)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str, batch_size: int, block_size: int, device: str = "cpu"):
    """
    Genera un pequeño lote de entradas (x) y objetivos (y)
    x: tensor de forma (batch_size, block_size)
    y: tensor de forma (batch_size, block_size) desplazado 1 posición
    """
    data_source = train_data if split == "train" else val_data
    # Genera índices aleatorios donde empezar a cortar el texto
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    
    # Extrae los bloques consecutivos
    x = torch.stack([data_source[i : i + block_size] for i in ix])
    # y es exactamente x pero desplazado un carácter hacia adelante
    y = torch.stack([data_source[i + 1 : i + block_size + 1] for i in ix])
    
    return x.to(device), y.to(device)
if __name__ == "__main__":
    print(f"Longitud del texto: {len(raw_text)} caracteres")
    print(f"Tamaño del vocabulario: {vocab_size} caracteres únicos")
    print(f"Vocabulario: {''.join(chars)}")
    
    ejemplo = "hola mundo"
    codificado = encode(ejemplo)
    print(f"\nTexto: '{ejemplo}'")
    print(f"Codificado: {codificado}")
    print(f"Decodificado: '{decode(codificado)}'")