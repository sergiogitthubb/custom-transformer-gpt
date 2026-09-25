import os
import requests
import torch
import tiktoken

# Dataset path
DATA_PATH = "data/input.txt"
def load_data():
    # Asegúrate de que el archivo existe en esta ruta
    DATA_PATH = "data/input.txt" 
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    return text

raw_text = load_data()

# --- NUEVA TOKENIZACIÓN BPE (Byte-Pair Encoding) ---
# Instanciamos el tokenizador oficial de GPT-2
enc = tiktoken.get_encoding("gpt2")

# El tamaño del vocabulario da un salto enorme: de 65 a 50.257
vocab_size = enc.n_vocab

def encode(s: str) -> list[int]:
    """Convierte texto en una lista de tokens BPE enteros"""
    return enc.encode(s, allowed_special={"<|endoftext|>"})

def decode(l: list[int]) -> str:
    """Convierte una lista de tokens BPE de vuelta a texto"""
    return enc.decode(l)
# ---------------------------------------------------

# 3. Convertir todo el texto a un único tensor 1D de enteros
data = torch.tensor(encode(raw_text), dtype=torch.long)

# 4. Separación en Train (90%) y Validación (10%)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str, batch_size: int, block_size: int, device: str = "cpu"):
    data_source = train_data if split == "train" else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    
    x = torch.stack([data_source[i : i + block_size] for i in ix])
    y = torch.stack([data_source[i + 1 : i + block_size + 1] for i in ix])
    
    return x.to(device), y.to(device)

if __name__ == "__main__":
    print(f"Longitud del texto original: {len(raw_text)} caracteres")
    print(f"Longitud del tensor de datos: {len(data)} tokens BPE")
    print(f"Tamaño del vocabulario: {vocab_size} tokens únicos")
    
    ejemplo = "hola mundo, esto es un test"
    codificado = encode(ejemplo)
    print(f"\nTexto: '{ejemplo}'")
    print(f"Codificado: {codificado}")
    print(f"Decodificado: '{decode(codificado)}'")