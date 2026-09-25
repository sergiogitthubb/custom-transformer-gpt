import torch
from dataset import vocab_size, decode, encode
from model import GPTLanguageModel


block_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Cargando el modelo...")

# 1. Creamos la estructura vacía del modelo
model = GPTLanguageModel(vocab_size, block_size)

# 2. Cargamos los pesos entrenados desde el archivo
model.load_state_dict(torch.load('nano_gpt_pesos.pt', map_location=device, weights_only=True))
m = model.to(device)

# Lo ponemos en modo evaluación 
m.eval()
print("Modelo cargado y listo.\n")

# 3. Interfaz básica en la terminal para interactuar
while True:
    prompt = input("Escribe el inicio de la frase (o 'salir'): ")
    if prompt.lower() == 'salir':
        break
    
    # Codificamos tu prompt inicial
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    
    print("\nGenerando...")
    # Le pedimos 500 tokens nuevos
    resultado = m.generate(context, max_new_tokens=500)
    
    # Decodificamos e imprimimos
    print("-----------------------------------")
    print(decode(resultado[0].tolist()))
    print("-----------------------------------\n")