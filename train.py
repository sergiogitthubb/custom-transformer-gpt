import torch
from dataset import vocab_size, get_batch, decode
from model import GPTLanguageModel

# Hyperparameters & Configuration
batch_size = 64  # ANTES 32. Procesamos más ejemplos en paralelo
block_size = 64  # ANTES 8. 

max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

# Reproducibility
torch.manual_seed(1337)

# Pasar block_size al modelo
model = GPTLanguageModel(vocab_size, block_size)
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size=batch_size, block_size=block_size, device=device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# SI ESTO NO SE IMPRIME, EL SCRIPT NO SE ESTÁ EJECUTANDO
print(f"Entrenando en el dispositivo: {device}")

for iter in range(max_iters):
    
    # 1. Evaluación periódica
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Iteración {iter:4d} | Pérdida Train: {losses['train']:.4f} | Pérdida Val: {losses['val']:.4f}")

    # 2. Extracción aleatoria del batch
    xb, yb = get_batch('train', batch_size=batch_size, block_size=block_size, device=device)

    # 3. Forward pass 
    logits, loss = m(xb, yb)

    # 4. Limpieza de gradientes previos
    optimizer.zero_grad(set_to_none=True)

    # 5. Cálculo de gradientes
    loss.backward()

    # 6. Actualización de parámetros
    optimizer.step()

print("Entrenamiento completado.\n")
torch.save(m.state_dict(), 'nano_gpt_pesos.pt')
print("Modelo guardado exitosamente en 'nano_gpt_pesos.pt'")
# Text generation using trained model
print("--- Generación de texto tras el entrenamiento ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
resultado = m.generate(context, max_new_tokens=400)

print(decode(resultado[0].tolist()))