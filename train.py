import torch
from dataset import vocab_size, get_batch, decode
from model import BigramLanguageModel

# Hyperparameters & Configuration
batch_size = 32
block_size = 8
max_iters = 10000
eval_interval = 1000
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

# Reproducibility
torch.manual_seed(1337)

# Model instance and optimizer
model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Training loop
print(f"Entrenando en el dispositivo: {device}")

for iter in range(max_iters):
    # Random data batch extraction
    xb, yb = get_batch('train', batch_size=batch_size, block_size=block_size, device=device)

    # b. Forward pass 
    logits, loss = m(xb, yb)

    # c. Clean previous gradient
    optimizer.zero_grad(set_to_none=True)

    # d. Gradient calculation
    loss.backward()

    # e. Update parameters
    optimizer.step()

    # Print periodic progress
    if iter % eval_interval == 0:
        print(f"Iteración {iter:4d} | Pérdida (Loss): {loss.item():.4f}")

print(f"Pérdida final: {loss.item():.4f}\n")

# Text generation using trained model
print("--- Generación de texto tras el entrenamiento ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
resultado = m.generate(context, max_new_tokens=400)

print(decode(resultado[0].tolist()))