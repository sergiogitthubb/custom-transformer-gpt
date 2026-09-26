import torch
import torch.nn as nn
from torch.nn import functional as F

# Hiperparámetros internos de la arquitectura en model.py
n_embd = 128     # ANTES 64. Doblamos la capacidad de representación de cada token
n_head = 4       # Lo mantenemos en 4 (128 / 4 = 32 dimensiones por cabeza)
n_layer = 6      # ANTES 4. Hacemos la red más profunda
dropout = 0.2

class Head(nn.Module):
    """ Una cabeza individual de self-attention """
    def __init__(self, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) 
        q = self.query(x) 
        
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei) # Evita depender demasiado de un solo token
        
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    """ Múltiples cabezas de atención trabajando en paralelo """
    def __init__(self, num_heads, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, block_size) for _ in range(num_heads)])
        # Capa de proyección para integrar el resultado en la conexión residual
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ Una capa lineal simple seguida de una no linealidad """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Proyección final para la conexión residual
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Bloque Transformer: Comunicación (Atención) seguida de Computación (FFWD) """
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, block_size)
        self.ffwd = FeedForward(n_embd)
        
        # LayerNorm estabiliza la red normalizando los valores antes de pasarlos a las capas
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Conexiones residuales (x = x + capa(norm(x)))
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Apilamos múltiples bloques Transformer uno detrás de otro
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        
        # Normalización final antes de predecir
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) 
        x = tok_emb + pos_emb 
        
        # Pasamos por todos los bloques Transformer
        x = self.blocks(x)
        
        # Normalizamos y sacamos las predicciones
        x = self.ln_f(x)
        logits = self.lm_head(x) 
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:] 
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] 
            logits = F.softmax(logits, dim=-1)
            concat_tensor = torch.multinomial(logits, num_samples=1)
            idx = torch.cat((idx, concat_tensor), dim=1)
        return idx