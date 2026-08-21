import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding_table=nn.Embedding(vocab_size,vocab_size)
    def forward(self, idx, targets=None):
        
        logits=self.embedding_table(idx)
        loss=None
        if targets is not None:
            B,T,C = logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(-1)
            loss = F.cross_entropy(logits,targets)
        return logits, loss
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,_=self(idx)
            logits=logits[:,-1,:]
            logits=F.softmax(logits,dim=-1)
            concat_tensor=torch.multinomial(logits,num_samples=1)
            idx=torch.cat((idx,concat_tensor),dim=1)
        return idx
