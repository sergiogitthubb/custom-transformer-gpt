## NANO-GPT FROM SCRATCH

DATA (hola) -> (embedding)FLOATING DENSE VECTOR [0.5, 0.2, 0.3] ->
TOKENIZATION, it is important since the transformer wouldn't know the order of the tokens;
We have two matrix the first one is the embedding matrix of size VocabSize x Embedding_Dimension (C)
And Positional Embeddings, in order to fix de positional doubts
Finally the rows linked of the words gets added into a one vector and thats the resultant tensor 