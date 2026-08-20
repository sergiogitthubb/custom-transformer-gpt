# Nano-GPT From Scratch

A step-by-step implementation of an autoregressive language model (GPT) based on the Transformer architecture, following the *from scratch* approach by Andrej Karpathy.

---

##  Current Project Status: Data Preparation & Tokenization

The project currently includes the full pipeline for dataset ingestion, character-level encoding, and batch generation for training.

### 1. Dataset Ingestion
* **Source:** *Tiny Shakespeare* (~1 MB of raw text).
* **Loading:** Automated download and in-memory loading into a contiguous stream.

### 2. Character-Level Tokenization
* **Vocabulary:** Extracted from unique characters present in the corpus ($V = 65$).
* **Mappings:**
  * `stoi` (*string to index*): Maps each character to a discrete integer ID (`torch.long`).
  * `itos` (*index to string*): Inverse lookup table to decode model outputs back into readable text.
* **Dataset Transformation:** Conversion of the entire text into a 1D tensor of contiguous integer indices stored in memory.

### 3. Data Partitioning (*Train / Val Split*)
* **Training (90%):** Data used for gradient computation and weight optimization via backpropagation.
* **Validation (10%):** Isolated holdout set used to evaluate generalization capabilities and monitor overfitting.

### 4. Batch Generator (`get_batch`)
* **Parallelism:** Extraction of 2D input matrices of shape `(batch_size, block_size)`.
* **Input ($x$) vs Target ($y$):** 1-position rightward time shift ($i+1$) to train all sub-sequences concurrently via next-token prediction.
* **Device Management:** Automatic memory placement of tensors to target hardware (`.to(device)`).

---

##  Conceptual Data Flow

```text
Raw Text ("hola")
        │
        ▼ (encode / stoi)
1D Integer Tensor [8, 14, 11, 0] (torch.long)
        │
        ▼ (get_batch: random sampling)
Input Matrix x: shape (B, T)
        │
        ▼ [Next Phase: Transformer Layers]
Token Embeddings + Positional Embeddings  ──▶  Dense Vectors (B, T, C)
```