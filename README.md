# Nano-GPT From Scratch

A complete, step-by-step implementation of an autoregressive generative language model (GPT) based on the Transformer architecture. Inspired by Andrej Karpathy's "from scratch" approach, this project evolves from a basic character-level text generator to a sub-word (BPE) foundation model trained on real-world encyclopedic data.

Developed as a rigorous engineering exercise to understand the mathematical foundations of Self-Attention, Multi-Head Attention, and model training under strict hardware constraints (6GB VRAM).

---

## 🚀 Project Highlights & Evolution

### 1. Dataset & Data Ingestion Pipeline
* **Initial State:** *Tiny Shakespeare* (~1 MB, character-level).
* **Final State:** 30 MB of clean Spanish Wikipedia.
* **Pipeline:** Utilizes Hugging Face's `datasets` library with **streaming** capabilities to download, clean, and process specific chunks of data without overloading local storage.

### 2. Tokenization: The BPE Upgrade
* **Initial State:** Character-level mapping ($V = 65$).
* **Final State:** Byte-Pair Encoding (BPE) using OpenAI's `tiktoken` (`gpt2` encoding).
* **Vocabulary Size:** $V = 50,257$ tokens.
* **Impact:** Drastically increases the model's receptive field, allowing the Transformer blocks to process whole words and semantic chunks rather than spending computational power learning spelling.

### 3. Transformer Architecture Implementation
Built entirely using PyTorch primitives, featuring:
* **Scaled Dot-Product Attention:** Implementation of `Queries`, `Keys`, and `Values`, with the crucial `C ** -0.5` variance normalization to prevent softmax saturation and vanishing gradients.
* **Multi-Head Attention:** Parallel attention heads concatenated to allow the model to focus on different linguistic phenomena simultaneously.
* **Feed-Forward Networks:** Non-linear expansion layers.
* **Pre-Norm Formulation:** `LayerNorm` applied before the attention and feed-forward blocks for training stability.
* **Residual Connections:** Additive bypasses to allow gradient flow deep into the network.
* **Regularization:** `Dropout` implemented across layers to prevent overfitting.

### 4. Hardware Optimization (VRAM Constraints)
Engineered to train efficiently on a legacy **NVIDIA GTX 1060 (6GB VRAM)**. Hyperparameters were strictly tuned to maximize GPU utilization (~93%) while avoiding `CUDA out of memory` errors:
* `batch_size = 64`
* `block_size = 64`
* `n_embd = 128` (Embedding dimensions)
* `n_heads = 4`, `n_layers = 6`

---

## 📁 Project Structure

* `prepare_data.py`: Connects to Hugging Face, streams the Spanish Wikipedia dataset, extracts the required MBs of clean text, and saves it locally.
* `dataset.py`: Handles BPE tokenization (`tiktoken`), integer tensor conversion, Train/Val splitting, and the `get_batch` generator for parallel tensor extraction.
* `model.py`: The core PyTorch architecture containing the `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, and `LanguageModel` classes.
* `train.py`: The training loop. Manages forward/backward passes, the AdamW optimizer, loss evaluation (CrossEntropy), and model checkpointing (`nano_gpt_pesos.pt`).
* `generate.py`: Autoregressive inference script. Loads the trained weights and generates novel text token-by-token based on a user-provided prompt.

---

## 🧠 Academic Insights: Syntax vs. Semantics
Training a ~15M parameter model on a 30 MB BPE dataset yields a fascinating phenomenon documented in LLM literature: **Word Salad with Perfect Syntax**.
* **Syntactic Acquisition:** The model successfully learns complex grammatical rules (gender/number agreement, punctuation, HTML table structures from Wikipedia markdown) purely through mathematical optimization in 5,000 iterations.
* **Semantic Boundary:** Due to the massive vocabulary size (50k) relative to the training data volume (30MB), the model lacks the "world knowledge" to connect concepts factually, resulting in sentences that are grammatically flawless but semantically surreal. This proves the architecture's ability to abstract structural rules independently of meaning.

---

## ⚙️ How to Run

1. Install dependencies:
   > pip install torch tiktoken datasets

2. Prepare the dataset (Downloads 30MB of Wikipedia to data/input.txt):
   > python prepare_data.py

3. Train the model (Note: adjust batch_size in train.py if you hit CUDA OOM):
   > python train.py

4. Generate text (Interactive prompt mode):
   > python generate.py

---

## 📊 Conceptual Data Flow

```text
Raw Text (Wikipedia) 
        │
        ▼ tiktoken (Byte-Pair Encoding)
1D Integer Tensor [45, 1205, 502, ...] (Vocabulary: 0 - 50,257)
        │
        ▼ get_batch (Random Sampling)
Input Matrix X: shape (B, T)
        │
        ▼ 
Token Embeddings (B, T, C) + Positional Embeddings (T, C)
        │
        ▼ 
[Nx Transformer Blocks]
  ├── Pre-LayerNorm
  ├── Multi-Head Self-Attention (Scaled Dot-Product)
  ├── Residual Add
  ├── Pre-LayerNorm
  ├── Feed-Forward Network
  └── Residual Add
        │
        ▼
Logits (B, T, Vocab_Size) 
        │
        ▼ Cross-Entropy / Softmax
Next Token Prediction