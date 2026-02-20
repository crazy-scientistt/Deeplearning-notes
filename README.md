# 🧠 Deep Learning Interview Revision Notes
## Phase 3: The Attention Revolution & Transformers

---

## 1. Seq2Seq Models & The Bottleneck Problem

### The Seq2Seq Architecture
- Introduced for machine translation (Sutskever et al., 2014).
- **Encoder**: An RNN reads the input sequence and compresses it into a single fixed-size **context vector** $\mathbf{c} = \mathbf{h}_T$.
- **Decoder**: Another RNN generates the output sequence one token at a time, conditioned on $\mathbf{c}$.

```
"The cat sat" → [ENCODER RNN] → context vector c → [DECODER RNN] → "Le chat s'est assis"
```

### The Fatal Flaw
- **Information bottleneck**: The entire input — regardless of length — must be compressed into one fixed-size vector.
- For long sentences (50+ words), early tokens are forgotten by the time the encoder finishes.
- **Empirical evidence**: Translation quality degrades sharply with sentence length.

---

## 2. The Attention Mechanism (Bahdanau, 2015)

### The Core Idea
> "Instead of reading from one compressed vector, let the decoder *look back* at all encoder hidden states and decide which ones are most relevant at each decoding step."

- At each decoder step $t$, compute a **weighted sum** over all encoder hidden states $\{\mathbf{h}_1, ..., \mathbf{h}_T\}$.
- The weights (attention scores) are **learned** — they tell the model which input positions to focus on.

### Attention Computation (Bahdanau / Additive)

**Step 1 — Score**: How relevant is encoder state $\mathbf{h}_i$ to decoder state $\mathbf{s}_t$?
$$e_{ti} = \mathbf{v}^T \tanh(W_1 \mathbf{h}_i + W_2 \mathbf{s}_t)$$

**Step 2 — Normalize**: Convert scores to a probability distribution via softmax:
$$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j} \exp(e_{tj})}$$

**Step 3 — Aggregate**: Compute context vector as a weighted sum:
$$\mathbf{c}_t = \sum_{i} \alpha_{ti} \mathbf{h}_i$$

- $\alpha_{ti}$ is the **attention weight** — how much decoder step $t$ attends to encoder position $i$.
- This creates an **alignment** between source and target tokens — interpretable and powerful.

### 💡 Common Interview Question
> *"What problem does attention solve that vanilla Seq2Seq cannot?"*

**Answer:** Vanilla Seq2Seq forces all source information through a single fixed-size vector — a hard bottleneck. Attention gives the decoder **dynamic, step-specific access** to the entire encoder output. At each decoding step, a fresh weighted combination of all encoder states is computed. This removes the bottleneck, handles long sequences gracefully, and provides an interpretable alignment between input and output tokens.

---

## 3. Scaled Dot-Product Attention (The Transformer's Engine)

### From Bahdanau to Dot-Product
- Vaswani et al. (2017) — *"Attention is All You Need"* — reformulated attention using **linear projections** and dot products, enabling full parallelism.

### Queries, Keys, and Values — The Library Analogy
- **Query (Q)**: What am I looking for? *(the decoder's current state — the search query)*
- **Key (K)**: What do I offer? *(each encoder position advertises its content)*
- **Value (V)**: What do I actually return? *(the content retrieved if matched)*

Think of it like a **soft dictionary lookup**: instead of returning one exact match, we return a weighted blend of all values, weighted by how well each key matches the query.

### The Formula
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$
- $QK^T$: Raw similarity scores — $n \times m$ matrix (every query × every key).
- $\frac{1}{\sqrt{d_k}}$: **Scaling factor** — critical.
- $\text{softmax}(\cdot)$: Normalize scores to a probability distribution.
- Final output: Weighted sum of values — $n \times d_v$.

### Why the $\sqrt{d_k}$ Scaling?
- Without scaling, dot products grow large in magnitude as $d_k$ increases (variance scales with $d_k$).
- Large dot products push softmax into **extremely saturated regions** — near one-hot distributions.
- This kills gradients (softmax becomes nearly zero everywhere except the max).
- Dividing by $\sqrt{d_k}$ keeps the dot products in a sensible range before softmax.

### 💡 Common Interview Question
> *"Why do we scale by $\sqrt{d_k}$ in attention?"*

**Answer:** The dot product $q \cdot k = \sum_{i} q_i k_i$ has variance proportional to $d_k$ when $q$ and $k$ are random unit-variance vectors. As $d_k$ grows (e.g., 64, 128), the magnitude grows too, pushing softmax into saturation — a near-zero gradient regime. Dividing by $\sqrt{d_k}$ normalizes the variance back to 1, keeping gradients healthy and attention weights well-distributed across positions.

---

## 4. Multi-Head Attention (MHA)

### Why Multiple Heads?
- Single attention looks at the sequence through **one lens** — one set of Q, K, V projections.
- Different heads can capture **different types of relationships simultaneously**:
  - Head 1: Syntactic dependencies (subject → verb)
  - Head 2: Coreference (pronoun → noun)
  - Head 3: Local context (adjacent tokens)
  - Head 4: Long-range semantic links

### The Mechanism
Project input into $h$ different Q, K, V subspaces, compute attention in each, then concatenate:

$$\text{head}_i = \text{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V)$$
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

- Each head operates in dimension $d_k = d_{\text{model}} / h$ — so total compute is comparable to single-head attention.
- $W^O$: Output projection that mixes information across heads.

### 💡 Common Interview Question
> *"What does each attention head learn?"*

**Answer:** Different heads specialize in different relationship types — some attend to syntactic structure, others to semantic similarity, some to positional proximity. This is empirically observed: specific heads in BERT and GPT have been shown to track subject-verb agreement, coreference, and even part-of-speech. MHA gives the model the capacity to jointly reason about many aspects of the sequence simultaneously, which a single attention head cannot.

---

## 5. The Full Transformer Architecture

### High-Level Structure
```
Input Tokens
     ↓
Token Embeddings + Positional Encoding
     ↓
┌─────────────────────────────────────┐
│         ENCODER BLOCK × N           │
│  ┌─────────────────────────────┐   │
│  │  Multi-Head Self-Attention  │   │
│  └──────────┬──────────────────┘   │
│             │ Add & Norm (Residual) │
│  ┌──────────▼──────────────────┐   │
│  │  Feed-Forward Network (FFN) │   │
│  └──────────┬──────────────────┘   │
│             │ Add & Norm           │
└─────────────┼───────────────────────┘
              ↓ (Context vectors)
┌─────────────────────────────────────┐
│         DECODER BLOCK × N           │
│  ┌─────────────────────────────┐   │
│  │  Masked Multi-Head Self-Attn│   │
│  └──────────┬──────────────────┘   │
│             │ Add & Norm           │
│  ┌──────────▼──────────────────┐   │
│  │  Cross-Attention            │   │
│  │  (Q from decoder,           │   │
│  │   K,V from encoder)         │   │
│  └──────────┬──────────────────┘   │
│             │ Add & Norm           │
│  ┌──────────▼──────────────────┐   │
│  │  Feed-Forward Network (FFN) │   │
│  └──────────┬──────────────────┘   │
│             │ Add & Norm           │
└─────────────┼───────────────────────┘
              ↓
        Linear + Softmax → Output Probabilities
```

### Key Components

**1. Self-Attention**
- Q, K, V all come from the **same sequence** — every token attends to every other token in the same sequence.
- In the encoder: **bidirectional** — every token sees the full context.
- In the decoder: **masked** (causal) — each token can only see past tokens.

**2. Feed-Forward Network (FFN)**
- Applied independently to each position: $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$
- Typically $d_{ff} = 4 \times d_{\text{model}}$ — this is where most parameters live in a Transformer.
- Acts as a **per-token memory** / feature extractor after attention has mixed information across positions.

**3. Add & Norm (Residual + Layer Norm)**
- **Residual connection**: $x \leftarrow x + \text{SubLayer}(x)$ — same purpose as ResNets: gradient highway.
- **Layer Norm**: Applied after (Post-LN, original paper) or before (Pre-LN, more stable training — used in modern LLMs).

### 💡 Common Interview Question
> *"What is the role of the FFN in a Transformer block, and why is it so large?"*

**Answer:** After self-attention mixes information across positions, the FFN processes each position independently and identically. It acts as a per-token computation step — research (Geva et al., 2021) shows FFN layers store factual associations (like key-value memories). The expansion to $4d_{\text{model}}$ gives the model representational capacity to apply complex transformations. About 2/3 of a Transformer's parameters live in FFN layers.

---

## 6. Encoder vs. Decoder Blocks

| Feature | Encoder Block | Decoder Block |
|---|---|---|
| **Self-attention** | Bidirectional (sees all tokens) | Causal / Masked (sees only past) |
| **Cross-attention** | None | Yes — attends to encoder output |
| **Primary use** | Understanding, representation | Generation, translation |
| **Examples** | BERT, RoBERTa | GPT, Llama, Claude |
| **Both** | — | T5, BART, original Transformer |

---

## 7. Positional Encoding

### Why It's Needed
- Self-attention is **permutation invariant** — shuffling the input tokens produces the same result (attention weights change but the mechanism doesn't know the order).
- Positional encodings inject **sequence order** information into the token embeddings.

### Sinusoidal Positional Encoding (Original — Vaswani 2017)
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

- Each position gets a unique vector of sine/cosine waves at different frequencies.
- **Absolute position** encoding — added once to the input embeddings.
- **Advantage**: Can generalize to sequence lengths not seen during training (extrapolation).
- **Disadvantage**: Fixed, not learned; doesn't model relative positions naturally.

### Learned Absolute Positional Embeddings
- Simply learn a lookup table of position embeddings: $PE \in \mathbb{R}^{L_{max} \times d_{\text{model}}}$.
- Used in BERT and early GPT models.
- **Disadvantage**: Cannot generalize beyond $L_{max}$ seen during training.

### Rotary Positional Embeddings (RoPE) — Modern Standard
- Used in: LLaMA, Mistral, Claude, Falcon, PaLM 2, and most modern LLMs.
- **Core idea**: Instead of adding positional info to embeddings, **rotate** the Q and K vectors in the attention computation by an angle proportional to their position.
- The dot product $q_m \cdot k_n$ then naturally depends on the **relative position** $(m - n)$ — not absolute positions.

$$\text{RoPE}(\mathbf{x}, m) = \mathbf{x} \cdot e^{im\theta}$$

**Why RoPE is superior:**
- Encodes **relative positions** intrinsically — the model knows "these two tokens are 5 apart," not just "this token is at position 37."
- **Extrapolates better** to longer sequences than learned absolute embeddings.
- Can be extended (e.g., YaRN, LongRoPE) to dramatically increase context window at inference.
- Compatible with KV caching (Phase 5 topic).

### ⚖️ Trade-offs: Sinusoidal vs. Learned vs. RoPE
| | Sinusoidal | Learned Absolute | RoPE |
|---|---|---|---|
| Relative position | Implicit | No | Yes (explicit) |
| Length generalization | Moderate | Poor | Good / Extendable |
| Parameters | None | $L_{max} \times d$ | None |
| Modern usage | Rare | BERT, GPT-2 | LLaMA, Mistral, most LLMs |

### 💡 Common Interview Question
> *"Why have most modern LLMs moved to RoPE over learned positional embeddings?"*

**Answer:** Learned absolute embeddings cannot generalize beyond the maximum sequence length seen during training — the model has literally never seen a position embedding for token #5000 if trained on sequences up to 4096. RoPE encodes position as a rotation in the Q/K space, and the attention score naturally depends only on the *relative* distance between tokens. This makes it more length-generalizable, and with techniques like YaRN, the effective context window can be extended at inference time without full retraining.

---

## 8. Causal Masking

### What Is It?
- Used in **decoder self-attention** to enforce the autoregressive property during training.
- Prevents token $i$ from attending to tokens $j > i$ (future tokens).
- Implemented by adding $-\infty$ to the attention logits for illegal positions before softmax:

$$\text{Masked Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

where $M_{ij} = 0$ if $j \leq i$, else $-\infty$.

- After softmax, $e^{-\infty} = 0$ — those positions get zero weight, effectively invisible.

```
Attention mask (lower triangular):
      t1  t2  t3  t4
  t1 [ 1   0   0   0 ]
  t2 [ 1   1   0   0 ]
  t3 [ 1   1   1   0 ]
  t4 [ 1   1   1   1 ]
```

### Why Is It Critical?

**Training**: The model is trained on full sequences with teacher forcing — the ground truth tokens are provided as inputs. Without masking, the model could "cheat" by looking at future tokens, making the task trivial and the model useless for generation.

**Generation**: During inference, tokens are generated **autoregressively** — one at a time, left to right. The mask ensures training behavior matches inference behavior.

### 💡 Common Interview Question
> *"Why is causal masking necessary during training of a decoder?"*

**Answer:** During training we process the entire target sequence in one forward pass for efficiency (teacher forcing). Without masking, position $t$ could attend to the correct answer at position $t+1$, making the task trivial — the model never actually learns to predict the next token, just to copy from the future. The causal mask enforces that each position can only see its own past, making training consistent with the autoregressive generation process at inference time.

---

## 9. Complexity & Scalability of Attention

### Time & Space Complexity
- Self-attention computes $QK^T$: an $n \times n$ matrix for a sequence of length $n$.
- **Time complexity**: $O(n^2 \cdot d)$
- **Space complexity**: $O(n^2)$ — storing the attention matrix.
- This is the **quadratic bottleneck** — doubling sequence length → 4× the memory.

### Why This Matters
- For $n = 1000$: manageable. For $n = 100{,}000$ (long documents): prohibitive.
- Motivates efficient attention variants: **FlashAttention**, **Linear Attention**, **Sparse Attention** — covered in Phase 5.

### ⚖️ Grand Trade-off: RNN vs. Transformer

| Property | RNN / LSTM | Transformer |
|---|---|---|
| **Parallelism** | None — sequential | Full — all positions at once |
| **Long-range dependencies** | Poor (vanishing gradient) | Excellent ($O(1)$ path length) |
| **Time complexity** | $O(n \cdot d^2)$ | $O(n^2 \cdot d)$ |
| **Memory (inference)** | $O(d)$ — fixed hidden state | $O(n^2)$ — grows with context |
| **Inductive bias** | Sequential order built-in | Must learn order via pos. encoding |
| **Scalability** | Poor | Excellent (scales with data + params) |

---

## 🗺️ Phase 3 Summary Map

```
Seq2Seq bottleneck
     ↓
Attention (Bahdanau) — dynamic weighted lookup over encoder states
     ↓
Scaled Dot-Product Attention: Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
     ↓
Multi-Head Attention — parallel attention in multiple subspaces
     ↓
Transformer Block:
  Self-Attention → Add & Norm → FFN → Add & Norm
     ↓
Encoder (bidirectional) vs. Decoder (causal / masked)
     ↓
Positional Encoding:
  Sinusoidal → Learned → RoPE (modern standard)
     ↓
Causal Masking — enforces autoregressive training/inference consistency
     ↓
Bottleneck: O(n²) attention — motivates FlashAttention, GQA (Phase 5)
```

---

*✅ Phase 3 Complete. Confirm when ready for **Phase 4: Large Language Models — Architecture & Training Pipeline (Pre-training → SFT → RLHF → DPO)**.*
