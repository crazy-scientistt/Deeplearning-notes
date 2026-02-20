# 🧠 Deep Learning Interview Revision Notes  
## Phase 1: Neural Network Foundations

---

## 1. Artificial Neurons, Weights, Biases & the Perceptron

### The Biological Metaphor
- A **neuron** in your brain gets signals from other neurons.  
- In a computer, we copy this: we have inputs (like $x_1, x_2, ...$), each multiplied by a **weight** ($w_i$).  
- Then we add a **bias** $b$, and put the result into an **activation function** $f$ (which decides whether the neuron should "fire" or not).  
- Output: $\hat{y} = f(\sum w_i x_i + b)$

**Example**: Imagine you want to decide whether to watch a movie. Inputs: $x_1$ = how much you like the actor (1–10), $x_2$ = how good the reviews are (1–10). Weights show how important each factor is (maybe actor weight 0.6, reviews weight 0.4). Bias can adjust the decision threshold.

### The Perceptron
- The simplest neuron: uses a **step function** – if sum is above a threshold, output 1; else 0.  
- Problem: It can only solve **linearly separable** problems (like AND, OR) but fails on XOR (where a single straight line cannot separate the two classes).  
- **Fix**: Stack many perceptrons together → **Multi-Layer Perceptron (MLP)**. This can learn complex, non‑linear boundaries.

### Weights & Biases
- **Weights** $w_i$: Tell how much each input matters. A high weight means that input strongly influences the output.  
- **Bias** $b$: Allows the neuron to fire even if all inputs are zero. It's like the "starting point" or intercept in a line equation $y = mx + b$.

**Example**: Suppose you're scoring a student: test score weight 0.7, homework weight 0.3. If the student gets 0 on both, bias could still give a small positive score if they participated in class.

### 💡 Common Interview Question
> *"What is the role of a bias term in a neural network?"*

**Answer in simple terms:** The bias helps the model be more flexible. Without bias, when all inputs are zero, the output must be zero. But in real life, sometimes we want a non‑zero output even with zero inputs. Bias lets us shift the whole function up or down.

---

## 2. Activation Functions

### Why We Need Them
- If we only use multiplications and additions (linear operations), stacking many layers is the same as one big linear layer. That can't learn complex patterns.  
- Activation functions add **non‑linearity** – they bend the line, so the network can learn curves, circles, etc.

### Key Activation Functions (with examples)

| Function | Formula | Range | Example Use |
|---|---|---|---|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | (0, 1) | Good for probability output (e.g., chance of rain) |
| **Tanh** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | (-1, 1) | Often used in hidden layers; output is centered around 0 |
| **ReLU** | $\max(0, x)$ | $[0, \infty)$ | Most common; if input positive, pass it; if negative, output 0 (like a light switch) |
| **Leaky ReLU** | $\max(0.01x, x)$ | $(-\infty, \infty)$ | Like ReLU but allows a tiny negative slope – avoids dead neurons |
| **GELU** | $x \cdot \Phi(x)$ (smooth approximation) | $(-\infty, \infty)$ | Used in modern transformers like BERT |
| **Swish** | $x \cdot \sigma(\beta x)$ | $(-\infty, \infty)$ | Used in some vision models |

**Example of ReLU**: If the input is 5, output 5; if input is –2, output 0. It's fast and simple.

### The Vanishing & Exploding Gradient Problem

- **Vanishing gradients**: When training very deep networks, the gradients (signals that tell how much to change weights) get smaller and smaller as they go back through layers. Early layers learn almost nothing.  
  - This often happens with Sigmoid or Tanh because their slopes are ≤ 0.25.  
- **Exploding gradients**: Gradients become huge – the model updates weights too much and training fails.

**How to fix them**:
- For vanishing: Use ReLU/GELU, add skip connections (like in ResNet), use good weight initialization.  
- For exploding: **Gradient clipping** – if the gradient vector is too large, scale it down to a maximum size.

**Example of gradient clipping**: Imagine you're walking and your step suddenly becomes 10 meters – you'd trip. Clipping reduces that step to a safe size like 1 meter.

### 💡 Common Interview Question
> *"Why did ReLU replace Sigmoid in hidden layers?"*

**Answer:** Sigmoid squashes values to 0–1. When the input is very large positive or negative, the slope becomes almost zero, so gradients vanish. ReLU keeps a slope of 1 for positive inputs, so gradients flow well. Also, ReLU is super fast: just $\max(0,x)$. But ReLU has a problem: if a neuron always gets negative inputs, it outputs zero forever – that's the **dying ReLU** problem. Leaky ReLU fixes that.

### ⚖️ Trade-offs: ReLU vs. GELU
| | ReLU | GELU |
|---|---|---|
| Speed | Faster | A bit slower |
| Gradient | Jumps at 0 | Smooth everywhere |
| Use case | CNNs, older models | Transformers (GPT, BERT) |

---

## 3. Feedforward Networks (MLPs)

### Architecture
- **Input layer** → one or more **hidden layers** → **output layer**.  
- Each hidden layer takes the output of the previous layer, multiplies by weights, adds bias, and applies an activation function.  
- **Universal Approximation Theorem**: A network with just one hidden layer and enough neurons can approximate any continuous function – but it doesn't tell us how easy it is to train.

**Example**: A network to predict house price: inputs = size, bedrooms, age; hidden layer learns combinations like "size × bedrooms"; output is price.

### Key Design Choices
- **Width** (how many neurons per layer): More neurons = more capacity to learn.  
- **Depth** (how many layers): Deeper networks can learn features step by step (e.g., edges → shapes → objects). Depth is often more efficient than width.

### 💡 Common Interview Question
> *"Why go deeper rather than wider?"*

**Answer:** Depth lets the model build hierarchical representations. For example, in recognizing a face, early layers detect edges, next layers detect eyes/nose, later layers detect whole faces. A wide but shallow network might need huge numbers of neurons to do the same. Deeper networks also generalize better with fewer parameters.

---

## 4. Loss Functions

### Regression: Mean Squared Error (MSE)
$$\mathcal{L}_{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$
- Measures the average squared difference between predicted and actual values.  
- Squaring makes large errors very costly (e.g., off by 10 → error 100).  
- Good for problems like predicting temperature, price, etc.

**Example**: You predict 25°, actual is 30°. Error = 5. Squared error = 25. If you were off by 10, squared error = 100 – so model will try harder to fix big mistakes.

### Classification: Cross-Entropy Loss
$$\mathcal{L}_{CE} = -\sum_{c} y_c \log(\hat{p}_c)$$
- For binary: $\mathcal{L} = -[y \log \hat{p} + (1-y)\log(1-\hat{p})]$
- Measures how different the predicted probability is from the true class.  
- If the model is very sure and wrong, the loss is huge.

**Example**: You're classifying an email as spam (1) or not spam (0). True label is spam (y=1). Model predicts probability 0.9 (good) → loss = $-\log(0.9) \approx 0.105$. If it predicts 0.1 (very wrong) → loss = $-\log(0.1) = 2.3$, much larger.

### 💡 Common Interview Question
> *"Why use Cross-Entropy instead of MSE for classification?"*

**Answer:** With MSE, if the model is confidently wrong (predicts 0 when true is 1), the gradient becomes tiny – learning stops. Cross‑entropy gives a big gradient when the model is wrong, so it corrects quickly. Also, cross‑entropy works naturally with softmax outputs.

---

## 5. Optimization & Learning

### Gradient Descent Variants

| Variant | How it works | Batch Size | Pros / Cons |
|---|---|---|---|
| **Batch GD** | Looks at all data before updating | All data | Stable but slow; needs huge memory |
| **Stochastic GD** | Updates after each sample | 1 | Fast updates but very jumpy |
| **Mini-batch GD** | Uses a small random batch | 32–512 | Most common – balance of speed and stability |

**Example**: Imagine you're trying to find the lowest point in a valley. Batch GD checks the whole valley before stepping; stochastic GD takes a step after every step you take; mini‑batch looks at a small patch and steps.

### Backpropagation & the Chain Rule
- **Forward pass**: Input goes through layers → get prediction → compute loss.  
- **Backward pass**: We go backwards, calculating how much each weight contributed to the loss (using calculus chain rule).  
- Then we adjust weights to reduce loss.

**Example**: If a weight was too high and caused a large error, we decrease it a bit.

### Optimizers

#### SGD with Momentum
- Instead of just using current gradient, we keep a "velocity" of previous gradients. This smooths the updates and helps escape small bumps.  
$$v_t = \beta v_{t-1} + (1-\beta)\nabla_w \mathcal{L}$$
$$w \leftarrow w - \eta v_t$$

**Example**: Like a ball rolling downhill – it keeps some speed from previous steps, so it doesn't stop at every tiny dip.

#### Adam (Adaptive Moment Estimation)
- Keeps track of two things: average of gradients (like momentum) and average of squared gradients (to adapt learning rate per weight).  
- This means each weight has its own learning rate that adjusts over time.  
- Good for problems with sparse data (like NLP).  

#### AdamW
- An improved version of Adam. In Adam, the weight decay (L2 penalty) didn't work correctly. AdamW **decouples** weight decay – applies it directly to weights.  
- Now it's the default for training large language models (GPT, Llama).

### 💡 Common Interview Question
> *"Why is AdamW preferred over Adam for training LLMs?"*

**Answer:** In Adam, weight decay (which helps prevent overfitting) is mixed with the adaptive learning rates, so it doesn't work as intended. AdamW separates weight decay from the gradient update, meaning all weights are shrunk by the same factor – this improves generalization, especially in huge models.

### ⚖️ Trade-offs: SGD vs. Adam
| | SGD + Momentum | Adam/AdamW |
|---|---|---|
| Generalization | Often better (finds flatter minima) | Can find sharper minima (may overfit) |
| Speed to converge | Slower | Faster |
| Hyperparameter sensitivity | Very sensitive to learning rate | More robust |
| Best for | CNNs, vision | NLP, transformers |

---

## 6. Regularization

### The Bias-Variance Trade-off
- **High bias (underfitting)**: Model too simple – misses patterns (like trying to fit a straight line to a curvy dataset).  
- **High variance (overfitting)**: Model too complex – memorizes training data, fails on new data.

**Example**: You're learning to recognize cats. High bias: model only uses color, so it misclassifies black cats. High variance: model memorizes every pixel of training cat photos, so a new cat slightly different is not recognized.

### L1 & L2 Regularization
- **L2 (Weight Decay)**: Adds penalty for large weights to the loss. Forces weights to be small.  
  $$Loss = original\_loss + \lambda \sum w_i^2$$
- **L1 (Lasso)**: Adds penalty for absolute value of weights. Can make some weights exactly zero – useful for feature selection.  
  $$Loss = original\_loss + \lambda \sum |w_i|$$

**Example**: In predicting house price, L1 might make weight of "number of windows" zero if it's not important, effectively removing that feature.

### Dropout
- During training, randomly turn off some neurons (set their output to 0) with probability $p$ (usually 0.1–0.5).  
- This forces the network to not rely too much on any single neuron – it must learn redundant representations.  
- At test time, all neurons are on, but outputs are scaled by $p$ to keep the average the same.

**Example**: Imagine a group project where randomly some members are absent during training – the team learns to work without always depending on the same person. During the final presentation, everyone is present.

### 💡 Common Interview Question
> *"How does Dropout act as a regularizer?"*

**Answer:** Dropout prevents neurons from co‑adapting – they can't rely on others to correct their mistakes. This makes each neuron learn more robust features. It's like training many smaller networks and averaging them.

### Normalization Techniques

| Method | What it normalizes | Where it's used |
|---|---|---|
| **Batch Norm** | Across the batch (all samples) for each feature | CNNs; needs large batch size |
| **Layer Norm** | Across features for each sample | Transformers, RNNs; works with any batch size |
| **RMS Norm** | Like LayerNorm but without subtracting mean | Modern LLMs (Llama, Mistral); faster |

**Why Normalize?**
- Keeps values in a range where gradients are healthy – avoids vanishing/exploding.  
- Allows faster training (can use higher learning rates).  
- Adds a bit of regularization.

**Layer Norm formula**: For a vector $\mathbf{x}$: subtract mean, divide by standard deviation, then scale and shift.  
**RMS Norm**: Just divide by root mean square (no mean subtraction) – saves computation.

### 💡 Common Interview Question
> *"Why do Transformers use Layer Norm instead of Batch Norm?"*

**Answer:** Batch Norm uses statistics across the batch – but in sequences, lengths vary and padding messes up the statistics. Also, Batch Norm needs big batches to be stable. Layer Norm works on each sample independently, so it's perfect for sequences and any batch size.

---

## 🗺️ Phase 1 Summary Map

```
Perceptron → MLP (needs non‑linear activations)
     ↓
Activation Functions → fix vanishing gradients (ReLU, GELU)
     ↓
Loss Functions → MSE (regression), Cross‑Entropy (classification)
     ↓
Backpropagation (chain rule) → Gradient Descent
     ↓
Optimizers: SGD → Momentum → Adam → AdamW
     ↓
Regularization: Dropout, L1/L2, BatchNorm / LayerNorm / RMSNorm
```

---

*✅ Phase 1 Complete. When you're ready, we can move to Phase 2: Specialized Architectures (CNNs & RNNs).*


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

