# 🧠 Deep Learning Interview Revision Notes
## Phase 1: Neural Network Foundations

---

## 🔹 Artificial Neuron

**Kya hota hai?**
Tumhara brain neurons se bana hai. Har neuron doosre neurons se signals leta hai, kuch process karta hai, aur aage bhej deta hai. Computer mein hum exactly yahi copy karte hain — artificially.

Ek artificial neuron yeh kaam karta hai:
- Inputs leta hai: $x_1, x_2, ...$
- Har input ko uske **weight** $w_i$ se multiply karta hai
- Sab add karta hai, phir **bias** $b$ add karta hai
- Result ko **activation function** mein daalta hai jo decide karta hai ke neuron fire kare ya nahi

$$\hat{y} = f\left(\sum w_i x_i + b\right)$$

---

## 🔹 Weights

**Kyun hote hain?**
Har input equally important nahi hota. Weight batata hai — "is input ko kitna seriously lena hai?" Zyada weight = zyada influence on final output.

**Example – Movie dekhni hai ya nahi?**
Maan lo:
- $x_1$ = Actor kitna pasand hai (1–10) → weight **0.6**
- $x_2$ = Reviews kitne ache hain (1–10) → weight **0.4**

Matlab tumhare liye actor, reviews se zyada matter karta hai. Actor 9/10 ho aur reviews 3/10 — tum phir bhi jaoge. Weight ne tumhari personal preference capture ki.

---

## 🔹 Bias

**Kyun chahiye?**
Maan lo tumhare saare inputs zero hain. Bina bias ke, output bhi hamesha zero hoga — koi baat nahi kya inputs the. Real life mein yeh almost kabhi sahi nahi hota. Bias ek "starting push" deta hai neuron ko, taake woh zero se aage bhi soch sake.

Equation $y = mx + b$ yaad hai? Wahan $b$ intercept tha — yahan bhi exactly same kaam karta hai.

**Example – Student Scoring**
Maan lo ek student ka:
- Test score → weight 0.7
- Homework → weight 0.3
- Dono mein zero aaya

Lekin us student ne poore semester class mein participate kiya. Bias woh participation credit hai jo usse phir bhi kuch marks deta hai — even when everything else is zero.

⚠️ Bina bias ke model bahut rigid ho jaata hai — real world ka data fit karna mushkil ho jaata hai.

---

## 🔹 Perceptron

**Kya hota hai?**
Sabse basic neuron. Sirf do kaam karta hai — inputs ka weighted sum nikalo, aur agar woh sum ek threshold se upar gaya toh output 1, nahi gaya toh 0. Bas.

**Problem kya hai?**
Yeh sirf **linearly separable** problems solve kar sakta hai. Matlab woh problems jahan ek seedhi line do groups ko alag kar sake.

**Example – AND vs XOR**
AND gate ke points graph pe plot karo — ek seedhi line easily 0s aur 1s alag kar deti hai. ✅ Perceptron khush.

Ab XOR karo — koi bhi seedhi line kaam nahi karti. Chahe kitni bhi try karo, koi na koi point galat side pe rahega. ❌ Perceptron completely fail.

**Fix kya hai?**
Akele ek perceptron se kaam nahi chalta — toh bahut saare stack karo ek ke upar ek. Yeh banta hai **Multi-Layer Perceptron (MLP)** jo complex, non-linear patterns seekh sakta hai.

⚠️ Yeh wahi moment tha jab researchers ko pata chala ke single perceptron ki serious limitations hain — aur deep learning ki actual journey yahan se shuru hui.

---

## 🔹 Activation Functions

**Kyun zaroori hain?**
Agar network mein sirf multiplications aur additions hon — chahe 10 layers lagao ya 100 — mathematically sab milake ek hi badi linear layer ke barabar hain. Aur linear layer sirf seedhi lines seekh sakti hai — curves, circles, complex patterns bilkul nahi.

Activation functions woh cheez hain jo is line ko modte hain. Non-linearity add karte hain taake network koi bhi complex shape seekh sake.

---

### 🔸 Sigmoid

**Kyun use karte hain?**
Koi bhi number lo — chahe $+1000$ ho ya $-1000$ — Sigmoid use karke output hamesha **0 aur 1 ke beech** aata hai. Matlab probability ki tarah behave karta hai.

**Kahan use hota hai?**
- Binary classification ki output layer mein (maslan, yeh email spam hai ya nahi? 1 = spam, 0 = not spam)
- LSTM ke andar jo "gates" hote hain unmein — yeh decide karne ke liye ke kitni information keep karni hai ya forget karni hai

**Example – Barish kal hogi?**
Maan lo input features hain: humidity (80%), pressure (1005 hPa). Model kuch compute karta hai, phir Sigmoid deta hai 0.75. Matlab: "Kal barish hone ka 75% chance hai."

---

### 🔸 Tanh

**Kyun use karte hain?**
Output ko **-1 se 1 ke beech** rakhta hai, aur average output 0 ke around hota hai. Isse optimization better hoti hai — gradients ek hi direction mein push nahi hote, dono taraf balance rehta hai.

**Kahan use hota hai?**
- Purane RNNs mein hidden layers mein
- Jab positive aur negative values dono chahiyein (maslan autoencoders mein)

**Example – Image Scaling**
Ek photo mein pixels ki values 0 se 255 tak hoti hain. Model mein daalne se pehle unhe -1 se 1 mein convert karna ho toh Tanh use karo. Toh: 0 (dark) → -1, 255 (bright) → +1. Model ko yeh normalized values zyada easily process hoti hain aur training faster hoti hai.

---

### 🔸 ReLU — Rectified Linear Unit

**Kyun use karte hain?**
Simple aur fast hai: agar input positive hai toh wohi output de do, agar negative hai toh 0 kar do. Positive numbers ke liye vanishing gradient ka issue nahi hota.

$$\text{ReLU}(x) = \max(0, x)$$

**Kahan use hota hai?**
Har jagah! Hidden layers mein default choice hai — especially CNNs (image classification) aur MLPs mein.

**Example – Cat Detector**
Maan lo ek neuron "pointy ears" detect karta hai. Agar usay strong signal milay (+5) toh output 5 — full fire. Agar weak ya negative signal milay (-2) toh output 0 — shut off. Is "shutting off" se network efficient rehta hai — sirf relevant neurons activate hote hain.

⚠️ **Problem:** Agar neuron ko hamesha negative input milta rahe toh woh hamesha 0 output karega aur permanently seekhna band kar dega — ise **"Dead Neuron"** ya **Dying ReLU problem** kehte hain. Isliye Leaky ReLU aaya.

---

### 🔸 Leaky ReLU

**Kyun use karte hain?**
ReLU ki Dying ReLU problem fix karta hai. Negative values ke liye bilkul zero nahi karta — ek **bahut choti si slope (0.01)** rakhta hai. Matlab neuron completely dead nahi hota, thoda sa gradient rehta hai aur woh potentially recover kar sakta hai.

**Kahan use hota hai?**
- Hidden layers mein jab suspect ho ke neurons die ho rahe hain
- Deep **GANs** mein especially

**Example – Dead Neuron Recovery**
Maan lo neuron ko input aaya -5.
- ReLU deta: **0** → gradient zero → neuron permanently dead ❌
- Leaky ReLU deta: **-0.05** → thoda gradient hai → neuron future mein recover kar sakta hai ✅

---

### 🔸 GELU — Gaussian Error Linear Unit

**Kyun use karte hain?**
ReLU ka smooth version hai. Yeh inputs ko unki probability ke hisaab se weight karta hai ke woh positive hain ya nahi. ReLU ki tarah hard cutoff nahi hai — smooth transition hai. Transformers mein residual connections ke saath yeh ReLU se better kaam karta hai.

**Kahan use hota hai?**
Modern LLMs ke hidden layers mein — **BERT, GPT, ViT** sab GELU use karte hain.

**Example – Smooth Switch**
ReLU ek light switch ki tarah hai — ya on ya off. GELU ek dimmer switch ki tarah hai — smoothly transition karta hai.
- Input 2.0 → output roughly 2.0
- Input -2.0 → output close to 0, lekin exactly 0 nahi — thoda gradient preserve hota hai jo learning ke liye helpful hai

---

### 🔸 Swish

**Kyun use karte hain?**
Self-gated hai — formula hai $x \cdot \sigma(x)$. Smooth hai aur deep models mein, especially image tasks mein, ReLU se better perform kiya hai experimentally.

**Kahan use hota hai?**
Vision models jaise **EfficientNet** mein, aur kabhi kabhi transformer-based architectures mein bhi.

**Example – Small Negative Values**
ReLU -3 ko seedha 0 kar deta. Swish ke saath: sigmoid(-3) ≈ 0.05, toh output = 0.05 × (-3) ≈ **-0.15**. Woh ek chota negative signal rakhta hai — gradient flow better hota hai.

---

### ⚖️ Quick Comparison Table

| Function | Range | Best Use Case |
|---|---|---|
| Sigmoid | (0, 1) | Binary classification output, LSTM gates |
| Tanh | (-1, 1) | Purane RNNs, autoencoders |
| ReLU | [0, ∞) | CNNs, MLPs — default hidden layer choice |
| Leaky ReLU | (-∞, ∞) | Jab dead neurons ka risk ho, deep GANs |
| GELU | (-∞, ∞) | BERT, GPT, ViT — modern transformers |
| Swish | (-∞, ∞) | EfficientNet, deep vision models |

---

## 🔹 Vanishing & Exploding Gradients

**Pehle samjho gradient kya hota hai:**
Gradient ek signal hai jo batata hai — "is weight ko kitna aur kis direction mein change karo." Yeh signal backward jaata hai — output se input ki taraf. Isi se model seekhta hai.

---

### Vanishing Gradient

**Kya hota hai?**
Deep networks mein yeh gradient signal peeche jaate jaate itna chota hota jaata hai ke early layers tak pahunchte pahunchte almost zero ho jaata hai. Early layers ko pata hi nahi chalta ke unhe kya change karna chahiye — woh practically seekhna band kar deti hain.

**Kyun hota hai?**
Sigmoid aur Tanh ki slopes ≤ 0.25 hoti hain. Har layer se guzarte waqt gradient is fraction se multiply hota hai. 10 layers ke baad: $0.25^{10}$ ≈ almost nothing.

**Example – Telephone Game**
Maan lo 10 logon ki line hai. Pehle waale ne kaha "weights ko thoda badao." Har banda message thoda chota karke aage bhejta hai. 10th banda sunता hai "...kuch karo shayad?" — original message completely lost.

---

### Exploding Gradient

**Kya hota hai?**
Ulta problem. Gradient bahut bada ho jaata hai — model weights ko itna zyada update karta hai ke training completely fail ho jaati hai. Numbers NaN (Not a Number) ho jaate hain aur sab kuch crash karta hai.

---

### Fix kaise karein?

**Vanishing ke liye:**
- **ReLU ya GELU** use karo — positive inputs pe slope 1 rehti hai, gradient theek se flow karta hai
- **Skip connections** add karo jaise ResNet mein — gradient directly early layers tak jump kar sakta hai, beech ki layers bypass karke
- Achhi **weight initialization** use karo — He initialization (ReLU ke liye) ya Xavier initialization

**Exploding ke liye:**
- **Gradient Clipping** — agar gradient ek certain size se bada ho jaaye toh use scale karke limit kar do

**Example – Gradient Clipping**
Maan lo tum chal rahe ho aur achanak ek qadam 10 meter bada ho jaata hai — tum gir jaoge. Clipping us qadam ko safe size, maslan 1 meter, tak limit kar deta hai. Direction same rehti hai, bas step controlled rehti hai.

---

### ⚖️ ReLU vs GELU

| | ReLU | GELU |
|---|---|---|
| Speed | Faster | Thoda slower |
| Gradient | 0 pe hard jump | Har jagah smooth |
| Best For | CNNs, older models | Transformers — GPT, BERT |

---

## 🔹 Feedforward Networks — MLP

**Kya hota hai?**
Sabse basic neural network architecture. Data ek direction mein flow karta hai — input se output tak. Koi loop nahi, koi backward connection nahi.

Structure:
**Input Layer → Hidden Layer(s) → Output Layer**

Har hidden layer yeh karta hai:
1. Pichli layer ka output leta hai
2. Weights se multiply karta hai
3. Bias add karta hai
4. Activation function apply karta hai
5. Result aglee layer ko deta hai

**Example – House Price Prediction**
- Inputs: ghar ki size, bedrooms ki tadaad, age of house
- Hidden layer combinations seekhti hai — maslan "size aur bedrooms ka combined effect kya hai?"
- Output: predicted price

**Universal Approximation Theorem:**
Ek single hidden layer wala network — agar neurons kaafi hon — theoretically koi bhi continuous function approximate kar sakta hai. Lekin "theoretically" important word hai. Practically, deeper networks zyada efficient hote hain aur train karna zyada aasan hota hai.

---

### Width vs Depth

**Width** — ek layer mein kitne neurons hain:
Zyada neurons = zyada capacity. Lekin zyada parameters bhi = zyada computation aur overfitting ka risk.

**Depth** — kitni layers hain:
Deeper network step-by-step features seekhta hai.

**Example – Face Recognition**
- Layer 1: Edges detect karta hai
- Layer 2: Shapes detect karta hai (aankhein, naak)
- Layer 3: Poora face detect karta hai

Ek wide lekin shallow network ko yahi karne ke liye astronomically zyada neurons chahiye honge. Depth same kaam kam parameters mein karta hai — aur generally better generalize bhi karta hai.

---

## 🔹 Loss Functions

Loss function batata hai — "model ki prediction kitni galat hai?" Training mein hum is loss ko minimize karte hain.

---

### MSE — Regression ke liye

$$\mathcal{L}_{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Kyun use karte hain?**
Predicted aur actual values ke beech ka average squared difference measure karta hai. Squaring kyun? Kyunki bade errors ko bahut zyada costly banata hai — model unhe fix karne ki zyada koshish karta hai. Aur negative aur positive errors cancel nahi hote.

**Kahan use hota hai?**
Jab output ek continuous number ho — temperature predict karna, ghar ki qeemat, stock price wagera.

**Example – Temperature Prediction**
Tumne predict kiya 25°C, actual tha 30°C.
- Error = 5, Squared error = **25**
- Agar 10 degree off hote: Squared error = **100**

Model badi mistakes ko 4x seriously leta hai — isliye unhe fix karna priority hoti hai.

---

### Cross-Entropy Loss — Classification ke liye

$$\mathcal{L}_{CE} = -\sum_{c} y_c \log(\hat{p}_c)$$

Binary ke liye:
$$\mathcal{L} = -[y \log \hat{p} + (1-y)\log(1-\hat{p})]$$

**Kyun use karte hain?**
Measure karta hai ke predicted probability true class se kitni alag hai. Model jitna zyada confident ho aur galat bhi ho — loss utna zyada bada hoga. Yeh model ko "confidently wrong" hone pe bahut badi penalty deta hai.

**Kahan use hota hai?**
Classification problems mein — binary ya multi-class dono.

**Example – Spam Detection**
True label: spam (y=1).
- Model predicts 0.9 (spam) → Loss = $-\log(0.9)$ ≈ **0.105** ✅ Achha prediction, chota loss
- Model predicts 0.1 (spam) → Loss = $-\log(0.1)$ = **2.3** ❌ Bahut galat, bahut bada loss — model yahan se bahut seekhega

---

### 💡 MSE vs Cross-Entropy — Classification mein MSE kyun nahi?

MSE ke saath ek bada masla hai: agar model confidently galat ho — maslan 0 predict kare jab true label 1 ho — gradient bahut chota ho jaata hai aur learning almost ruk jaati hai. Ise **saturation problem** kehte hain.

Cross-Entropy is situation mein bahut bada gradient deta hai — model jaldi correct karta hai. Aur yeh softmax outputs ke saath naturally kaam karta hai jo multi-class classification mein use hota hai.

---

## 🔹 Optimization & Gradient Descent

Training ka goal hai loss minimize karna — yani woh point dhundna jahan model sabse kam galat ho.

---

### Gradient Descent ke Types

| Variant | Kaise Kaam Karta Hai | Batch Size | Pros / Cons |
|---|---|---|---|
| **Batch GD** | Poora data dekh ke ek update karta hai | Sab data | Stable lekin bahut slow, huge memory chahiye |
| **Stochastic GD** | Har ek sample ke baad update | 1 | Fast lekin bahut jumpy — zigzag karta hai |
| **Mini-batch GD** | Ek chota random batch use karta hai | 32–512 | Sabse common — speed aur stability ka balance |

**Example – Valley mein lowest point dhundna**
Aankhein band hain, tumhe lowest point dhundna hai sirf qadam le ke feel karna hai.
- **Batch GD:** Pehle poori valley ka complete map banao, phir ek qadam lo. Accurate lekin impossibly slow.
- **Stochastic GD:** Har second ek random direction mein qadam lo. Fast lekin itna erratic ke kabhi kabhi upar bhi chale jaate ho.
- **Mini-batch GD:** Apne aas paas ka chota area feel karo, phir qadam lo. Dono ka balance — yeh practical choice hai.

---

### Backpropagation & Chain Rule

**Forward Pass:**
Input layers se guzarta hai → prediction milti hai → loss calculate hota hai.

**Backward Pass:**
Loss se peeche ki taraf jaate hain. Calculus ki **chain rule** use karke calculate karte hain ke har weight ne loss mein kitna contribute kiya — phir weights adjust karte hain taake loss kam ho.

**Chain Rule kyun chahiye?**
Network mein layers pe layers hain, har layer ki calculation doosri pe depend karti hai. Chain rule se hum is layered dependency ko peeche ki taraf unravel kar sakte hain — ek ek layer karke.

**Example – Weight Adjustment**
Maan lo ek weight bahut zyada tha aur usne bada error cause kiya. Backprop batata hai: "yeh weight 0.3 se kam karo." Agle iteration mein loss thoda kam hoga. Yeh process thousands of times repeat hota hai jab tak model achha na ho jaaye.

---

### Optimizers

---

### 🔸 SGD with Momentum

**Kyun use karte hain?**
Simple SGD mein sirf current gradient use hota hai — har step independent hai. Momentum ke saath hum ek **velocity** maintain karte hain jo pichle gradients ka running average hota hai. Yeh updates smooth karta hai aur chote bumps se nikalne mein help karta hai.

$$v_t = \beta v_{t-1} + (1-\beta)\nabla_w \mathcal{L}$$
$$w \leftarrow w - \eta v_t$$

$\beta$ usually 0.9 hota hai — matlab 90% previous velocity, 10% current gradient.

**Example – Ball Rolling Downhill**
Ek ball pahad se neeche roll ho rahi hai. Woh har chote patthar pe nahi rukti — momentum se aagey nikalti rehti hai. SGD with Momentum bilkul aise — local minima se niklne mein help karta hai jo plain SGD mein problem tha.

---

### 🔸 Adam — Adaptive Moment Estimation

**Kyun use karte hain?**
Adam do cheezein simultaneously track karta hai:
1. **Gradients ka average** (first moment) — momentum ki tarah
2. **Squared gradients ka average** (second moment) — har weight ki learning rate adapt karne ke liye

Iska result: **har weight ki apni alag learning rate** hoti hai jo automatically adjust hoti rehti hai. Jo weights bahut update ho rahe hain unki learning rate automatically slow ho jaati hai. Jo weights kam update ho rahe hain unki fast rehti hai.

**Kahan use hota hai?**
Sparse data wali problems mein especially achha kaam karta hai — maslan NLP mein jahan bahut se words rarely aate hain.

---

### 🔸 AdamW — Improved Adam

**Kyun Adam se better hai?**
Adam mein ek hidden bug tha — **weight decay** (jo overfitting rokti hai) adaptive learning rates ke saath mix ho jaata tha aur properly kaam nahi karta tha.

AdamW ne yeh fix kiya: weight decay ko gradient update se **decouple** kar diya — ab weight decay directly weights pe apply hoti hai, learning rate se bilkul alag.

**Kahan use hota hai?**
**GPT, Llama** jaise large language models train karne ka ab yeh **default choice** hai.

**Example – Weight Decay ka Fark**
Adam mein weight decay aur learning rate ek saath mix hote the — jaise ek wire mein do alag signals mix ho jaayein, dono corrupt ho jaate hain. AdamW ne inhe alag wires de diye — dono theek se kaam karte hain.

---

### ⚖️ SGD vs Adam

| | SGD + Momentum | Adam/AdamW |
|---|---|---|
| Generalization | Aksar better — flatter minima dhundta hai | Sharper minima, overfitting ka thoda risk |
| Convergence Speed | Slower | Bahut faster |
| Learning Rate Sensitivity | Bahut sensitive — tuning mushkil | Zyada robust |
| Best For | CNNs, vision tasks | NLP, Transformers |

**Flatter minima kyun better hai?**
Flat minima pe agar model thoda shift bhi ho jaaye — naya data aaye — toh loss zyada nahi badhtaa. Model stable rehta hai. Sharp minima pe thodi si shift pe loss bahut bada ho jaata hai — model fragile hota hai aur new data pe fail karta hai.

---

## 🔹 Regularization — Overfitting Se Kaise Bachein

---

### Bias-Variance Tradeoff

**High Bias — Underfitting:**
Model bahut simple hai. Patterns pakad nahi pa raha. Jaise ek curvy dataset pe seedhi line fit karna chaaho — woh kabhi theek se fit nahi hogi.

**High Variance — Overfitting:**
Model bahut complex hai. Training data memorize kar leta hai — har noise, har outlier. Naye data pe completely fail ho jaata hai.

**Example – Cat Recognition**
- **High Bias:** Model sirf color use karta hai. Kale cats galat classify hote hain kyunki model ne enough features nahi seekhe.
- **High Variance:** Model ne training photos ke har pixel memorize kar liye. Thoda alag angle se cat aaye — model pehchaan nahi sakta. Usne patterns seekhe hi nahi, sirf images ratta maar li.

Goal hai in dono ke beech balance — **regularization** yahi achieve karne mein help karta hai.

---

### L1 & L2 Regularization

Dono loss function mein ek **penalty** add karte hain taake weights control mein rahein.

**L2 — Weight Decay:**
$$Loss = original\_loss + \lambda \sum w_i^2$$
Large weights pe penalty lagaata hai. Saare weights chote karne par majboor karta hai lekin kisi ko exactly zero nahi karta.

**L1 — Lasso:**
$$Loss = original\_loss + \lambda \sum |w_i|$$
Kuch weights ko **exactly zero** kar sakta hai — jo effectively un features ko model se remove kar deta hai. Feature selection ke liye useful.

**$\lambda$ kya hai?**
Ek hyperparameter — control karta hai ke regularization kitni strong ho. Zyada $\lambda$ = zyada penalty = chote weights.

**Example – House Price Prediction**
Model ke paas ek feature hai "number of windows." L1 is feature ka weight zero kar sakta hai agar yeh price ke liye actually important nahi — woh feature effectively remove ho jaata hai. Model simpler aur more interpretable ban jaata hai.

---

### Dropout

**Kya hota hai?**
Training ke doran, randomly kuch neurons off kar do — unka output zero set kar do — ek probability $p$ ke saath (usually 0.1 se 0.5).

**Kyun kaam karta hai?**
Network kisi ek neuron pe zyada depend karna band kar deta hai. Har neuron ko independently useful cheez seekhni padti hai — redundant representations banti hain.

**Test time pe kya hota hai?**
Saare neurons on rehte hain, lekin outputs ko $(1-p)$ se scale kar diya jaata hai taake average same rahe training jaisa.

**Example – Group Project**
Training ke doran har din randomly kuch team members absent hote hain. Puri team seekhti hai ke kisi ek pe depend na karein — sabko sab kuch thoda thoda aana chahiye. Final presentation pe — test time — saare present hain aur team bahut better perform karti hai kyunki koi single point of failure nahi tha.

⚠️ **Dropout ek saath kai chote networks train karta hai** — har alag dropout mask ek alag architecture hai. Test time pe yeh sab effectively average ho jaate hain. Ensemble methods aksar better generalize karte hain — yahi dropout ka secret hai.

---

### Normalization Techniques

Normalization ka matlab hai values ko ek controlled range mein rakhna taake training smooth rahe aur gradients healthy rahein.

| Method | Kya Normalize Karta Hai | Kahan Use Hota Hai |
|---|---|---|
| **Batch Norm** | Batch ke across — saare samples ke liye har feature | CNNs mein, large batch size chahiye |
| **Layer Norm** | Har sample ke andar — saare features ek sample ke liye | Transformers, RNNs — koi bhi batch size |
| **RMS Norm** | Layer Norm jaisa lekin mean subtract nahi karta | Modern LLMs — Llama, Mistral — faster |

**Normalize kyun karein?**
- Values healthy range mein rehti hain — vanishing ya exploding gradients nahi hote
- Higher learning rates use kar sakte ho — training faster hoti hai
- Thodi si regularization bhi automatically add hoti hai

---

### 🔸 Batch Norm

**Kaise kaam karta hai?**
Ek batch ke saare samples dekhta hai, har feature ka mean aur variance nikalta hai, aur normalize karta hai. Phir learnable parameters se scale aur shift karta hai.

**Problem kahan hai?**
Sequences ke saath kaam nahi karta — lengths vary karti hain, padding statistics corrupt karta hai. Chote batch sizes pe bhi unreliable ho jaata hai.

---

### 🔸 Layer Norm

**Kaise kaam karta hai?**
$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

Har ek sample ke andar saare features ka mean aur variance nikalta hai — batch se koi lena dena nahi. Isliye koi bhi batch size ho, koi bhi sequence length ho — perfectly kaam karta hai.

**Example – Why Transformers use Layer Norm:**
Ek sentence mein 5 words hain, doosre mein 50. Batch Norm dono ko ek saath normalize karne ki koshish karta — statistics mess ho jaati. Layer Norm har sentence ko independently normalize karta hai — koi masla nahi.

---

### 🔸 RMS Norm

**Kaise kaam karta hai?**
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

Layer Norm jaisa hai lekin mean subtract karna skip kar deta hai — sirf root mean square se divide karta hai. Computation save hoti hai aur modern LLMs mein equally effective hai.

**Kahan use hota hai?**
**Llama, Mistral** — yeh models Layer Norm ki jagah RMS Norm use karte hain speed ke liye.

---

## 🗺️ Phase 1 — Complete Summary

```
Perceptron — basic linear classifier
     ↓ Problem: XOR jaisi non-linear problems fail
     ↓
MLP — neurons stack karo, complexity badhao
     ↓ Problem: sirf linear ops = ek badi linear layer
     ↓
Activation Functions — non-linearity add karo
ReLU, GELU — vanishing gradient bhi fix karte hain
     ↓
Loss Functions:
  → MSE: Regression ke liye
  → Cross-Entropy: Classification ke liye
     ↓
Backpropagation — chain rule se gradients nikalo
     ↓
Gradient Descent — weights update karo
  → Batch / Stochastic / Mini-batch
     ↓
Optimizers:
  SGD → SGD + Momentum → Adam → AdamW
     ↓
Regularization — overfitting se bacho:
  → Dropout
  → L1 / L2
  → BatchNorm / LayerNorm / RMSNorm
```

✅ **Phase 1 Complete —


# 🧠 Deep Learning Interview Revision Notes
## Phase 2: Specialized Architectures (Before Transformers)

---

## 🔹 Convolutional Neural Networks (CNNs)

**CNNs kyun chahiye images ke liye?**
Agar tum ek normal MLP image pe lagao, toh woh har pixel ko alag alag dekhta hai — usse pata nahi ke paas paas ke pixels ka koi relation hai. Maslan, cat ki aankhon ke pixels ek saath hote hain — yeh proximity matter karti hai. MLP yeh nahi samajhta.

CNNs teen important ideas pe kaam karte hain:
- **Local connectivity** — paas ke pixels zyada connected hote hain door walon se. Network pehle chote regions dekhta hai.
- **Translation invariance** — cat top-left corner mein ho ya bottom-right mein, cat hi hai. Network usse kahi bhi pehchaan sakta hai.
- **Hierarchical features** — simple cheezein pehle seekhta hai (edges), phir complex (aankhein, fur), phir poora cat.

---

## 🔹 Convolution Operation

**Kaise kaam karta hai?**
Ek chota sa pattern detector hota hai jise **filter** ya **kernel** kehte hain — maslan 3x3 pixels ka window. Yeh window poori image pe slide karta hai. Har position pe pixels ki values ko filter ke numbers se multiply karo aur add karo — ek naya value milta hai. Yeh process poori image pe karo toh ek **feature map** milta hai jo batata hai ke woh pattern kahan kahan hai.

**Example – Horizontal Edge Detector**
Maan lo tumhara filter horizontal edges dhundne ke liye design kiya gaya hai. Jahan bhi image mein horizontal line hai, wahan feature map mein high value aayegi. Slide karo poori image pe — tumhare paas ek complete map hai ke horizontal edges kahan hain.

Hum ek saath kaafi filters train karte hain — maslan 64 filters. Har filter alag pattern dhundta hai: edges, corners, textures. Sab milake 64 feature maps bante hain.

---

**Weight Sharing kya hota hai aur kyun important hai?**
Wahi ek filter — same numbers — poori image pe slide karta hai. Matlab ek hi pattern detector har jagah use hota hai.

Normal layer mein har pixel pair ke liye alag weight chahiye — millions of parameters. Ek 3x3 filter mein sirf **9 numbers** hain. Yeh massive saving hai.

**Interview Question — Weight sharing kyun help karta hai?**
Kyunki ek filter ek pattern seekhta hai — maslan vertical line — aur woh pattern image mein kahi bhi dhoondh sakta hai. Network pattern care karta hai, position nahi. Yahi **translation invariance** hai.

---

## 🔹 Convolution Settings — Important Terms

| Term | Kya Hota Hai | Example |
|---|---|---|
| **Kernel Size** | Window kitna bada hai | 3x3 chote details, 5x5 broader patterns |
| **Stride** | Har step mein window kitne pixels move karta hai | Stride 2 = output chota hoga |
| **Padding** | Edges pe fake pixels add karna taake corners cover hon | "same" padding = output same size rehta hai |
| **Number of Filters** | Ek saath kitne patterns dhundh rahe ho | 64 filters = 64 feature maps |

**Output size ka formula:**
$$\text{Output width} = \left\lfloor \frac{W - k + 2P}{s} \right\rfloor + 1$$

**Example — Size Calculate Karo**
Input 32x32, kernel 3x3, stride 1, padding 1 → output **32x32**. Padding ne size same rakha.

---

## 🔹 Pooling — Feature Maps Shrink Karna

Convolution ke baad feature maps shrink karte hain — data kam karo aur network ko small shifts ke liye robust banao.

| Type | Kaise Kaam Karta Hai | Example |
|---|---|---|
| **Max Pooling** | 2x2 window mein jo sabse badi value ho usse rakh lo | [1, 5, 2, 3] → **5**. Strongest signal survive karta hai |
| **Average Pooling** | 2x2 window ki average lo | [1, 5, 2, 3] → **2.75**. Smoother result |
| **Global Average Pooling** | Poori feature map ki ek average value nikalo | 7x7 map ke 49 values → sirf **1 number** per channel |

**Interview Question — Global Average Pooling kyun use karte hain?**
Poori feature map — maslan "eye detector" map — ko ek number mein compress karta hai: "kitna 'eye' present hai?" Parameters drastically kam ho jaate hain, overfitting kam hoti hai. Aur har channel directly ek concept se correspond karta hai — model interpretable hota hai.

---

## 🔹 Receptive Field

**Kya hota hai?**
Ek neuron original image ka kitna area dekh sakta hai — yeh uska **receptive field** hai.

- Layer 1 ka neuron: sirf 3x3 patch dekhta hai
- Layer 2 ka neuron: Layer 1 ke patches combine karta hai — original image ka zyada bada area dikhta hai
- Jitna deep jaao, utna bada area dikhai deta hai

**Example — Layer by Layer**
Layer 1: 3x3 filter → receptive field **3x3**
Layer 2: phir 3x3 filter → receptive field **5x5** original image ka

**Dilated (Atrous) Convolution kya hai?**
Adjacent pixels ki jagah kuch skip karo — receptive field bahut tezi se barhta hai bina extra parameters ke. Image segmentation mein useful hai jahan wide view chahiye hoti hai.

---

## 🔹 Important CNN Architectures

| Model | Key Idea | Kyun Important |
|---|---|---|
| **LeNet-5** | Pehla real CNN — handwritten digits ke liye | Sab kuch yahan se shuru hua |
| **AlexNet** | Deep CNN on GPU, ReLU aur Dropout use kiya | ImageNet 2012 jeeta — deep learning boom shuru |
| **VGGNet** | Sirf 3x3 filters, bahut deep | Dikhaaya ke chote filters stack karna kaam karta hai |
| **GoogLeNet/Inception** | Alag alag size ke filters parallel mein | Efficient — 1x1 conv se channels reduce kiye |
| **ResNet** | Skip connections add kiye | 100+ layers ke very deep networks train ho sake |
| **EfficientNet** | Depth, width, resolution sab saath scale kiye | Kam parameters mein top performance |

---

## 🔹 ResNet — Skip Connections

**Kyun special hai?**
ResNet mein har block yeh compute karta hai:

$$\text{output} = \text{block}(\text{input}) + \text{input}$$

Block sirf **change** (residual) seekhta hai — poori cheez nahi. Agar block kuch nahi karta, output = input. Koi damage nahi.

Yeh "degradation problem" solve karta hai — jahan aur layers add karne se performance actually worse hoti thi.

**Interview Question — Skip connections kyun help karte hain?**
Bina skip connections ke, deep networks "identity mapping" (output = input) seekhna struggle karte hain kyunki layers non-linear hain. Skip connection ek shortcut deta hai — gradient directly addition ke through peeche flow kar sakta hai. Early layers ko strong signals milte hain. Isliye bahut deep networks train karna possible hua.

---

## 🔹 Recurrent Neural Networks (RNNs)

**CNNs kyun nahi chahiye sequences ke liye?**
CNNs fixed-size grids pe kaam karte hain — images theek hain. Lekin sentences, audio, stock prices — yeh sequences hain, varying length ki, aur **order matter karta hai**. "Dog bites man" aur "Man bites dog" mein same words hain — lekin meaning bilkul alag.

RNNs sequences ko step by step process karte hain aur ek **hidden state** maintain karte hain — yeh memory hai jo pichle steps ki information carry karti hai.

---

**RNN kaise kaam karta hai?**
Har time step $t$ pe:
- Input $x_t$ aata hai — maslan ek word
- Pichla hidden state $h_{t-1}$ already hai — memory
- Naya hidden state banta hai:

$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$$

- Output $y_t$ — maslan next word ki prediction

Aur important baat: same weights ($W_{hh}, W_{xh}, b$) har step pe use hote hain — time ke across weight sharing.

**Example — Next Word Prediction**
Sentence: "I am ___"
- t=1: input "I" → hidden state "I" encode karta hai
- t=2: input "am" → hidden state "I am" combine karta hai
- t=3: hidden state se next word predict karo → "hungry" ya "tired"

---

## 🔹 Vanishing Gradients in RNNs

**Kya problem hoti hai?**
Training mein error ko time ke through peeche bhejna padta hai — ise **Backpropagation Through Time (BPTT)** kehte hain. Har step pe gradient $W_{hh}$ se multiply hota hai.

- Agar $W_{hh}$ ki "strength" 1 se kam hai → gradient har step pe shrink hota hai → 50 steps baad almost zero
- Agar 1 se zyada hai → gradient explode karta hai → training unstable

**Interview Question — Vanilla RNN lamba yaad kyun nahi rakh sakta?**
Kyunki hidden state baar baar same weight matrix se multiply hota hai. Socho 0.9 ko baar baar multiply karo: $0.9^{50}$ ≈ **0.005** — almost kuch nahi. Early words ka signal completely gayab ho jaata hai.

**Fix:**
- Exploding ke liye: **Gradient Clipping**
- Vanishing ke liye: **LSTM ya GRU**

---

## 🔹 LSTM — Long Short-Term Memory

**Badi idea kya hai?**
LSTM ek **cell state** $C_t$ introduce karta hai — yeh ek highway ki tarah hai jo time ke through flow karta hai. Isme sirf simple addition hoti hai — heavy matrix multiplication nahi. Gradient ke liye clear path milta hai.

**Gates kya hote hain?**
Valves ki tarah — control karte hain ke kitni information pass ho. Sigmoid use karte hain — output 0 se 1 ke beech. 0 = block, 1 = poora through.

---

**LSTM ke andar 4 steps:**

**Step 1 — Forget Gate $f_t$**
Decide karta hai ke purani cell state mein se kya bhulaana hai. $h_{t-1}$ aur $x_t$ dekhta hai, 0–1 output karta hai.

Maslan: topic change ho gaya sentence mein — purana subject bhool jaao.

**Step 2 — Input Gate $i_t$**
Decide karta hai ke kaunsi nayi information cell state mein store karni hai. Wahi $h_{t-1}$ aur $x_t$ dekhta hai.

**Step 3 — Candidate Cell $\tilde{C}_t$**
Nayi values suggest karta hai jo add ho sakti hain. Tanh use karta hai — output -1 se 1 ke beech.

**Step 4 — Cell State Update**
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

Pehle purani cell state ko forget gate se multiply karo — jo bhulaana tha woh gaya. Phir nayi candidate values add karo input gate se scale karke.

**Step 5 — Output Gate $o_t$**
Decide karta hai ke cell state ka kaunsa hissa hidden state ke roop mein output ho.
$$h_t = o_t \odot \tanh(C_t)$$

---

**Interview Question — LSTM vanishing gradient kaise solve karta hai?**
Cell state addition se update hota hai — matrix multiplication se nahi. Jab backpropagate karte hain, gradient addition ke through seedha flow karta hai. Forget gate sirf element-wise multiply karta hai — yeh repeated matrix multiplication se bahut safer hai. Forget gate ko seekhna hai ke gradients ko alive kaise rakha jaaye — aur woh seekh jaata hai.

---

## 🔹 GRU — Gated Recurrent Unit

**LSTM ka simpler cousin.**
LSTM ke forget aur input gates ko ek **update gate** mein combine kar diya. Cell state aur hidden state bhi merge kar diye. Sirf do gates hain:

- **Update Gate $z_t$** — kitna purana hidden state rakhna hai
- **Reset Gate $r_t$** — nayi candidate compute karte waqt purana hidden state kitna forget karna hai

Kam parameters = faster training, aur aksar LSTM jaisi hi performance.

---

**Interview Question — LSTM ya GRU — kaunsa choose karein?**

| | LSTM | GRU |
|---|---|---|
| Parameters | Zyada (4 gates) | Kam (3 gates) |
| Speed | Slower | Faster |
| Long Sequences | Thoda better ho sakta hai | Aksar same hi |
| Use Case | Maximum memory control chahiye | Efficiency chahiye, simpler model |

---

## 🔹 RNNs ka Bada Problem — Sequential Processing

**Yeh section crucial hai — yahan se samjhoge ke Transformers kyun bane.**

RNNs mein do major flaws hain:

**Flaw 1 — Sequential hai:**
$h_t$ compute karne ke liye $h_{t-1}$ finish hona zaroori hai. Sequence ko parallelize nahi kar sakte. Modern GPUs parallel computation ke liye bane hain — RNN unka fayda nahi utha sakta. Long sequences pe training painfully slow hoti hai.

**Flaw 2 — Information Bottleneck:**
Translation jaisi tasks mein poora input sentence ek akele hidden state $h_T$ mein squeeze karna padta hai — phir output generate hota hai. Long sentences mein early words dilute ho jaate hain aur practically khum jaate hain — chahe LSTM hi kyun na ho.

**Example — Long Sentence Translation**
"The cat that was sitting on the mat near the window in the old house ate the mouse."

Tak tak tak — yeh poora sentence ek vector mein compress karna hai. "cat" ka information tab tak almost gone ho chuka hoga jab tak "ate" tak pahuncho. Translation galat hogi.

**Interview Question — Transformers ne kaunsa fundamental bottleneck solve kiya?**
Do cheezein. Pehli: RNNs ek word at a time process karte hain — no parallelism, slow. Doosri: poori sequence ek fixed-size vector mein compress hoti hai — information lose hoti hai. Transformers mein har word directly har doosre word ko dekh sakta hai ek hi step mein, aur sab words parallel process hote hain. Bottleneck khatam.

---

## ⚖️ CNN vs RNN vs Transformer — Quick Comparison

| Property | CNN | RNN / LSTM | Transformer |
|---|---|---|---|
| **Best For** | Images, grids | Sequences — speech, text | Sequences aur practically kuch bhi |
| **Parallelism** | High — sab pixels ek saath | Zero — step by step | Full — sab positions ek saath |
| **Long-range Dependencies** | Receptive field se limited | RNN poor, LSTM theek | Excellent — direct connections |
| **Memory** | Fixed-size kernels | Fixed-size hidden state | $O(n^2)$ attention matrix — bada |
| **Built-in Bias** | Locality, translation invariance | Temporal order | Koi nahi — positional info khud seekhna padta hai |

---

## 🗺️ Phase 2 — Complete Summary

```
Images → CNN
  ├── Convolution — filters slide karte hain, weights share hote hain
  ├── Pooling — shrink karo, important info rakho
  ├── Receptive Field — deep layers zyada dekhte hain
  └── ResNet — skip connections → bahut deep networks possible

Sequences → RNN
  ├── Hidden State — step by step memory
  ├── Vanishing Gradients — BPTT long memory kill karta hai
  ├── LSTM — cell state + gates → long memory solve
  ├── GRU — simpler, faster LSTM
  └── Sequential Bottleneck → isliye Transformers bane
```

✅ **Phase 2 Complete**

# 🧠 Deep Learning Interview Revision Notes
## Phase 3: The Attention Revolution & Transformers

---

## 🔹 Seq2Seq — Pehle Translation Kaise Hoti Thi?

Transformers se pehle, translation ke liye ek two-part system use hota tha:

- **Encoder** — ek RNN jo input sentence word by word parhta tha. Jab poora sentence parh leta, toh ek single vector produce karta tha — ek list of numbers jo supposedly **poore sentence ka meaning** capture karta tha.
- **Decoder** — doosra RNN jo woh single vector leta aur output sentence word by word generate karta tha.

**Example:**
```
"The cat sat on the mat"
→ [ENCODER RNN]
→ ek vector (maslan 512 numbers)
→ [DECODER RNN]
→ "Le chat s'est assis sur le tapis"
```

---

**Badi problem kya thi?**
Socho tumhe ek poori book yaad karni hai — lekin sirf **ek sentence** likh sakte ho sab capture karne ke liye. Bahut kuch choot jaayega na?

Exactly yahi ho raha tha:
- Poora input sentence — chahe kitna bhi lamba ho — ek **fixed-size vector** mein squeeze hota tha
- Lambe sentences (50+ words) mein, encoder jab tak end tak pahunchta, beginning ke words "forget" ho chuke hote the
- Research ne prove kiya ke sentence lamba hone pe translation quality significantly worse hoti thi

**Analogy:** Poori movie ko ek tweet mein summarize karo. End tak pahunchte pahunchte beginning bhool chuke hoge.

---

## 🔹 Attention Mechanism — Bahdanau, 2015

**Brilliant idea kya tha?**
Ek vector ki jagah — decoder **saare encoder hidden states** ko dekh sake aur decide kare ke har step pe kaunsa word most important hai?

Yahi attention hai.

**Real Life Se Samjho:**
Jab tum koi sentence translate karte ho, tum naturally alag alag parts pe focus karte ho. "The cat sat" translate karte waqt jab "chat" (cat) likhne wala ho — tum mostly "cat" word dekh rahe ho original mein, "the" ya "sat" nahi.

Attention exactly yahi karta hai — har output word ke liye, saare input words mein se decide karta hai ke kiski taraf kitna "dhyan" dena hai.

---

**Mathematically kaise kaam karta hai?**

**Step 1 — Score:**
Har decoder step pe — jab koi word output hone wala ho — calculate karo ke har encoder word kitna relevant hai abhi ke liye.
Matlab: "Word #1 pe abhi kitna focus karun? Word #2 pe?"

**Step 2 — Normalize:**
In scores ko percentages mein convert karo jo 100% add up hon — softmax use karo.
Ab attention weights hain: maslan word #1 ko 10% focus, word #2 ko 70%, word #3 ko 20%.

**Step 3 — Aggregate:**
In percentages se saare encoder hidden states ka weighted sum lo.
Result: ek **context vector** — is specific step ke liye ek custom summary jo sirf relevant information highlight karta hai.

---

**Interview Question — Attention ne bottleneck kaise solve kiya?**
Sab kuch ek vector mein cramp karne ki jagah, decoder seedha kisi bhi encoder word se information grab kar sakta hai — kabhi bhi. Iska matlab:
- Lambe sentences forget nahi hote — early words still accessible hain
- Har output word ko apna custom "summary" milta hai jo sirf uske liye relevant hai
- Tum literally dekh sakte ho ke model ne kaunse input words pe focus kiya — interpretable hai

---

## 🔹 Scaled Dot-Product Attention — Transformer ka Engine

**Library Analogy se samjho — Query, Key, Value:**

Socho tum library mein ho books dhundhne gaye:
- **Query (Q)** — tum kya dhoondh rahe ho. "Mujhe machine learning ki books chahiye."
- **Key (K)** — har book kya advertise karti hai. Book A: "Python programming", Book B: "Deep learning basics", Book C: "Cooking recipes."
- **Value (V)** — book ka actual content.

Librarian tumhara query dekhta hai, har book ki key se match karta hai, aur jo best match ho uski content laata hai — lekin blend karke. Mostly Book B (deep learning) milti hai, thodi Book A (Python) bhi agar relevant ho.

Transformer mein:
- **Q** — abhi kya dhoondh raha hoon? (current decoder state)
- **K** — har input position kya offer karta hai? (har word ka "advertisement")
- **V** — us position ki actual information (word ka meaning)

---

**Formula:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Step by step:

1. **$QK^T$** — query ko keys se multiply karo. Score matrix milta hai — bada number = achha match.
2. **$\sqrt{d_k}$ se divide** — scores scale down karo (neeche explain hai).
3. **Softmax** — scores ko percentages mein convert karo (attention weights).
4. **V se multiply** — in percentages se values blend karo.

---

**Interview Question — $\sqrt{d_k}$ se divide kyun karte hain?**

Maan lo random numbers hain. Jab do random vectors of length $d_k$ multiply karte ho, sum barhta jaata hai jaise $d_k$ barhta hai.
- $d_k = 64$ → dot product around **8**
- $d_k = 512$ → dot product around **22**

Ab yeh bade numbers (22) softmax mein daalo — woh bahut extreme ho jaata hai. Almost 100% weight sirf largest score pe, baaki sab pe nearly 0%. Yeh bahut confident lekin brittle decision hai. Gradients tiny ho jaate hain kyunki softmax saturate ho jaati hai.

$\sqrt{d_k}$ se divide karo — numbers normal range pe aate hain (variance ~1), softmax smooth rehti hai, gradients theek se flow karte hain.

**Analogy:** Ek test 100 marks ka hai, doosra 1000 marks ka. Fairly compare karne ke liye scale karna padega. $\sqrt{d_k}$ wahi scaling factor hai.

---

## 🔹 Multi-Head Attention (MHA)

**Multiple attention heads kyun chahiye?**

Yeh sentence analyze karo: *"The dog chased the cat because it was hungry."*

"It" kya refer karta hai — dog ya cat? Is ek sentence mein kaafi alag alag relationships hain:
- Subject-verb: "dog" → "chased"
- Pronoun reference: "it" → ? (resolve karna hai)
- Adjective-noun: "hungry" → kaun?

Ek single attention mechanism ko yeh saari relationships ek saath capture karni hon — that's too much for one lens.

**Multi-head attention multiple "lenses" deta hai:**
- Head 1: syntactic relationships (subject-verb)
- Head 2: coreference (pronouns aur unke nouns)
- Head 3: local context (paas ke words)
- Head 4: long-range semantic links

---

**Kaise kaam karta hai?**
Input ko **h different sets** of Q, K, V mein project karo — alag alag learned projections se. Har set (head) independently attention compute karta hai. Phir sab results concatenate karo aur original dimension pe project karo.

**Analogy:** Multiple experts ek hi sentence analyze kar rahe hain — har ek apni specialty ke saath. Phir sab ki opinions combine karo.

---

**Interview Question — Kya heads actually alag cheezein seekhte hain?**
Haan! Research ne show kiya hai BERT aur GPT mein:
- Kuch heads syntactic relationships mein specialize karte hain (subjects aur verbs)
- Kuch positional relationships pe (paas ke words)
- Kuch coreference track karte hain (pronouns → nouns)
- Kuch semantic similarity capture karte hain

Yeh specialization training se naturally emerge hoti hai — model khud figure out karta hai ke alag patterns ke liye alag heads useful hain. Bina multiple heads ke, ek attention "jack of all trades, master of none" hota.

---

## 🔹 Full Transformer Architecture — Ek Block Ke Andar Kya Hota Hai?

**High level pe:** Multiple identical blocks stack karo. Har block do main cheezein karta hai:
1. **Self-Attention** — tokens ek doosre se baat karte hain, information share karte hain
2. **Feed-Forward Network (FFN)** — har token gathered information pe deeply sochta hai

---

**Step by Step — Ek Block:**

**Input:** Token embeddings ka sequence — har word ek vector mein convert hua.

**Step 1 — Self-Attention:**
Har token har doosre token ko dekhta hai aur poochta hai: "Tere paas mere liye kya relevant information hai?"
Result: updated token representations jo ab contextual information carry karte hain.
Maslan, "bank" next to "river" ka alag representation hoga, "bank" next to "money" ka bilkul alag.

**Step 2 — Add & Norm (Residual + Layer Norm):**
Input ko attention output ke saath add karo (residual connection).
Phir Layer Normalization apply karo.
Input kyun add karte hain? Same trick jo ResNet mein thi — gradients ke liye highway deta hai, training bahut aasan ho jaati hai.

**Step 3 — Feed-Forward Network (FFN):**
Har token independently ek chhote neural network se guzarta hai.
Expansion hoti hai: $d_{\text{model}}$ → $4 \times d_{\text{model}}$ → wapas $d_{\text{model}}$ — beech mein ReLU.
Think of it as: doosre tokens se information gather karne ke baad, ab har token us information ko deeply process karta hai.

**Step 4 — Add & Norm phir se:**
FFN output ko uske input ke saath add karo, phir normalize.

---

**Interview Question — FFN itna bada kyun hota hai?**
Transformer ke **2/3 parameters FFN layers mein** hote hain! Research ne show kiya ke FFNs **key-value memories** ki tarah act karti hain — factual knowledge store karti hain. Maslan, ek language model ki FFN store karti hai ke "Paris" associated hai "capital" aur "France" se. $4 \times d_{\text{model}}$ expansion itni saari associations store karne ki capacity deta hai. Attention ne tokens ke across information mix kiya — ab FFN har token ko yeh stored knowledge access karne deta hai.

---

## 🔹 Encoder vs Decoder Blocks

| Feature | Encoder Block | Decoder Block |
|---|---|---|
| **Self-Attention** | Saare tokens dekh sakta hai (bidirectional) | Sirf past tokens dekh sakta hai (causal/masked) |
| **Cross-Attention** | Nahi | Haan — encoder output dekhta hai |
| **Best For** | Understanding, representation | Generation, translation |
| **Example Models** | BERT, RoBERTa | GPT, LLaMA, Claude |
| **Dono Use Karte Hain** | — | T5, BART, original Transformer |

---

**Encoder — The Understander**
Poora input sequence parhta hai aur rich representations banata hai jahan har word ko har doosre word ka pata hai — left aur right dono. Perfect hai tasks ke liye jahan poora context samajhna zaroori hai:
- Sentiment analysis ("not good" negative hai — dono words dekhne chahiye)
- Question answering (answer kahin bhi ho sakta hai)
- Named entity recognition

**Example:** "The bank by the river was flooded" — encoder jaanta hai "bank" matlab river bank hai, financial bank nahi — kyunki usne "river" baad mein dekha.

---

**Decoder — The Generator**
Text ek word at a time generate karta hai — left to right. Har step pe sirf woh words dekh sakta hai jo already generate ho chuke hain. Perfect hai:
- Text generation
- Translation (cross-attention se encoder dekh ke)
- Chatbots

**Example:** "The cat sat" generate karte waqt — "sat" generate karte time "The" aur "cat" visible hain, future words nahi.

---

## 🔹 Positional Encoding

**Kyun zaroori hai?**
Mind-blowing fact: Self-attention khud word order ki parwah nahi karta! Agar input sentence shuffle karo, self-attention same set of values produce karta hai — bas rearranged. Mathematically yeh **permutation invariant** hai.

Lekin word order obviously matter karta hai: "dog bites man" aur "man bites dog" bilkul alag hain!

Toh position information inject karni padti hai — alag alag tarike se.

---

**Positional Encodings ki Evolution:**

**1. Sinusoidal — Original Transformer**
Alag alag frequencies ki sine aur cosine waves use ki. Har position ko math formulas se ek unique pattern milta hai.
- ✅ Theoretically koi bhi sequence length handle kar sakta hai
- ❌ Fixed hai, learned nahi; relative positions sirf implicit hain

**2. Learned Absolute Embeddings**
Ek lookup table seekho: position 1 ko vector P1, position 2 ko P2, wagera. BERT aur early GPT mein use hua.
- ✅ Data ke hisaab se adapt kar sakta hai
- ❌ Training se lambi sequences handle nahi kar sakta

**3. RoPE — Rotary Positional Embeddings — Modern Standard**
LLaMA, Mistral, Claude, PaLM 2 — practically saare modern LLMs yeh use karte hain.
Position embeddings add karne ki jagah — Query aur Key vectors ko unki position ke hisaab se **rotate** karo.

**RoPE kaise kaam karta hai (simplified):**
Maan lo do tokens positions $m$ aur $n$ pe hain. Unka dot product (attention score) depend karta hai $(m - n)$ pe — unke beech ki **relative distance** pe. Toh model jaanta hai "yeh tokens 5 apart hain" — na ke sirf "yeh token position 37 pe hai."

---

**Interview Question — RoPE kyun better hai?**
Learned absolute embeddings ke saath: agar training 2048 length pe hui aur inference 4096 pe karo — model ne positions 2049–4096 ke embeddings kabhi dekhe hi nahi. Completely lost.

RoPE sirf relative distances care karta hai. Agar training mein do tokens 500 positions apart the, model us distance ke patterns seekhta hai. Inference mein lambi sequences ho — same relative distances exist karte hain — model generalize kar sakta hai. Isliye modern LLMs context windows extend kar sakte hain bina full retraining ke.

---

## 🔹 Causal Masking

**Kya hota hai?**
Decoder (jaise GPT) mein training ke waqt hum next word predict karna chahte hain. Lekin efficiency ke liye poora sequence ek saath process hota hai. Bina masking ke, ek word future words dekh ke "cheat" kar sakta hai!

**Example:**
Sentence: "The cat sat"
- Position 1 ("The") predict kare "cat"
- Position 2 ("cat") predict kare "sat"

Lekin agar position 2 training ke doran position 3 ("sat") dekh sake — trivial ho jaata hai. Bas copy karo jo dikha! Model actually predict karna kabhi nahi seekhta.

---

**Masking kaise kaam karta hai?**
Attention scores mein softmax se pehle ek mask add karo:
- Allowed connections (past tokens) — score as-is rakho
- Forbidden connections (future tokens) — score $-\infty$ set karo

Softmax ke baad, $e^{-\infty} = 0$ — future tokens ko zero attention weight milta hai. Effectively invisible.

**Visual — Lower Triangular Mask:**
```
     t1   t2   t3   t4
t1 [ 1    0    0    0 ]  (t1 sirf khud dekh sakta hai)
t2 [ 1    1    0    0 ]  (t2 t1 aur khud dekh sakta hai)
t3 [ 1    1    1    0 ]  (t3 t1, t2, khud dekh sakta hai)
t4 [ 1    1    1    1 ]  (t4 sab past dekh sakta hai)
```

---

**Interview Question — Causal masking critical kyun hai?**
Socho ek exam hai jahan har question ke neeche answer likha hai. Agar tum answer dekh ke question parho — tum kabhi actually solve karna nahi seekhoge, bas copy karoge. Causal masking un future answers ko cover karta hai. Training ke doran model ko actual next word dekhe bina predict karna padta hai — exactly jaise real generation mein hoga. Training aur actual use case match karte hain.

---

## 🔹 Attention ki Complexity & Scalability

**Quadratic Problem kya hai?**
Self-attention $QK^T$ compute karta hai — yeh ek $n \times n$ matrix banata hai sequence length $n$ ke liye.

- 1,000 tokens → **1,000,000** attention scores — manageable
- 100,000 tokens → **10,000,000,000** scores — impossible to store!

**Quadratic bottleneck:** Sequence length double karo — memory **4 guna** ho jaati hai.

---

**Yeh modern AI ke liye kyun matter karta hai?**
- Early Transformers sirf 512 ya 1024 tokens tak limited the
- Lambe documents truncate karne padte the
- Isliye **FlashAttention** (GPU memory cleverly use karna) aur **sparse attention** develop kiye gaye
- Aur isliye 1M token context window itna big deal hai — quadratic bottleneck overcome karna padta hai

---

**RNN vs Transformer — Final Comparison:**

| Property | RNN / LSTM | Transformer |
|---|---|---|
| **Parallelize kar sakte ho?** | Nahi — ek step at a time | Haan — sab positions ek saath |
| **Lamba yaad rakhna** | Poor — gradients vanish | Excellent — direct connections |
| **Processing time** | $O(n)$ sequential steps | $O(1)$ parallel steps lekin heavy compute |
| **Memory** | Chota, fixed | Sequence length ke saath barhta hai ($n^2$) |
| **Hardware Friendly** | Poor — sequential = GPU slow | Excellent — parallel = GPU fast |

**Simple Summary:** Transformers memory ke badle parallelism lete hain. Zyada memory use karte hain lekin modern hardware pe bahut faster chalte hain — kyunki sab kuch parallel hota hai.

---

## 🗺️ Phase 3 — Complete Summary

```
Seq2Seq — bottleneck tha: poora sentence ek vector mein
     ↓
Attention — decoder saare encoder states dekh sakta hai
     ↓
Scaled Dot-Product Attention:
  Q (query) × K (keys) → softmax → V (values) blend
     ↓
Multi-Head Attention — multiple parallel lenses
  har head alag pattern dhundta hai
     ↓
Transformer Block:
  Self-Attention (tokens baat karte hain)
  → Add & Norm
  → FFN (har token sochta hai)
  → Add & Norm
     ↓
Encoder — bidirectional, sab kuch dekhta hai (BERT)
Decoder — causal, sirf past dekhta hai (GPT, Claude)
     ↓
Positional Encoding — order inject karna:
  Sinusoidal → Learned → RoPE (relative, generalizes)
     ↓
Causal Masking — future words se cheating rokna
     ↓
O(n²) bottleneck — lambe sequences expensive
  → FlashAttention, sparse attention yahan aate hain
```

✅ **Phase 3 Complete —


# 🧠 Deep Learning Interview Revision Notes
## Phase 4: Large Language Models — Architecture & Training Pipeline

---

## 🔹 Teen Types ke LLM Architectures

---

### 1. Encoder-Only (BERT, RoBERTa)

**Kya karta hai?**
Poora input ek saath parhta hai aur deeply samajhta hai. Har word har doosre word ko dekh sakta hai — left bhi, right bhi. Jaise ek room mein sab se ek saath baat kar sako.

**Training trick kya hai?**
**Masked Language Modeling** — 15% words chhupa do aur model se guess karwao.

**Kahan use hota hai?**
Understanding tasks mein — maslan movie review positive hai ya negative, text mein logon ke naam dhundna, ya paragraph mein se question ka answer nikalna.

**Kya nahi kar sakta?**
Naya text generate nahi kar sakta — essay likhna, chat karna. Kyunki usne kabhi words ek ek karke produce karna seekha hi nahi.

**Example:**
```
Input: "The [MASK] sat on the mat."
Model dono sides dekh ke predict karta hai → "cat"
```

---

### 2. Decoder-Only (GPT, LLaMA, Claude)

**Kya karta hai?**
Text ek word at a time generate karta hai — left to right. Har word sirf apne pehle wale words dekh sakta hai — aagey nahi jhankta. Jaise book parh rahe ho aur next page nahi dekha.

**Training trick kya hai?**
**Next-word prediction** — "The cat sat" diya, model predict kare "on", phir "the", phir "mat".

**Kahan use hota hai?**
Essay likhna, coding, chatting, question answering — koi bhi task jahan naya text produce karna ho.

**Aaj kal sabse popular architecture yahi hai** — GPT, LLaMA, Mistral, Gemini sab yahi.

**Example:**
```
"The cat" → predict → "sat" → predict → "on" → predict → "the" → predict → "mat"
```

---

### 3. Encoder-Decoder (T5, BART)

**Kya karta hai?**
Dono combine karta hai — encoder se input poori tarah samajhta hai (bidirectional), decoder se output step by step generate karta hai aur saath mein encoder ka output bhi dekhta rehta hai.

**Training trick kya hai?**
Text ke spans mask karo — kuch words ek special token se replace karo — aur model se reconstruct karwao.

**Kahan use hota hai?**
Translation, summarization — jahan ek input lo aur ek different output produce karo.

**Example:**
```
Input (English): "The cat sat on the mat."
Output (French): "Le chat s'est assis sur le tapis."
```

---

### ⚖️ Teen Architecture Analogy

- **Encoder-only** — ek librarian jo books mein information dhundne mein expert hai, lekin nai book likh nahi sakta
- **Decoder-only** — ek author jo kuch bhi likh sakta hai, lekin deep understanding ke liye context chahiye
- **Encoder-decoder** — ek translator jo poori book parhta hai aur doosri language mein summary likhta hai

---

## 🔹 Industry Decoder-Only Pe Kyun Shift Hui?

Yeh bahut common interview question hai. Teen bade reasons hain:

---

**Reason 1 — Ek model, sab kuch:**
Decoder-only ke saath koi bhi task sirf prompt likh ke ho jaata hai.
- Translation chahiye? `"Translate to French: The cat sat →"`
- Summary chahiye? `"Summarize: [long article] →"`
- Poem chahiye? `"Write a poem about rain →"`

BERT (encoder-only) ke saath har task ke liye alag "head" (extra layers) chahiye tha, alag training chahiye thi. Decoder-only universally kaam karta hai — bas text complete karo.

---

**Reason 2 — Training bahut efficient hai:**
Next-word prediction mein **har ek word** ek learning signal deta hai.
100 words ka sentence = **100 predictions** seekhne ke liye.

BERT mein sirf woh ~15% words se seekhte the jo mask kiye the. Bahut kam efficient.

---

**Reason 3 — Scale pe magical abilities emerge hoti hain:**
Jab decoder-only models bahut bade ho jaate hain — zyada data, zyada parameters — woh cheezein karne lagte hain jo explicitly train nahi ki gayi thi. Math problems solve karna, reasoning karna. Ise **in-context learning** kehte hain — prompt mein kuch examples dikhaao, model figure out kar leta hai. BERT-style models mein yeh magic kabhi nahi aya.

---

## 🔹 Teen-Step Training Pipeline

Modern LLMs ek baar mein train nahi hote. Teen phases hoti hain:

```
Raw internet text (trillions of words)
        ↓
  1. PRE-TRAINING — base model banta hai
        ↓
  2. SUPERVISED FINE-TUNING (SFT) — instructions follow karna seekhta hai
        ↓
  3. ALIGNMENT (RLHF ya DPO) — helpful, harmless, honest banta hai
```

---

## 🔹 Phase 1 — Pre-Training

**Kya hota hai?**
Model ko **internet ka bada hissa** khilaya jaata hai — books, Wikipedia, Reddit, code, sab kuch. Model ka kaam: har sentence mein next word predict karo.

**Yeh kyun kaam karta hai?**
Next word achhi tarah predict karne ke liye model ko seekhna padta hai:
- Grammar aur language structure
- Duniya ke baare mein facts — maslan "Paris is the capital of France"
- Reasoning patterns — maslan "2+2=" dekha toh "4" predict karo

**Result kya milta hai?**
Ek **base model** jo text complete karne mein great hai — lekin instructions follow karna nahi jaanta, helpful hona nahi jaanta. Koi bhi prompt doge, woh complete kar dega — chahe harmful ho.

**Example:**
Agar prompt diya "How to make a bomb:" — base model instructions complete kar sakta hai kyunki usne internet pe aisa text dekha tha. Yeh dangerous hai — isliye agle steps hain.

---

## 🔹 Phase 2 — Supervised Fine-Tuning (SFT)

**Kya hota hai?**
Base model ko ek **chote, high-quality dataset** pe fine-tune karte hain — (prompt, ideal response) pairs jo humans ne likhe hain.

| Prompt | Ideal Response |
|---|---|
| "What is the capital of France?" | "The capital of France is Paris." |
| "Summarize: [article]" | [short summary] |

**Loss kaise calculate hoti hai?**
Same next-word prediction — lekin loss sirf **response part** pe calculate hoti hai. Prompt sirf context hai.

**SFT kya sikhata hai?**
- Instructions kaise follow karni hain
- Conversation format — kab "User:" kab "Assistant:" likhna hai
- Style — concise raho, sources cite karo wagera

**Key insight:**
Model pehle se facts jaanta hai pre-training se. SFT sikhata hai ke **kaise jawab dena hai** — jaise ek knowledgeable insaan ko achha teacher banana.

**Kitna data chahiye?**
Millions nahi chahiye. **10,000–50,000 high-quality examples** kaafi hain. Quality, quantity se zyada matter karti hai.

**Example:**
Agar chahte ho model hamesha politely jawab de, toh include karo:
`"Explain gravity."` → `"Gravity is a force that attracts objects with mass. It's what keeps us on Earth!"`

**Limitation kya hai?**
SFT sirf examples imitate karta hai. Yeh explicitly nahi sikhata ke ek achhe answer ko thode worse answer se prefer karo. Isliye alignment chahiye.

---

## 🔹 Phase 3 — Alignment

**Kyun zaroori hai?**
SFT ke baad model instructions follow karta hai — lekin phir bhi:
- Sahi lekin rude ya unhelpful answers de sakta hai
- "I don't know" kab kehna hai yeh nahi jaanta
- Harmful responses mein trick ho sakta hai

Hum chahte hain model **Helpful, Honest, aur Harmless (HHH)** ho. Alignment yeh sikhata hai.

Do main methods hain: **RLHF** (purana, complex) aur **DPO** (naya, simple).

---

### RLHF — Reinforcement Learning from Human Feedback

**Step 1 — Human preferences collect karo:**
Kaafi prompts ke liye SFT model se multiple responses generate karo. Humans ko dono dikhao aur **rank** karwao — kaunsa better hai, kaunsa worse. Result: (good response, bad response) pairs milte hain.

**Step 2 — Reward Model (RM) train karo:**
Ek alag model train karo — often SFT model ki copy — jo responses ko **score** kare. Yeh human preference predict karna seekhta hai: response diya, number output karo — zyada = better. Training objective: good response ko bad response se zyada score milna chahiye.

**Step 3 — LLM ko Reinforcement Learning se fine-tune karo (PPO):**
Ab hamare paas reward model hai jo koi bhi response score kar sakta hai. **Proximal Policy Optimization (PPO)** use karo SFT model update karne ke liye — taake woh responses generate kare jo RM se high scores lein.

Important: ek penalty add hoti hai agar model original SFT model se bahut door chala jaaye — yeh "reward hacking" rokta hai jahan model reward model ke loopholes exploit karne lagta hai.

**RLHF complicated kyun hai?**
- Teen models simultaneously maintain karne padte hain: SFT model, Reward Model, aur jo train ho raha policy hai
- PPO unstable hai — hyperparameters mein choti si change training tod sakti hai
- Expensive hai kyunki continuously naye responses generate karne padte hain (online training)

---

### DPO — Direct Preference Optimization (2023)

**Brilliant insight kya tha?**
Mathematically, reward model train karna skip kar sakte ho completely! Directly preference data se LLM optimize kar sakte ho.

**Kaise kaam karta hai?**
- SFT model se shuru karo — ise reference model kaho
- Har preference pair (good response, bad response) ke liye loss yeh encourage karta hai:
  - Reference model se compare karke **good response** ki probability badhao
  - Reference model se compare karke **bad response** ki probability ghataao
- Yeh ek simple supervised loss hai — koi RL nahi, koi reward model nahi

**Intuition:**
> "Model ko kehna hai: 'Mujhe good answer dene ki zyada aadat daalni hai pehle se, aur bad answer dene ki aadat ghataani hai.'"

**DPO easy kyun hai?**
- Reward model train nahi karna
- RL loop nahi — bas standard loss jaise SFT
- Stable, fast, aur cheap
- Bahut saare open-source models mein use hota hai — Zephyr, LLaMA 3

**Trade-off kya hai?**
RLHF kabhi kabhi thoda better perform kar sakta hai kyunki woh training ke doran naye responses explore karta hai. Lekin DPO bahut simpler hai aur aksar kaafi hota hai.

---

## 🔹 Modern Architecture Details — LLaMA Style

Original Transformer 2017 mein tha — ab hum jaante hain kuch cheezein optimal nahi thi. Modern LLMs improved components use karte hain:

| Component | Original | Modern (LLaMA) | Kyun Change Kiya? |
|---|---|---|---|
| **Normalization Position** | Residual ke baad (Post-LN) | Residual se pehle (Pre-LN) | Pre-LN bahut deep models mein zyada stable hai |
| **Normalization Type** | LayerNorm | RMSNorm | RMSNorm faster hai (mean subtract nahi karta), equally effective |
| **Position Encoding** | Sinusoidal | RoPE | Relative positions better handle karta hai, lambi sequences pe generalize karta hai |
| **Attention** | Multi-Head | Grouped-Query Attention (GQA) | Memory bachata hai, inference fast karta hai |
| **FFN Activation** | ReLU | SwiGLU | Same compute mein better performance |
| **Bias Terms** | Haan | Nahi | Thodi better generalization, parameters bhi bachte hain |

---

**SwiGLU kya hai?**
Simple ReLU ki jagah, SwiGLU ek gate use karta hai:

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$$

Do parallel linear layers hain — ek doosre ka flow control karta hai. Zyada expressive hai. PaLM, LLaMA sab use karte hain.

---

**Pre-LN vs Post-LN:**

**Post-LN (original):** Pehle residual add karo, phir normalize karo. Output ka variance bada ho sakta hai — training unstable ho jaati hai.

**Pre-LN (modern):** Pehle normalize karo, phir layer apply karo, phir residual add karo. Gradients better flow karte hain, training smooth rehti hai.

---

## 🔹 Full LLM Lifecycle — Visual Summary

```
INTERNET TEXT (trillions of tokens)
        ↓
╔══════════════════════════════════╗
║         PRE-TRAINING             ║
║   Next-word prediction           ║
║   (weeks, thousands of GPUs)     ║
╚══════════════════════════════════╝
        ↓  Base Model — language aur facts jaanta hai
╔══════════════════════════════════╗
║   SUPERVISED FINE-TUNING (SFT)   ║
║   ~10k–100k (prompt, response)   ║
║   pairs — humans ne likhe        ║
╚══════════════════════════════════╝
        ↓  SFT Model — instructions follow karta hai
╔══════════════════════════════════╗
║           ALIGNMENT              ║
║  ┌─────────────────────────┐    ║
║  │ RLHF: Reward Model+PPO  │    ║
║  │          YA              │    ║
║  │ DPO: Direct Preference  │    ║
║  └─────────────────────────┘    ║
╚══════════════════════════════════╝
        ↓  Aligned Model — helpful, harmless, honest
```

---

## 💡 Key Interview Questions

**Q: Industry BERT se GPT-style models pe kyun shift hui?**
GPT ek hi architecture se sab kuch kar sakta hai — bas prompt likho. Training bhi efficient hai — har word ek learning signal hai. Aur scale pe in-context learning jaisi magical abilities emerge hoti hain. BERT ko har task ke liye alag setup chahiye tha.

**Q: Pre-training aur SFT mein kya difference hai?**
Pre-training general knowledge sikhata hai internet text se — "kya jaanna hai." SFT sikhata hai ke questions ka properly jawab kaise dena hai — "kaise respond karna hai." Pre-training ek knowledgeable insaan banana hai, SFT use achha teacher banana hai.

**Q: RLHF simple terms mein kaise kaam karta hai?**
Pehle humans se alag alag answers compare karwao aur best choose karwao. Phir ek "reward model" train karo jo human jaisa score kare. Phir reinforcement learning se LLM ko tweak karo taake woh high-scoring answers generate kare — lekin original model se bahut door na jaaye.

**Q: DPO ne kya problem solve ki?**
Reward model train karne ki zaroorat khatam ki, aur unstable RL loop bhi khatam kiya. Directly preference data se LLM optimize hota hai — alignment ek standard supervised task ban jaata hai. Simple, fast, stable.

**Q: Modern LLMs RoPE kyun use karte hain absolute position embeddings ki jagah?**
Absolute embeddings woh sequences handle nahi kar sakte jo training se lambi hoon — model ne woh positions dekhi hi nahi. RoPE positions ko rotations ke roop mein encode karta hai — attention relative distances pe depend karta hai. Yeh longer contexts pe generalize karta hai bina retraining ke.

---

## 🗺️ Phase 4 — Complete Summary

```
Teen architectures:
  Encoder-only (BERT) — samajhna, generate nahi
  Decoder-only (GPT) — generate karna — aaj ka standard
  Encoder-Decoder (T5) — input samjho, output generate karo

Decoder-only kyun jeet gaya?
  → Universal (prompt se sab kuch)
  → Efficient training (har word = signal)
  → Scale pe emergent abilities

Training Pipeline:
  Pre-training → base model (facts, language)
  SFT → instruction following
  Alignment → helpful, harmless, honest
    RLHF: reward model + PPO (powerful lekin complex)
    DPO: direct preference (simple, stable, modern choice)

Modern upgrades (LLaMA style):
  Pre-LN, RMSNorm, RoPE, GQA, SwiGLU, no bias
```

✅ **Phase 4 Complete — Phase 5 bhejo!**
