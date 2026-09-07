# PGN (Polynomial Gated Network) — V9 Release 1

PGN is a compact artificial neural network topology designed to handle non-linear classification using significantly fewer parameters than traditional fully connected layers.

Instead of stacking deep hidden layers to learn complex decision boundaries, PGN uses two parallel feature branches, an explicit second-order polynomial fusion step, and a sigmoid gate. This allows the network to capture feature interactions early without exploding parameter counts.

---

## The Core Idea

Standard dense layers rely on linear combinations ($W \cdot X + b$) followed by point-wise activations. Learning high-order feature interactions usually requires adding more layers or increasing hidden units, both of which increase weight count rapidly.

PGN approaches non-linearity differently:

1. **Dual Parallel Branches ($G_1$ and $G_2$):** Process features along two low-dimensional parallel paths.
2. **Polynomial Interaction ($Poly$):** Explicitly computes second-order interactions between branch outputs ($out_{g1}^2$, $out_{g1} \cdot out_{g2}$, $out_{g2}^2$).
3. **Sigmoid Control Gate ($Gate$):** Uses $out_{g2}$ as a dynamic scalar filter to decide how strongly polynomial features pass through.
4. **Direct Fusion:** Bypasses intermediate bottlenecks by feeding $out_{g1}$, $out_{g2}$, and the gated interaction signal straight to the output logits.

---

## Math & Forward Pass Pipeline

Given input vector $X$:

### 1. Parallel Branching
$$z_{g1} = X W_{g1} + b_{g1} \quad \implies \quad a_{g1} = \text{LeakyReLU}(z_{g1})$$
$$out_{g1} = a_{g1} W_{g1\_out}$$

$$z_{g2} = X W_{g2} + b_{g2} \quad \implies \quad a_{g2} = \text{LeakyReLU}(z_{g2})$$
$$out_{g2} = a_{g2} W_{g2\_out}$$

### 2. Polynomial Interaction & Gating
$$Poly = (out_{g1})^2 + (out_{g1} \cdot out_{g2}) + (out_{g2})^2$$
$$Gate = \sigma(\text{clip}(out_{g2}, -12, 12))$$
$$red\_node = (Poly \cdot Gate) + b_{red}$$

### 3. Feature Concatenation & Output
$$Signals = \begin{bmatrix} out_{g1} & out_{g2} & red\_node \end{bmatrix}$$
$$z_{final} = Signals \cdot W_{fuse} + b_{out}$$
$$\hat{Y} = \text{Softmax}(z_{final})$$

---

## Benchmark Comparison

Evaluated across 20 random seeds over 1200 epochs against a baseline Dense Net with 8 hidden units:

### Iris Classification (3 classes, 4 features)
* **Dense Baseline:** 67 parameters | ~95.5% Test Acc | ~0.44s
* **PGN (V9 Release 1):** **37 parameters** | **~95.4% Test Acc** | **~0.46s**

### Breast Cancer Diagnosis (2 classes, 30 features)
* **Dense Baseline:** 275 parameters | ~96.4% Test Acc | ~0.83s
* **PGN (V9 Release 1):** **137 parameters** | **~96.3% Test Acc** | **~0.59s**

---

## Getting Started

### Requirements
- Python 3.8+
- NumPy
- scikit-learn

The neural network code is in main.py.
