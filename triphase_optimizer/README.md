# Triphase Optimizer: Physics-Inspired Loss Function

[](https://pytorch.org/)
[](https://opensource.org/licenses/MIT)
[](http://makeapullrequest.com)

> **A lightweight, drop-in loss function for Coherence-Driven & Noise-Robust Deep Learning.**

**Triphase Optimizer** is a novel loss function designed to improve stability, information coherence, and noise robustness in neural networks. Inspired by *Triphase Cosmology Theory*, it treats the learning process as a "coherence event" among three phase-spaces.

It integrates three complementary components into a single interference-based objective:

1.  **Accuracy Term ($H_+$)**: Standard CrossEntropy (Reality)
2.  **Stability Term ($H_-$)**: L2 Regularization (Gravity/Tension)
3.  **Coherence Term ($iH_{im}$)**: Entropy Minimization (Information Coherence)

Together, they help models learn stable and coherent representations, especially under **high-entropy / noisy environments**.

-----

## Why Triphase Optimizer?

Traditional optimizers often focus strictly on minimizing prediction error. Triphase Optimizer explicitly controls three dimensions:

  * **Accuracy** — Fits the data (Standard learning)
  * **Stability** — Prevents exploding weights & overfitting (Structural constraint)
  * **Coherence** — Reduces internal uncertainty/entropy (Informational consistency)

This produces a model that maintains **higher confidence calibration** and performs better when training data is corrupted by noise.

### Key Features

  * **Noise Robustness:** Improves generalization under noisy training conditions.
  * **Coherence:** Dramatically reduces prediction entropy.
  * **Universal Compatibility:** Works with any PyTorch model (CNN, ResNet, Transformer).
  * **Lightweight:** Tiny implementation (\< 20 lines of code).
  * **Optimizer Agnostic:** Works alongside SGD, Adam, AdamW, etc.

-----

## Experimental Results

We tested the optimizer with a **CNN on the MNIST dataset**, injecting extreme Gaussian noise ($\sigma=1.2$) into the training data to evaluate robustness.

### 1\. Test Accuracy (The "Golden Cross")

While the baseline (Standard SGD) hits a ceiling due to overfitting noise, **Triphase Optimizer** initially lags (rejecting noise) but eventually overtakes the baseline to achieve higher peak accuracy.

\<img src="./result\_accuracy.png" width="800"\>

### 2\. Information Coherence (Entropy)

Triphase dramatically reduces internal entropy throughout training. Lower entropy indicates more stable, confident predictions — a hallmark of coherent internal representations.

\<img src="./result\_coherence.png" width="800"\>

-----

## How It Works

The loss function is defined as the sum of interference terms:

$$L_{total} = L_{CE} + \alpha \cdot L_{2} + \beta \cdot L_{Entropy}$$

Where:

  * $L_{CE}$: CrossEntropy Loss (Positive Phase)
  * $L_{2}$: L2 Regularization on weights (Negative Phase)
  * $L_{Entropy}$: Entropy of the output distribution (Imaginary Phase)

| Parameter | Role | Recommended Range |
| :--- | :--- | :--- |
| **`alpha`** | Controls structural stability (Gravity) | `0.0001` - `0.01` |
| **`beta`** | Controls coherence / entropy reduction | `0.1` - `0.5` |

*Setting `alpha=0, beta=0` reproduces standard behavior (Baseline).*

-----

## Installation

1.  **Clone this repository:**

    ```bash
    git clone https://github.com/yourname/triphase-optimizer.git
    cd triphase-optimizer
    ```

2.  **Install dependencies:**

    ```bash
    pip install torch torchvision matplotlib
    ```

-----

## Usage Example (PyTorch)

It is designed to be **Plug-and-Play**. No changes to your existing model or optimizer are needed.

```python
import torch
import torch.optim as optim
# Import the custom loss
from triphase_loss import TriphaseLoss

# 1. Define Model & Optimizer
model = MyCNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 2. Initialize Triphase Loss
# alpha: Gravity/Stability, beta: Coherence
criterion = TriphaseLoss(alpha=0.0005, beta=0.2)

# 3. Training Loop
for data, target in train_loader:
    optimizer.zero_grad()
    outputs = model(data)
    
    # Calculate Loss (Returns tuple, take the first element as total loss)
    loss, l_pos, l_neg, l_im = criterion(outputs, target, model)
    
    loss.backward()
    optimizer.step()
```

-----

## Project Files

```text
.
├── triphase_loss.py        # Core implementation (The Loss Function)
├── experiment.py           # Script to reproduce the MNIST noise experiment
├── result_accuracy.png     # Benchmark result
├── result_coherence.png    # Benchmark result
├── README.md               # This file
├── README_JA.md            # Japanese documentation
└── pdf/
    └── Triphase_Cosmology.pdf  # Theoretical Background
```

> **Note:** The PDF contains the theoretical background ("Triphase Cosmology"), but reading it is not required to use this tool effectively.

-----

## Citation

If you use this optimizer in your research or project, please cite:

```bibtex
@misc{triphase2025,
  title={Triphase Optimizer: A Physics-Inspired Loss Function for Coherence-Driven Learning},
  author={Sato, Yohei},
  year={2025},
  howpublished={GitHub Repository},
  url={https://github.com/yourname/triphase-optimizer}
}
```

## Contributions

Pull requests, issues, and benchmarks on other datasets (CIFAR-10, ImageNet, NLP tasks) are welcome\!

## License

[MIT License](https://www.google.com/search?q=LICENSE)

-----

## Japanese Documentation

日本語のドキュメントはこちら: [README\_JA.md](https://www.google.com/search?q=README_JA.md)