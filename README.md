# ⚛️ SPECTRA v26.3

### *Soft X-ray Physics-Engineered Computing for Tomographic Reconstruction & Analysis*

**Google Solutions Challenge 2026 · Theme: Unbiased AI Decision · Open Innovation Track**
**SDG 7: Affordable and Clean Energy**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vijaideen/SPECTRA-GSC2026/blob/main/SPECTRA_v26_3.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-vijaideen%2FSPECTRA--GSC2026-181717?logo=github)](https://github.com/vijaideen/SPECTRA-GSC2026)
![SDG 7](https://img.shields.io/badge/SDG-7%20Affordable%20%26%20Clean%20Energy-FDBE00)
![Track](https://img.shields.io/badge/Track-Open%20Innovation-4285F4)

---

## 🚀 Live MVP

| | |
|---|---|
| **▶ Run in Colab** | [https://colab.research.google.com/github/vijaideen/SPECTRA-GSC2026/blob/main/SPECTRA_v26_3.ipynb](https://colab.research.google.com/github/vijaideen/SPECTRA-GSC2026/blob/main/SPECTRA_v26_3.ipynb) |
| **GitHub Repo** | [https://github.com/vijaideen/SPECTRA-GSC2026](https://github.com/vijaideen/SPECTRA-GSC2026) |
| **Runtime Required** | T4 GPU (free in Colab) |
| **Training Time** | ~5–7 minutes |
| **Inference Speed** | < 1 ms on GPU |

> ⚠️ Set runtime to **T4 GPU** before running — `Runtime → Change runtime type → T4 GPU → Save`

---

## 🎯 The Problem This Project Solves

> *"Computer programs now make life-changing decisions about who gets a job, a bank loan, or even medical care. However, if these programs learn from flawed or unfair historical data, they will repeat and amplify those exact same discriminatory mistakes."*
> — Google Solutions Challenge 2026, Theme: Unbiased AI Decision

Most AI systems today are trained on **human-labelled historical data**. That data reflects the real world — including its inequalities, its prejudices, and its systemic discrimination. When an AI learns from this data, it doesn't just inherit those biases. It **automates and amplifies** them, making biased decisions faster and at greater scale than any human could.

**The core question SPECTRA addresses:**
> *Can an AI system make accurate, consequential decisions without ever learning from biased human-labelled data?*

**SPECTRA's answer: yes.**

---

## 🔓 Open Innovation Track

SPECTRA is submitted under the **Open Innovation track** of the Unbiased AI Decision theme.

Instead of building a bias detector for hiring or lending, SPECTRA demonstrates a **fundamentally different solution** to the bias problem — one that **eliminates the source of bias entirely** by replacing human-labelled training data with objective physical law.

The innovation is not a tool that finds bias after the fact. It is a proof that an AI can be **structurally incapable of inheriting human bias** — because it was never trained on human judgement in the first place.

---

## 💡 How SPECTRA Eliminates Bias

The root cause of AI bias is simple: models learn what humans decided in the past. SPECTRA learns what **physics dictates** instead — a standard that is objective, verifiable, and completely independent of any human demographic, cultural, or historical context.

```
❌ Traditional AI:
   Human-labelled data → Model learns human bias → Biased decisions at scale

✅ SPECTRA:
   Physics laws only → Model learns objective constraints → Unbiased decisions
```

SPECTRA is a **Physics-Informed Neural Network (PINN)** applied to Soft X-Ray (SXR) tomography in nuclear fusion reactors. It reconstructs 2D plasma emissivity profiles from noisy sensor measurements — making real-time decisions that directly affect reactor safety and energy output — **with zero human-labelled data points in its training signal.**

Every decision SPECTRA makes is governed by **15 physics-derived constraints** — not by what a human operator decided in a previous experiment. There is no historical dataset to be unfair. There is no labeller whose prejudice can contaminate the model. The training signal is plasma physics, which applies equally to every particle in the reactor.

---

## 🌍 Why This Matters Beyond Fusion

The principle SPECTRA demonstrates is universal. Wherever AI decisions carry high stakes and training data carries historical bias, the same approach applies:

| Domain | The Bias Problem | The SPECTRA Principle Applied |
|---|---|---|
| **Medical diagnosis** | Models trained on historical diagnoses that under-served minority populations | Physics/biology-constrained model trained on objective physiological laws |
| **Credit scoring** | Models trained on loan histories that reflect decades of redlining | Rule-constrained model trained on objective financial risk principles |
| **Criminal justice** | Models trained on arrest records reflecting unequal policing | Constraint-based model trained on case facts, not demographic proxies |
| **Fusion energy** | Models trained on past operator decisions and labelled experimental data | Physics-constrained PINN — **this project** |

SPECTRA is the **proof of concept** that physics-constrained, label-free AI works in a real high-stakes domain. The architectural principle — replace human labels with objective constraints — is the blueprint.

---

## ⚛️ What SPECTRA Does

Nuclear fusion is one of humanity's most promising paths to clean, unlimited energy. But a key barrier to making fusion work commercially is **plasma instability** — sudden disruptions that can terminate the plasma and damage the reactor.

To prevent disruptions, reactors need to continuously monitor what's happening inside the plasma in real time. One critical tool is **Soft X-Ray (SXR) tomography**: arrays of detectors around the reactor measure radiation passing through the plasma along different paths. From these measurements, you need to reconstruct a full 2D image of the plasma's internal state.

This is genuinely hard:
- The reconstruction is mathematically **ill-posed** — many possible images could match the same measurements
- The measurements are always **noisy**
- Classical methods blur away important details or require manual tuning per experiment
- Standard ML can't help — there's **no labelled dataset** of correct plasma images to train on

**SPECTRA solves this by training entirely on physics.** Instead of asking *"what did human experts label as correct before?"*, it asks *"what does plasma physics say must be true?"*

```
SXR sensor measurements (noisy)
            ↓
SPECTRA: physics-only reconstruction  ← this project
            ↓
Plasma instability detection
            ↓
Real-time reactor control response
            ↓
Stable plasma → sustained fusion → clean energy → SDG 7
```

---

## 🏗️ How It Works

### Phase 1 — Forward Simulator
Builds a physics-accurate model of the reactor geometry and generates synthetic plasma measurements. Produces:
- A synthetic plasma emissivity phantom from a Bremsstrahlung profile model
- A chord geometry matrix via Siddon ray tracing through an elliptical plasma boundary
- Noisy measurements with physically correct Poisson shot noise and Gaussian electronic noise

### Phase 2 — PINN Training
Trains the neural network using 15 physics constraints as the only training signal. No human labels at any point. 10,000 epochs across a 3-stage curriculum.

---

## 🧠 Architecture

| Component | Choice | Why |
|---|---|---|
| Network type | Deep MLP + skip connection, 6 hidden layers | Enough capacity to invert the 1D → 2D mapping |
| Activation | SiLU + LayerNorm | Stable training with physics-based loss landscapes |
| Output | Sigmoid → `[0,1]` radial profile | Physically bounded — negative emissivity is impossible |
| Optimiser | Adam + warmup cosine decay | Smooth, stable convergence |
| Training strategy | 3-stage curriculum: low → medium → high noise | Prevents collapse on hard cases early in training |
| Framework | JAX + Flax + Optax | Full JIT compilation, entirely GPU-accelerated |

---

## ⚙️ The 15-Term Physics Loss — The Bias Elimination Mechanism

**Zero ground truth. Zero human labels. Zero historical bias.**

| Loss Term | Weight | What It Enforces |
|---|---|---|
| `l_data` (Poisson-weighted) | 1.000 | Measurements consistent with SXR detector noise physics |
| `l_sym` | 0.100 | Plasma must be symmetric about the midplane |
| `l_neg` | 0.100 | Emissivity cannot be negative — physically impossible |
| `l_fsa` | 0.050 | Flux-surface averaging consistency |
| `l_edge` | 0.050 | Emissivity must approach zero at the plasma boundary |
| `l_tv` | 0.020 | Profile must be smooth — sharp artificial edges are unphysical |
| `l_abel` | 0.020 | Must satisfy Abel transform self-consistency |
| `l_xcam` | 0.020 | Both detector cameras must agree on an axisymmetric plasma |
| `l_brem` | 0.010 | Core profile shape must follow the Bremsstrahlung emission law |
| `l_mfi` | 0.005 | Minimum Fisher Information — avoids over-fitting fine structure |
| `l_smooth2d` | 0.005 | 2D spatial gradients must be physically smooth |
| `l_diffusion` | 0.005 | Laplacian diffusion regularisation |
| `l_radial_mono` | 0.002 | Soft radial monotonicity prior |
| `l_phys` | 5e-4 | 1D radial curvature smoothness |
| `l_energy` | 5e-4 | Total energy proxy consistent with chord-integrated measurements |

Ground truth is loaded **only** for post-hoc evaluation metrics after training fully completes — it never enters any gradient computation.

---

## 🧪 Three Plasma Profiles Tested

| Profile | Physics Mode | Description |
|---|---|---|
| **Peaked** | H-mode | Sharp core emission peak with pedestal shoulder |
| **Broad** | L-mode | Wider, lower-gradient profile |
| **Hollow** | Reversed shear | Off-axis emission ring — the most challenging case |

---

## 📊 Benchmarks

SPECTRA compared against six methods on identical hardware (JAX, T4 GPU, same geometry matrix):

| Method | Type |
|---|---|
| Backprojection | Classical, no regularisation |
| JAX-Tikhonov (CG) | Iterative, L-curve regularisation |
| JAX-MFI | Iterative, Minimum Fisher Information |
| Moving Average | Signal-domain smoothing |
| Gaussian Filter | Signal-domain smoothing |
| Savitzky-Golay | Polynomial fitting |
| **★ SPECTRA v26.3** | **Physics-informed neural network** |

**Cross-noise robustness** tested at three independent noise levels:

| Level | Sigma | Represents |
|---|---|---|
| Low | σ = 0.001 | Clean detector — ~52 dB SNR |
| Medium ★ | σ = 0.003 | Calibrated WEST tokamak detector — ~43 dB SNR |
| High | σ = 0.008 | Degraded detector conditions — ~32 dB SNR |

---

## 🖥️ Live UI — 5 Tabs

| Tab | What You See |
|---|---|
| ◉ **Live Console** | Real-time training output streamed as the model trains |
| 〜 **Loss Curves** | Live chart of all 15 physics loss terms updating during training |
| ◈ **Results & Plots** | 5-panel output: ground truth · backprojection · SPECTRA · error map · 1D profile |
| ⚖️ **Benchmark** | Side-by-side comparison against all classical methods with full metrics table |
| 🔬 **Noise Robustness** | Cross-noise evaluation: metrics table, radial profiles, and 2D reconstruction maps |

---

## 🚀 Getting Started

### What You Need
- A Google account
- Google Colab — free tier is fine
- **T4 GPU runtime** — free in Colab, required for training speed
- No local installation needed

### Run It

**1. Click the link below to open directly in Colab:**

👉 **[https://colab.research.google.com/github/vijaideen/SPECTRA-GSC2026/blob/main/SPECTRA_v26_3.ipynb](https://colab.research.google.com/github/vijaideen/SPECTRA-GSC2026/blob/main/SPECTRA_v26_3.ipynb)**

**2. Set runtime to T4 GPU:**
```
Runtime → Change runtime type → T4 GPU → Save
```

**3. Run all cells:**
```
Runtime → Run all
```

**4. Open the Live UI:**
Wait for Cell 1 to finish installing (~2 min). Click the public URL printed under Cell 3.

**5. Wait for training (~5–7 min):**
All 5 tabs populate automatically when done.

> ⚠️ Keep your Colab tab open while using the UI — the tunnel closes when the session ends.

---

## 📁 Project Structure

```
SPECTRA_v26_3.ipynb        Main notebook — open in Google Colab
README.md                  This file

Generated automatically at runtime:
├── app.py                 Streamlit Live UI (written by Cell 2, launched by Cell 3)
├── sxr_data/
│   ├── epsilon_true_2d.npy       Ground truth phantom (post-hoc eval only)
│   ├── g_noisy.npy               Noisy SXR chord measurements
│   ├── W_matrix.npz              Chord geometry matrix
│   └── rho.npy                   Radial coordinate array
└── sxr_pinn_results/
    ├── epsilon_reconstructed.npy  SPECTRA reconstruction output
    ├── training_history.npy       Loss curve data
    └── benchmark_results.npy      Comparison metrics
```

---

## ⚠️ Scope and Limitations

SPECTRA is a **proof of concept** validated on physics-consistent synthetic data — standard methodology in fusion tomography research (Anton et al. RSI 1996, Odstrcil et al. RSI 2016).

| Limitation | Current State | Planned Next Step |
|---|---|---|
| Geometry | Elliptical plasma boundary (κ=1.3) | Integrate full EFIT MHD equilibrium |
| Data | Synthetic phantoms only | Validate against real WEST tokamak SXR data |
| Hardware | Timing on Colab T4 | Re-benchmark on dedicated tokamak control hardware |
| Integration | Reconstruction step only | Connect to full disruption prediction pipeline |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: flax` | Re-run Cell 1 |
| First epoch slow (~15 s) | Normal — JAX JIT compiles on first run |
| `lax.scan` compile slow (~20 s) | Normal — XLA compiles the CG graph once then caches |
| Out of memory on T4 | Reduce `grid_size` to `64` in `CFG` (Cell 5) |
| Streamlit page blank | Wait ~8 s after Cell 3, then hard-refresh the browser tab |
| UI URL not clickable | Re-run Cell 3 — Colab tunnel may need a moment |
| Benchmark tab empty | Make sure Cell 10 ran before opening the UI |
| Noise Robustness tab empty | Populates automatically after benchmark — wait for Phase 4 |
| `Missing required data files` | Re-run Cells 6 and 8 before running Cells 10 or 11 |

---

## 📚 References

- Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks. *Journal of Computational Physics.*
- Mazon et al. (2015). Soft x-ray tomography for real-time applications. *Fusion Engineering and Design.*
- Hutchinson, I.H. *Principles of Plasma Diagnostics.* Cambridge University Press.
- Ertl et al. (1996). Minimum Fisher Information regularisation. *Review of Scientific Instruments.*
- Anton et al. (1996). X-ray tomography on the TCV tokamak. *Review of Scientific Instruments.*
- Odstrcil et al. (2016). Optimized tomography methods for plasma emissivity reconstruction. *Review of Scientific Instruments.*

---

## 🏆 Submission Info

| | |
|---|---|
| **Competition** | Google Solutions Challenge 2026 |
| **Theme** | Unbiased AI Decision |
| **Track** | Open Innovation |
| **SDG** | SDG 7 — Affordable and Clean Energy |
| **Version** | SPECTRA v26.3 |
| **Stack** | JAX · Flax · Optax · Streamlit · Python |
| **Hardware** | Google Colab T4 GPU |
| **MVP Link** | [Open in Colab](https://colab.research.google.com/github/vijaideen/SPECTRA-GSC2026/blob/main/SPECTRA_v26_3.ipynb) |
| **GitHub** | [vijaideen/SPECTRA-GSC2026](https://github.com/vijaideen/SPECTRA-GSC2026) |

---

*SPECTRA — an AI that makes decisions based on the laws of physics, not the biases of history.*
