# ⚛️ VICTOR v26.6
### *Variational Inference for Confined Tokamak Output Reconstruction*

**Google Solutions Challenge 2026 · SDG 7: Affordable and Clean Energy · Open Innovation Track**

---

## 🔗 Links

| | |
|---|---|
| **🌐 Live MVP** | [https://spectra-west-3105564668.us-central1.run.app/](https://spectra-west-3105564668.us-central1.run.app/) |
| **📓 Notebook** | [VICTOR_v26_6_public.ipynb](./VICTOR_v26_6_public.ipynb) |
| **📦 Repository** | [https://github.com/vijaideen/VICTOR-GSC2026](https://github.com/vijaideen/VICTOR-GSC2026) |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vijaideen/VICTOR-GSC2026/blob/main/VICTOR_v26_6_public.ipynb)

---

## 🌍 Problem Statement

### Track: Open Innovation — Unbiased AI Decision

Fusion energy is one of humanity's most ambitious scientific endeavours — a potential source of virtually limitless, carbon-free electricity. The **WEST Tokamak** (CEA, Cadarache, France) is one of the world's leading experimental fusion reactors, actively advancing the science that will underpin ITER and future commercial fusion plants.

At the heart of safe, stable fusion operation lies **plasma diagnostics** — the ability to monitor the state of superheated plasma confined inside the tokamak in real time. One of the most critical diagnostic modalities is **Soft X-ray (SXR) tomography**: reconstructing the 2D spatial distribution of plasma emissivity from a sparse set of line-of-sight chord measurements made by camera arrays around the vessel.

**The challenge is profound:**

- The plasma emissivity field is a 2D continuous function that must be inferred from only **128 discrete chord measurements** — a severely underdetermined inverse problem.
- Classical reconstruction methods (Filtered Backprojection, Tikhonov regularisation, MFI) are either too noisy, too slow, or require careful manual tuning per experimental shot.
- Deep learning approaches traditionally require large labelled datasets of ground-truth emissivity maps — which **do not exist** for real tokamak discharges. The plasma is the unknown; you cannot label what you are trying to find.
- Real-time fusion control demands sub-millisecond inference — incompatible with iterative classical solvers that take seconds per frame.

> **The core problem:** How do you build a neural network that reconstructs plasma emissivity accurately, in real time, without ever having seen a ground-truth example?

This is not just a research question. Poor or slow plasma diagnostics directly contribute to **disruptions** — sudden losses of plasma confinement that can damage reactor components, delay experiments, and set back the timeline to commercial fusion energy. Every improvement in real-time diagnostic quality is a direct contribution to **SDG 7: Affordable and Clean Energy**.

---

## 💡 Our Solution — VICTOR v26.6

### What VICTOR Does

VICTOR is a **Physics-Informed Neural Network (PINN)** that reconstructs soft X-ray plasma emissivity on the WEST tokamak in **~0.5 milliseconds**, trained entirely **without ground-truth labels**, by learning directly from 15 physics constraints embedded in the loss function.

### Why This Aligns with Unbiased AI Decision

The Unbiased AI Decision track challenges teams to build AI systems that make decisions grounded in transparent, verifiable reasoning rather than opaque pattern matching over biased datasets. VICTOR embodies this philosophy at its core:

- **No training data bias** — VICTOR never trains on labelled plasma measurements. There are no human-annotated datasets that could encode prior biases about what a "correct" plasma reconstruction looks like. Every reconstruction is derived purely from physics.
- **Transparent decision criteria** — every output VICTOR produces can be audited against its 15 physics loss terms. A reactor operator can inspect exactly which physical constraints are satisfied and by how much.
- **Reproducible and explainable** — the physics priors (Maxwell's equations, flux surface geometry, radial monotonicity) are published scientific laws, not learned heuristics.
- **Quantifiably fair across plasma regimes** — VICTOR is validated across peaked, broad, and hollow plasma profiles and three noise levels, demonstrating consistent performance without regime-specific tuning.

This is AI that earns trust through physics, not through black-box fitting to historical data.

---

## 🧠 Technical Architecture

### Core Innovation: Ground-Truth-Free Training

VICTOR replaces the supervised loss with **15 physics-based loss terms** — mathematical expressions of what the plasma emissivity field must satisfy, derived from plasma physics theory:

| Loss Term | Physical Constraint Enforced |
|---|---|
| `L_data` | Consistency with measured SXR chord integrals (W·ε ≈ g) |
| `L_tv` | Total variation regularisation — smooth emissivity, no artefacts |
| `L_phys` | 1D radial curvature prior from TORAX Te/ne profiles |
| `L_diff` | Diffusion-like Laplacian smoothness (∇²ε ≈ 0 in bulk plasma) |
| `L_smooth2d` | 2D spatial smoothness across the reconstruction grid |
| `L_fsa` | Flux surface alignment — iso-emissivity contours follow B-field topology |
| `L_mfi` | Maximum Fisher Information regularisation |
| `L_energy` | Total radiated power consistency with bolometry |
| `L_rmono` | Radial monotonicity — emissivity decreases from core to edge |
| `L_edge` | Edge decay prior — near-zero emissivity beyond ρ = 0.9 |
| `L_neg` | Non-negativity — emissivity is a physical quantity, always ≥ 0 |
| `L_sym` | Toroidal up-down symmetry prior |
| `L_brem` | Bremsstrahlung curvature constraint in core |
| `L_abel` | Abel inversion consistency |
| `L_xcam` | Cross-camera measurement consistency |

### Network Architecture

| Component | Specification |
|---|---|
| Input | 128 normalised SXR chord measurements |
| Hidden layers | 6 × [256, 512, 512, 512, 256, 128] neurons |
| Activation | SiLU + LayerNorm |
| Output | 128-point radial emissivity profile (expanded to 128×128 2D map) |
| Parameters | ~400,000 trainable weights |
| Framework | JAX + Flax + Optax (JIT-compiled, GPU-accelerated) |

### Performance vs Classical Methods

| Method | Runtime | PSNR | Real-Time Capable? |
|---|---|---|---|
| Filtered Backprojection | ~0.1 ms | ~18 dB | ⚠️ Fast but low quality |
| JAX-Tikhonov (CG) | ~50 ms | ~22 dB | ❌ Too slow |
| JAX-MFI | ~120 ms | ~23 dB | ❌ Too slow |
| **VICTOR v26.6** | **~0.5 ms** | **~28 dB** | **✅ Real-time capable** |

### Deployment Stack

VICTOR is deployed as a live interactive web application on Google Cloud:

- **Streamlit** — live reconstruction dashboard with profile and noise toggles
- **Google Cloud Run** — containerised, autoscaling deployment
- **Vertex AI (Gemini 2.5 Flash)** — expert AI analysis of reconstruction results
- **JAX + Flax + Optax** — GPU-accelerated PINN inference

**Live MVP:** [https://spectra-west-3105564668.us-central1.run.app/](https://spectra-west-3105564668.us-central1.run.app/)

---

## 📁 Repository Structure

```
VICTOR_v26_6_public.ipynb   ← Main notebook (complete, self-contained)
README.md                   ← This file
```

The notebook is fully self-contained. Running it top-to-bottom on a Colab T4 GPU runtime will:
1. Install all dependencies
2. Generate the synthetic SXR dataset
3. Train the PINN (~5–7 minutes)
4. Evaluate and benchmark against classical methods
5. Deploy the live UI to Google Cloud Run (optional, requires GCP project)

---

## 🚀 How to Run

### Prerequisites

- Google Colab account (free tier works; **T4 GPU runtime required**)
- For cloud deployment only: a Google Cloud project with billing enabled

### Step-by-Step Instructions

#### 1. Open in Google Colab

Click the badge below or open `VICTOR_v26_6_public.ipynb` directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vijaideen/VICTOR-GSC2026/blob/main/VICTOR_v26_6_public.ipynb)

#### 2. Switch to T4 GPU Runtime

In Colab: **Runtime → Change runtime type → T4 GPU**

VICTOR requires a GPU for sub-second inference. On CPU, training will be ~50× slower.

#### 3. Run All Cells (Top to Bottom)

The cells are ordered for sequential execution:

| Cell | What It Does | Est. Time |
|---|---|---|
| **1** | Install JAX, Flax, Optax, Streamlit | ~2 min |
| **2** | Write Streamlit Live UI (`app.py`) | Instant |
| **3** | Launch local Streamlit server | Instant |
| **4** | Verify JAX + GPU | Instant |
| **5** | Define Phase 1: Forward physics simulator | Instant |
| **6** | Run Phase 1 — generate synthetic SXR dataset | ~30 sec |
| **7** | Define Phase 2: VICTOR PINN architecture | Instant |
| **8** | Train PINN (10,000 epochs, 3-stage curriculum) | ~5–7 min |
| **9** | Evaluate reconstruction — metrics + plots | ~1 min |
| **10** | Benchmark vs classical methods | ~2 min |
| **11** | Cross-noise robustness evaluation | ~1 min |
| **12** | Download all results as ZIP | Instant |
| **13** | Export trained weights for deployment | Instant |
| **14** | Write `vertex_analyst.py` (Vertex AI module) | Instant |
| **15** | Write `Dockerfile` + `requirements.txt` | Instant |
| **16** | ✏️ **Set your GCP Project ID** (edit required) | Instant |
| **17** | Authenticate with Google Cloud | ~1 min |
| **18** | Build Docker image + deploy to Cloud Run | ~8–10 min |

#### 4. Configure Google Cloud Deployment (Cell 16)

Before running Cell 16, edit the following two lines with your own values:

```python
PROJECT_ID   = 'your-gcp-project-id'    # ← Your GCP project ID
SERVICE_NAME = 'your-service-name'       # ← Your Cloud Run service name
```

Everything else (region, authentication, Vertex AI) is handled automatically.

#### 5. View Results

After Cell 8 completes, results are available in:
- **Results & Plots tab** — 2D emissivity heatmaps, radial profiles, KPI cards
- **Benchmark tab** — VICTOR vs classical methods comparison
- **Noise Robustness tab** — performance across low / medium / high noise
- **Live Console** — real-time training output

After Cell 18, the live UI is publicly accessible at your Cloud Run URL.

---

## 📊 Interpreting Results

### Key Metrics

| Metric | Description | Target (WEST SXR standard) |
|---|---|---|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) | > 25 dB considered high quality |
| **CC** | Pearson Correlation Coefficient | > 0.95 considered excellent |
| **MSE** | Mean Squared Error (2D pixel-wise) | Lower is better |
| **Inference time** | Time for a single reconstruction | < 1 ms for real-time control |

### Plasma Profile Modes

| Mode | Description | Typical Fusion Context |
|---|---|---|
| `peaked` | Sharp core peak with pedestal shoulder | H-mode — high-confinement |
| `broad` | Wider, lower core temperature profile | L-mode — standard operation |
| `hollow` | Off-axis emission ring | Reversed shear — advanced scenarios |

### Noise Levels

| Level | sigma | Approximate SNR |
|---|---|---|
| `low` | 0.001 | ~52 dB — clean detector |
| `medium` | 0.003 | ~43 dB — calibrated WEST GEM detector ★ |
| `high` | 0.008 | ~32 dB — degraded detector |

★ Medium noise matches the calibrated noise level of the WEST SXR GEM camera array.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: flax` | Re-run Cell 1 |
| `CUDA out of memory` | Reduce `grid_size` to 64 in the `CFG` dict in Cell 5 |
| Slow XLA compilation | Normal on first run — JIT cache warms up after epoch 1 |
| Streamlit not loading | Check Cell 3 output for the Colab proxy URL |
| Vertex AI error | Ensure `PROJECT_ID` is set in Cell 16 and your Cloud Run service account has the **Vertex AI User** IAM role |
| `W_matrix.npz not found` | Run Cell 6 (Phase 1) before Cell 8 (training) |
| Cloud Run deploy fails | Ensure Cloud Run API, Artifact Registry API, and Cloud Build API are all enabled in your GCP project |

---

## 📈 Impact & SDG 7 Alignment

| Impact Area | How VICTOR Contributes |
|---|---|
| **Faster disruption prediction** | Sub-ms emissivity maps enable earlier detection of MHD instabilities |
| **Reduced reactor downtime** | Better diagnostics → fewer disruptions → more plasma-on time |
| **Open science** | Fully open-source, reproducible, deployable on any tokamak geometry |
| **AI for clean energy** | Demonstrates that physics-informed AI can solve real fusion challenges |
| **No labelled data needed** | Deployable on real experimental shots from day one — no annotation pipeline required |
| **Unbiased decisions** | Physics laws, not historical datasets, drive every reconstruction |

---

## 📄 Citation

```
VICTOR v26.6 — Variational Inference for Confined Tokamak Output Reconstruction
Google Solutions Challenge 2026 · SDG 7: Affordable and Clean Energy
Open Innovation Track — Unbiased AI Decision
```

---

## 📜 Licence

Apache License 2.0 — free to use, modify, and distribute with attribution. See [LICENSE](./LICENSE) for full terms.

---

*Built with JAX · Flax · Optax · Streamlit · Google Cloud Run · Vertex AI (Gemini 2.5 Flash)*
