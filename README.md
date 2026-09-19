# NanoGPT Playground

> A character-level GPT built **from math** — NumPy-only transformer with hand-derived backprop, plus a live browser playground that runs the real exported weights. No frameworks, no API.

[![Live Demo](https://img.shields.io/badge/demo-live-success?style=flat-square)](https://flynntaggart26.github.io/nanogpt-playground/)
[![Dependencies](https://img.shields.io/badge/dependencies-numpy--only-blue?style=flat-square)](./model.py)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](./LICENSE)

**[▶ Live demo — https://flynntaggart26.github.io/nanogpt-playground/](https://flynntaggart26.github.io/nanogpt-playground/)** · [Demo script](#-60-second-demo-script) · [Architecture](#-architecture) · [Verified](#-verified-correct) · [Run](#-run-locally)

---

## 📌 Overview

Everyone demos LLM *apps*. This project builds the model itself: token embeddings, causal multi-head self-attention, MLP blocks, layer norm, Adam — every gradient hand-derived in `model.py`, numerically verified against finite differences (worst rel. error `9e-06`).

Train on CPU in minutes, then open the playground: type a prompt, set temperature, and watch *your* weights generate text with a live per-head attention heatmap.

## ✨ Features

| Capability | Detail |
|------------|--------|
| 🧮 **GPT from scratch** | 1-layer / 2-head / 64-dim tiny-default (~54K params); `--big` 2-layer / 128-dim. Char-level, context 16–32 |
| 📉 **Real training** | `train.py`: Adam, train/val split, loss curve, temperature samples — `3.14 → 0.13` on CPU |
| 🌐 **Web playground** | `index.html` re-implements the forward pass in JS (bit-exact vs Python: `3.55e-15`), prompt → generate, temperature slider, attention heatmap, loss chart |
| ⚡ **Instant demo** | Full weights lazy-load via `fetch`; 35 KB micro fallback keeps first paint instant; works offline after load |
| 🔬 **Debug suite** | Gradient check, single-batch overfit (`2.33 → 0.0002`), export shape validation, sampling determinism, multi-config smoke tests |

## 🎤 60-Second Demo Script

1. **"No frameworks"** — show `model.py` imports: only `numpy`. Point at `loss_and_grad`.
2. **Generate** — type `To be, o`, temperature 0.7 → coherent continuation. Crank to 1.5 → gibberish. Drop to 0 → greedy. *"Temperature reshapes the same distribution."*
3. **Attention** — point at head heatmap: last char attends to recent context, never the future (causal mask). *"This is what 'attention' literally is."*

## 🧠 Architecture

```
text → char tokenizer → [embed + pos] → [LN → masked MHA → +resid → LN → GELU-MLP → +resid] ×L → LN → logits → CE loss
```

- **Attention:** scaled dot-product, causal `-1e9` mask, H heads split/merged explicitly
- **Norm:** per-token LayerNorm (γ/β learned), ε=1e-5 — identical constants in Python and JS
- **Optimizer:** Adam (β₁=0.9, β₂=0.999), grad clip 1.0
- **Export:** float16-rounded JSON (`weights.json`) + loss curve; JS mirror verified bit-exact

## ✅ Verified Correct

- Gradient check vs finite differences: max rel. err `9.23e-06` → **PASS**
- Single-batch overfit `2.328 → 0.0002` → **PASS**
- JS forward vs Python on same weights: `3.55e-15` → **PASS** (the `1.8e-2` gap vs full precision is pure float16 quantization, ordering-preserving)
- `--big` + edge configs: grad shapes OK → **PASS**

## 📁 Project Structure

```
nanogpt-playground/
├── model.py               # TinyGPT: forward + hand-derived backward + sampling (NumPy only)
├── train.py               # CPU training: subset-first, Adam, quantized export, samples
├── index.html             # Playground: JS inference + attention viz + loss chart
├── weights.json           # trained checkpoint (quantized, lazy-loaded)
├── fallback_weights.json  # 35 KB micro model for instant first paint
└── sample_*.txt           # proof outputs at temperatures 0.7 / 1.0
```

## 🚀 Run Locally

```bash
git clone https://github.com/Flynntaggart26/nanogpt-playground.git
cd nanogpt-playground
pip install numpy
python train.py              # quick proof-of-life (~minutes on CPU)
python train.py --full       # full 1MB Shakespeare (needs internet once)
python train.py --big --full # deeper model, better samples
python -m http.server        # → http://localhost:8000
```

## 🗺️ Roadmap

- [x] NumPy GPT + verified backprop + web playground
- [ ] Top-k / top-p sampling controls in playground
- [ ] BPE tokenizer + word-level demo
- [ ] Colab GPU notebook for `--big` training

## 📚 References

- Vaswani et al., *Attention Is All You Need*, 2017
- Karpathy, *NanoGPT* / *Let's build GPT* — inspiration for the tiny-config philosophy

## 👤 Author

Built by **Flynn Taggart** — CS applicant portfolio: distributed systems (raft-lab) + data viz (Climate & Energy) + ML from scratch (this).

## 📄 License

MIT — free for classroom and interview use. Corpus: tiny-Shakespeare (public domain).
