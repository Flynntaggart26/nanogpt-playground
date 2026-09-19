# NanoGPT Playground

> A character-level GPT built **from mathematics** — a NumPy-only transformer with hand-derived backprop, byte-pair tokenization, and a live browser playground that runs the real exported weights. No PyTorch. No API keys. No black boxes.

[![Live Demo](https://img.shields.io/badge/demo-live-success?style=flat-square)](https://flynntaggart26.github.io/nanogpt-playground/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Flynntaggart26/nanogpt-playground/blob/main/train_colab.ipynb)
[![Dependencies](https://img.shields.io/badge/dependencies-numpy--only-blue?style=flat-square)](./model.py)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](./LICENSE)

**[▶ Live demo — https://flynntaggart26.github.io/nanogpt-playground/](https://flynntaggart26.github.io/nanogpt-playground/)**
· [60-second demo](#-60-second-demo-script) · [Architecture](#-architecture-deep-dive) · [Training results](#-training-results)
· [Verification](#-verification--debugging) · [BPE tokenizer](#-bpe-tokenizer) · [Decoding](#-decoding-temperature--top-k--top-p)
· [Run locally](#-run-locally) · [Playground guide](#-playground-guide)

---

## 📌 Overview

Every week a new team demos an app *wrapped around* someone else's LLM API. This project goes the opposite direction: **it builds the model itself**, from the embedding lookup to the last gradient, and then lets you play with the finished artifact in your browser.

Concretely, this repository contains:

1. **`model.py`** — a complete decoder-only transformer in ~200 lines of NumPy: token + positional embeddings, causal multi-head self-attention, GELU MLP blocks, layer normalization, and a full hand-derived backward pass with Adam. The only import is `numpy`.
2. **`train.py`** — a CPU training harness (train/val split, batching, loss curves, temperature samples, quantized export) with three operating modes: 3-minute proof-of-life, full-corpus training, and word-level BPE training.
3. **`bpe.py`** — a byte-pair encoding tokenizer in pure Python (byte-level base vocabulary, greedy lowest-rank merges), sharing one `bpe.json` artifact with the browser.
4. **`index.html`** — a zero-dependency playground that re-implements the forward pass *and* the BPE encoder in JavaScript, verified numerically identical to Python, with live generation, per-head attention heatmaps, a BPE token inspector, and the real training loss curve.
5. **Checkpoints + proof** — real trained weights (`weights.json`), a 35 KB micro model for instant first paint (`fallback_weights.json`), BPE artifacts, sample outputs, and a one-click Colab notebook.

If you are evaluating this for university admission (or hiring): the interesting file is `model.py:loss_and_grad`. Everything else exists to prove that function correct — gradient checks, overfit tests, cross-language verification, and a demo you can touch.

## ✨ Features

| # | Capability | Detail |
|---|------------|--------|
| 1 | 🧮 **GPT from scratch** | Decoder-only transformer, NumPy only. Tiny-default: 1 layer / 2 heads / 64-dim (~54–59K params, context 16). `--big`: 2 layers / 4 heads / 128-dim, context 32 |
| 2 | 📉 **Real CPU training** | Adam (β₁=0.9, β₂=0.999, grad clip 1.0), 90/10 train/val split. Full Shakespeare run: loss `4.19 → 1.94` over 2000 iters |
| 3 | 🌐 **Browser playground** | JS forward pass verified **bit-exact** vs Python (`3.55e-15` max diff). Prompt → generate, temperature / top-k / top-p controls, per-head attention heatmap, training loss chart |
| 4 | 🔪 **BPE tokenizer** | `bpe.py`: 256 merges, vocab 512, byte-level base (total coverage). 1.62 chars/token, exact roundtrip on 10 KB. Playground token inspector renders any string as colored token chips — JS encoder verified **byte-identical** to Python on 8 adversarial strings |
| 5 | 🔬 **Word-level pipeline** | `train.py --bpe` trains a real model on BPE tokens end-to-end: loss `6.26 → 4.05`, samples decode to emerging English words |
| 6 | 🎛️ **Modern decoding** | Temperature, top-k, and nucleus (top-p) sampling in *both* Python and JS, with kept-set parity verified across languages |
| 7 | ⚡ **Instant demo** | Full weights lazy-load via `fetch`; micro fallback keeps first paint instant; shareable, works offline after load |
| 8 | 📓 **One-click Colab** | `train_colab.ipynb`: clone → smoke test → `--big` training → loss plot → checkpoint download. Runtime → Run all |
| 9 | ✅ **Verification culture** | Gradient checks, overfit tests, determinism tests, export shape validation, multi-config smoke tests — every claim below has a number attached |

## 🎤 60-Second Demo Script

For presentations and interviews:

1. **"No frameworks" (10s)** — open `model.py`, show the imports: only `numpy`. Scroll to `loss_and_grad`: *"Every gradient here is hand-derived — attention backward, GELU backward, LayerNorm backward."*
2. **Generate (30s)** — playground, prompt `ROMEO:`, temperature 0.7 → Shakespeare-flavored continuation. Crank temperature to 1.5 → gibberish. Drop to 0 → greedy, deterministic. *"Temperature reshapes the same distribution the model learned."* Then set top-k 5: *"Same model, sharper choices — watch the kept-count readout."*
3. **Attention (20s)** — point at the head heatmap: the last character attends to recent context, never the future. *"That's the causal mask. This is what 'attention' literally is — a weighted average, and you can see the weights."*

**If you have 2 minutes**, add act 4: paste `unbelievable ROMEO:` into the **BPE inspector**. *"Real models don't see characters — they see tokens. `unbelievable` becomes `un + believ + able`-style chunks. My encoder produces byte-identical output to the Python one, and I can prove it."*

## 🧠 Architecture Deep-Dive

Pipeline per training step:

```
text → tokenizer (char ids or BPE ids)
     → [token embed + learned pos embed]              (V×E) + (T×E)
     → ×L blocks: LN → masked MHA → +residual
                  LN → GELU-MLP (4E) → +residual
     → final LN → linear head (E×V) → logits → cross-entropy vs shifted targets
```

### Forward pass (shapes for B=Batch, T=context, E=dim, H=heads, D=E/H)

| Stage | Operation |
|-------|-----------|
| Embed | `x = Wte[idx] + Wpe` → `(B,T,E)` |
| Norm | Per-token LayerNorm, learned γ/β, ε=1e-5 |
| QKV | `Q,K,V = h·Wq, h·Wk, h·Wv`, reshaped to `(B,H,T,D)` |
| Attention | `softmax(QKᵀ/√D + causal_mask)` → `(B,H,T,T)`; causal mask = `-1e9` above diagonal |
| Merge | Heads concatenated → `(B,T,E)`, projected by `Wo`, residual add |
| MLP | `GELU(h·W1+b1)·W2+b2` with 4E hidden width, residual add |
| Head | Final LayerNorm → `logits = h·Wh+bh` → `(B,T,V)` |

### Backward pass (`TinyGPT.loss_and_grad`)

Reverse-mode through the whole graph in one method: softmax-cross-entropy → head → final norm → each block (MLP branch with GELU backward, attention branch with softmax-Jacobian backward `P⊙(dP − Σ)` and Q/K/V paths) → both LayerNorms → token/positional embedding accumulation via `np.add.at`. Parameter count is exact — no hidden state, no autograd, no framework.

### Model sizes

| Config | Layers / Heads / Dim / Ctx | Vocab | Params |
|--------|---------------------------|-------|--------|
| tiny-default (char, full) | 1 / 2 / 64 / 16 | 65 | 59,265 |
| `--big` (char) | 2 / 4 / 128 / 32 | 65 | ~400K |
| tiny-default (BPE proof) | 1 / 2 / 64 / 16 | 512 | 116,928 |
| micro fallback (web) | 1 / 1 / 16 / 8 | 23 | 4,135 (35 KB) |

### Optimizer

Adam, learning rate `3e-3` (tiny) — β₁=0.9, β₂=0.999, ε=1e-8, per-tensor gradient clipping at 1.0. Batch 16, reshuffled random windows every step (a form of data augmentation for free).

## 📉 Training Results

All runs on CPU, all reproduced from the scripts in this repo:

| Run | Command | Corpus | Result |
|-----|---------|--------|--------|
| Proof-of-life | `python train.py` | 200 KB subset | `3.14 → 0.13` (300 iters, minutes) |
| Full char | `python train.py --full` | 1.1M chars Shakespeare, vocab 65 | `4.19 → 1.94` (2000 iters, val tracks train) |
| BPE proof | `python train.py --bpe` | 200 KB → 124K BPE tokens, vocab 512 | `6.26 → 4.05` (150 iters; baseline ln 512 ≈ 6.24) |

Sample evolution tells the story honestly: the subset run memorizes its repetitive fallback text; the full run produces Early-Modern flavor with speaker labels (`XLANUS:`, `ROMEO:`-style turns); the BPE run at 150 iters emits emerging real words (`the`, `and`, `you`, `have`) inside word-salad — exactly what a small model early in training should do. Samples are saved as `sample_*.txt` at temperatures 0.7 and 1.0.

> **Reproducibility note:** the environment used for development was briefly offline, so early checkpoints trained on a built-in fallback excerpt. The shipped `weights.json` is the full-Shakespeare run. Anyone can reproduce or beat it with one command.

## 🔪 BPE Tokenizer

Real LLMs don't read characters — they read *tokens*. `bpe.py` implements byte-pair encoding from first principles:

- **Base vocabulary:** all 256 bytes → total coverage, so *any* string (emoji, accents, code) encodes without `<unk>` failures.
- **Training:** count adjacent pairs across the corpus, merge the most frequent, repeat 256 times. Early merges learn `e`, `th`, `he`; later ones learn `ing`, `tion`-style chunks and full common words.
- **Encoding:** split text into word-ish pieces, then greedily merge the lowest-ranked available pair until convergence. Deterministic by construction.
- **Artifact:** one `bpe.json` (9.6 KB) shared by Python training and the JS inspector — a single source of truth across languages.

Measured quality: **1.62 chars/token** on held-out Shakespeare (2000 chars → 1233 tokens), **exact roundtrip** on 10 KB including edge cases (empty string, mixed whitespace, emoji, accented Latin). The playground's JS encoder reproduces Python's output **byte-identically on all 8 adversarial test strings**.

## 🎛️ Decoding: Temperature / Top-k / Top-p

Training learns a distribution; decoding decides how bravely to sample from it. Both languages implement the same pipeline (verified identical kept-sets on 8 cases including `k > vocab`, `k = 1`, `p = 0.05`):

1. **Temperature** divides logits before softmax. `T → 0` = greedy argmax (deterministic); `T = 0.7` = balanced; `T ≥ 1.2` = adventurous gibberish. Same weights, different personalities.
2. **Top-k** keeps only the k highest-logit tokens, then renormalizes. Cuts the long tail of nonsense.
3. **Top-p (nucleus)** keeps the smallest set whose cumulative probability ≥ p. Adaptive: confident predictions sample narrowly, uncertain ones broadly.

The playground shows a live readout — `[sampling from top N/V]` — so audiences *see* the filter working. Guidance for demos: `T=0.7, k=0, p=1.0` (pure), `T=1.0, k=5` (focused), `T=1.0, p=0.9` (classic nucleus feel).

## ✅ Verification & Debugging

Nothing here is claimed without a number. The full debug history:

| Check | Method | Result |
|-------|--------|--------|
| Backprop correctness | Finite-difference gradient check, 10 tensors | Worst rel. err `9.23e-06` → **PASS** |
| Optimizer health | Single-batch overfit | `2.328 → 0.0002` → **PASS** |
| JS forward mirror | Same weights, same input, both languages | Max diff `3.55e-15` → **PASS** |
| Quantization effect | Full-precision vs float16-rounded weights | `1.8e-2` gap, ordering-preserving (expected, harmless for sampling) |
| Sampling determinism | Same seed twice | Identical → **PASS** |
| Config robustness | 1-layer/16-dim through 2-layer/128-dim | Grad shapes exact → **PASS** |
| BPE merge causality | Every merge references only older tokens | 256/256 → **PASS** |
| BPE vocab consistency | 30 sampled entries = byte-concat of pair | → **PASS** |
| BPE roundtrip | 10 KB + emoji/accents/whitespace edges | Exact → **PASS** |
| BPE JS parity | 8 adversarial strings | Byte-identical → **PASS** |
| Top-k/p parity | 8 cases incl. `k=1`, `k>vocab`, `p=0.05` | Identical kept sets → **PASS** |
| Filter edges | k=0/negative, p=0/>1, in both languages | Always ≥1 valid id, distributions sum to 1 → **PASS** |
| Export integrity | Checkpoint shapes vs fresh model | Exact (`weights.json`, `weights_bpe.json`) → **PASS** |
| Char-path regression | After BPE refactor, 10-iter smoke run | Loss drops; shipped checkpoint restored byte-for-byte → **PASS** |

Bugs actually caught by this process (not hypothesized): a key-type bug in BPE training (character strings vs byte ids), a missing parenthesis in the JS GELU, and one Windows console-encoding red herring. The harness files live in `/tmp`, never in the repo.

## 📁 Project Structure

```
nanogpt-playground/
├── model.py               # TinyGPT: forward + hand-derived backward + kept_indices + sampling
├── train.py               # modes: proof-of-life / --full / --big / --bpe; quantized export
├── bpe.py                 # learn merges, byte-safe encode/decode
├── index.html             # playground: generation + attention viz + loss chart + BPE inspector
├── train_colab.ipynb      # one-click free-GPU reproduction
├── weights.json           # full-Shakespeare checkpoint (quantized, lazy-loaded)
├── fallback_weights.json  # 35 KB micro model for instant first paint
├── bpe.json               # 256 merges, vocab 512 (Python ↔ JS shared)
├── weights_bpe.json       # word-level proof checkpoint (not used by playground)
└── sample_*.txt           # proof outputs (char + BPE, temps 0.7 / 1.0)
```

Total shipped code: `model.py` (~200 lines) + `train.py` + `bpe.py` + `index.html` — readable in one sitting, which is the point.

## 🚀 Run Locally

Prerequisites: Python 3 + `pip install numpy`. Nothing else — no torch, no CUDA, no API keys.

```bash
git clone https://github.com/Flynntaggart26/nanogpt-playground.git
cd nanogpt-playground
pip install numpy

# 1. proof-of-life: tiny model, subset data, minutes on CPU
python train.py

# 2. the real thing: full 1MB Shakespeare (needs internet once for download)
python train.py --full

# 3. deeper model, better samples
python train.py --big --full

# 4. word-level: learn tokenizer, then train on its tokens
python bpe.py --merges 256
python train.py --bpe

# 5. playground (needs HTTP for fetch(); file:// blocks it)
python -m http.server   # → http://localhost:8000
```

Free GPU path: open `train_colab.ipynb` in Colab → **Runtime → Run all**. It clones the repo, smoke-tests, trains `--big`, plots the curve, and hands you a `weights.json` to download.

## 🧭 Playground Guide

| Panel | What to do | What to notice |
|-------|-----------|----------------|
| Generate | Prompt `To be, o`, Generate → | Continuation in period flavor; prompt echoed in white, generation in teal |
| Temperature | 0 → 0.7 → 1.5 | Deterministic → balanced → unhinged (same weights!) |
| Top-k / Top-p | k=5 or p=0.9 at T=1.0 | Kept-count readout shrinks; quality sharpens |
| Attention | After any generation | Per-head last-token weights; upper triangle always empty (causality) |
| Loss chart | — | The actual training curve from `weights.json`, first→final loss |
| BPE inspector | `unbelievable ROMEO:` → Tokenize | Words fracture into learned chunks with token ids + compression ratio |

Keyboard-free, mobile-friendly, no build step. If weights fail to load you opened it via `file://` — serve over HTTP.

## 🧩 Design Decisions (and Honest Trade-offs)

- **NumPy, not PyTorch** — the entire pedagogical value is in the handwritten backward pass. A framework would delete the project's reason to exist.
- **Char-level default, BPE as proof** — char models train to coherence in minutes on CPU; BPE at this scale needs far more compute for equal fluency. Both pipelines exist; the playground demos the fluent one and *inspects* the sophisticated one. No cherry-picking: the BPE samples in the repo are the real 150-iter outputs, salad and all.
- **float16-rounded export** — halves checkpoint size for web loading at a measured `1.8e-2` logit cost that doesn't change sampling order. Verified, not assumed.
- **Lazy-load + micro fallback** — the demo never stares at a spinner: 35 KB of real (tiny) weights paint instantly, full weights stream in behind.
- **Single shared `bpe.json`** — one tokenizer artifact, two languages, parity-tested. Drift between training and demo tokenization is impossible by construction.

## ⚠️ Limitations

- Tiny models memorize more than they generalize; Shakespeare pastiche, not understanding. Stated plainly so nobody mistakes fluency for reasoning.
- No validation-based early stopping or learning-rate schedule — fixed Adam LR, fixed iters. Good enough at this scale, would matter at larger scale.
- BPE merges learned on Shakespeare alone — a general-English corpus would give better tokens (README of a future version).
- The JS forward pass is O(T²·E) with plain loops — fine for T≤32, not a serving engine and never claimed to be.

## 🗺️ Roadmap

- [x] NumPy GPT + verified backprop + web playground
- [x] Top-k / top-p sampling in Python + playground (parity-verified)
- [x] BPE tokenizer + inspector + word-level training proof
- [x] Colab notebook → [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Flynntaggart26/nanogpt-playground/blob/main/train_colab.ipynb)
- [ ] BPE-native playground generation (swap char checkpoint for a fully-trained BPE model)
- [ ] 1K-merge BPE on general English + longer `--big` run
- [ ] Learning-rate schedule + early stopping in `train.py`

## 📚 References

- Vaswani et al., *Attention Is All You Need*, 2017 — the architecture
- Ongaro & Ousterhout-style clarity as aspiration; Karpathy's *NanoGPT* / *Let's build GPT* — the tiny-config philosophy and char-level starting point
- HuggingFace *minbpe* (Karpathy) — the greedy BPE reference approach mirrored in `bpe.py`

## 👤 Author

Built by **Flynn Taggart** — CS applicant portfolio exploring computer science from three directions: distributed systems ([raft-lab](https://github.com/Flynntaggart26/raft-lab)) · data visualization ([Climate & Energy](https://github.com/Flynntaggart26/Climate-energy)) · machine learning from scratch (this repo).

## 📄 License

MIT — free for classroom and interview use. Training corpus: tiny-Shakespeare (public domain, via Karpathy's char-rnn mirror).
