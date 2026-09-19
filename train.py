"""Train TinyGPT on CPU — baked-in mitigations: tiny-default, subset-first, quantized export.

Usage:
  python train.py                 # quick proof-of-life: subset, 300 iters, ~minutes on CPU
  python train.py --full          # full 1MB Shakespeare, 2000 iters
  python train.py --big --full    # 2-layer/128-dim model, slower, better samples
  python train.py --iters 100     # smoke test
"""
import argparse
import json
import os
import urllib.request

import numpy as np

from model import TinyGPT

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
FALLBACK = ("To be, or not to be, that is the question. " * 200
            + "Friends, Romans, countrymen, lend me your ears. " * 200)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_text(full):
    path = os.path.join(HERE, "tinyshakespeare.txt")
    if full and not os.path.exists(path):
        try:
            print("downloading tiny-shakespeare (~1MB)…")
            urllib.request.urlretrieve(DATA_URL, path)
        except Exception as e:
            print(f"download failed ({e}); using built-in fallback text")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
    else:
        text = FALLBACK
    if not full:
        text = text[:200_000]  # subset-first: instant proof-of-life
    return text


def adam_init(params):
    return {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in params.items()}


def adam_step(params, grads, state, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, t=1, clip=1.0):
    for k in params:
        g = np.clip(grads[k], -clip, clip)
        m, v = state[k]
        m[:] = b1 * m + (1 - b1) * g
        v[:] = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        params[k] -= lr * mh / (np.sqrt(vh) + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full 1MB corpus (default: 200KB subset)")
    ap.add_argument("--big", action="store_true", help="2-layer/128-dim model (default: 1-layer/64-dim tiny)")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bpe", action="store_true",
                    help="train on BPE tokens from bpe.json (proof of word-level pipeline)")
    a = ap.parse_args()

    text = load_text(a.full)
    bpe = None
    if a.bpe:
        from bpe import encode as bpe_encode, decode as bpe_decode, get_encoder
        bpe = json.load(open(os.path.join(HERE, "bpe.json"), encoding="utf-8"))
        rank = get_encoder(bpe)
        if not a.full:
            text = text[:200_000]
        print("BPE-encoding corpus…")
        data = np.array(bpe_encode(text, bpe, rank), dtype=np.int64)
        V = len(bpe["vocab"])
        itos = None
        print(f"BPE tokens: {len(data):,} (vs {len(text):,} chars)")
    else:
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        data = np.array([stoi[ch] for ch in text], dtype=np.int64)
        V = len(chars)
    n = int(len(data) * 0.9)
    train, val = data[:n], data[n:]

    if a.big:
        model = TinyGPT(V, block_size=32, n_layer=2, n_head=4, n_embd=128, seed=a.seed)
    else:
        model = TinyGPT(V, block_size=16, n_layer=1, n_head=2, n_embd=64, seed=a.seed)
    print(f"model params: {model.n_params():,} | vocab: {V} | corpus chars: {len(text):,}")
    print(f"config: {'big' if a.big else 'tiny-default'} | data: {'full' if a.full else 'subset-first'}"
          f" | tok: {'bpe' if a.bpe else 'char'}")

    iters = a.iters or (2000 if a.full else 300)
    T = model.cfg["T"]
    B = a.batch
    rng = np.random.default_rng(a.seed)
    opt = adam_init(model.p)
    losses = []

    def batch(split):
        d = train if split == "train" else val
        ix = rng.integers(0, len(d) - T - 1, size=B)
        x = np.stack([d[i:i + T] for i in ix])
        y = np.stack([d[i + 1:i + T + 1] for i in ix])
        return x, y

    l0, _ = model.loss_and_grad(*batch("train"))
    print(f"iter 0 baseline loss {l0:.3f}")
    for t in range(1, iters + 1):
        x, y = batch("train")
        loss, grads = model.loss_and_grad(x, y)
        adam_step(model.p, grads, opt, lr=a.lr, t=t)
        losses.append(loss)
        if t % max(1, iters // 10) == 0 or t == iters:
            lv, _ = model.loss_and_grad(*batch("val"))
            print(f"iter {t}/{iters} train {loss:.3f} val {lv:.3f}")

    # samples at two temperatures
    if a.bpe:
        start, enc_name, dec = [32], "bpe", lambda ids: bpe_decode(ids, bpe)
        sample_tag, w_name = "_bpe", "weights_bpe.json"
    else:
        start, enc_name = [stoi.get("T", 0)], "char"
        dec = lambda ids: "".join(itos[i] for i in ids)
        sample_tag, w_name = "", "weights.json"
    for temp in (0.7, 1.0):
        ids = model.sample(start, 300, temperature=temp, seed=1)
        sample = dec(ids)
        with open(os.path.join(HERE, f"sample_t{temp}{sample_tag}.txt"), "w", encoding="utf-8") as f:
            f.write(sample)
        print(f"--- {enc_name} sample temp={temp} ---\n{sample[:200]}\n…")

    # quantized float16 export for the web demo (mitigation: small + lazy-loadable)
    q = {k: np.round(v.astype(np.float16).astype(np.float64), 4).tolist()
         for k, v in model.p.items()}
    out = {"cfg": {**model.cfg, "quant": "float16-rounded", "tok": enc_name},
           "chars": chars if not a.bpe else [bpe["vocab"][str(i)] for i in range(V)],
           "params": q,
           "loss": losses[-1] if losses else None,
           "loss_curve": [round(float(x), 4) for x in losses[::max(1, len(losses) // 60)]]}
    with open(os.path.join(HERE, w_name), "w", encoding="utf-8") as f:
        json.dump(out, f)
    import os as _os
    sz = _os.path.getsize(os.path.join(HERE, w_name)) / 1024
    print(f"saved {w_name} ({sz:.0f} KB, quantized) + samples. loss {losses[0]:.3f} -> {losses[-1]:.3f}")


if __name__ == "__main__":
    main()
