"""Tiny GPT from scratch — NumPy only, no ML frameworks.

Baked-in risk mitigations:
  - TINY default: 1 layer / 2 heads / 64-dim (~12K params) -> trains in minutes on CPU
  - full config available via --big (2 layers / 4 heads / 128-dim)
Hand-derived backprop. Quantized float16 export for the web demo.
"""
import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_backward(dout, x):
    c = np.sqrt(2 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    d_inner = c * (1 + 3 * 0.044715 * x ** 2)
    return dout * (0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * d_inner)


def layernorm_forward(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    return xhat * g + b, (x, xhat, mu, var, g, eps)


def layernorm_backward(dout, cache):
    x, xhat, mu, var, g, eps = cache
    E = x.shape[-1]
    dxhat = dout * g
    inv = 1.0 / np.sqrt(var + eps)
    dvar = (dxhat * (x - mu) * -0.5 * (var + eps) ** -1.5).sum(-1, keepdims=True)
    dmu = (dxhat * -inv).sum(-1, keepdims=True) + dvar * (-2 * (x - mu)).mean(-1, keepdims=True)
    dx = dxhat * inv + dvar * 2 * (x - mu) / E + dmu / E
    dg = (dout * xhat).sum(axis=(0, 1))
    db = dout.sum(axis=(0, 1))
    return dx, dg, db


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class TinyGPT:
    def __init__(self, vocab_size, block_size=16, n_layer=1, n_head=2, n_embd=64, seed=0):
        assert n_embd % n_head == 0
        rng = np.random.default_rng(seed)
        V, T, E = vocab_size, block_size, n_embd
        self.cfg = dict(V=V, T=T, L=n_layer, H=n_head, E=E, D=E // n_head)
        sc = lambda f, shape: (rng.standard_normal(shape) * f).astype(np.float64)
        self.p = {}
        self.p["wte"] = sc(0.1, (V, E))
        self.p["wpe"] = sc(0.1, (T, E))
        for i in range(n_layer):
            k = f"b{i}_"
            self.p[k + "wq"] = sc(0.1 / np.sqrt(E), (E, E))
            self.p[k + "wk"] = sc(0.1 / np.sqrt(E), (E, E))
            self.p[k + "wv"] = sc(0.1 / np.sqrt(E), (E, E))
            self.p[k + "wo"] = sc(0.1 / np.sqrt(E), (E, E))
            self.p[k + "ln1g"] = np.ones(E)
            self.p[k + "ln1b"] = np.zeros(E)
            self.p[k + "w1"] = sc(0.1 / np.sqrt(E), (E, 4 * E))
            self.p[k + "b1"] = np.zeros(4 * E)
            self.p[k + "w2"] = sc(0.1 / np.sqrt(E), (4 * E, E))
            self.p[k + "b2"] = np.zeros(E)
            self.p[k + "ln2g"] = np.ones(E)
            self.p[k + "ln2b"] = np.zeros(E)
        self.p["lng"] = np.ones(E)
        self.p["lnb"] = np.zeros(E)
        self.p["wh"] = sc(0.1 / np.sqrt(E), (E, V))
        self.p["bh"] = np.zeros(V)

    def n_params(self):
        return sum(v.size for v in self.p.values())

    def forward(self, idx):
        """idx: (B,T) int -> logits (B,T,V), cache."""
        p, cfg = self.p, self.cfg
        B, T = idx.shape
        E, H, D, L = cfg["E"], cfg["H"], cfg["D"], cfg["L"]
        c = {"idx": idx}
        x = p["wte"][idx] + p["wpe"][None, :T, :]
        for i in range(L):
            k = f"b{i}_"
            c[k + "x_in"] = x
            h, c1 = layernorm_forward(x, p[k + "ln1g"], p[k + "ln1b"])
            c[k + "ln1c"] = c1
            q = h @ p[k + "wq"]
            kk = h @ p[k + "wk"]
            v = h @ p[k + "wv"]
            qh = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
            kh = kk.reshape(B, T, H, D).transpose(0, 2, 1, 3)
            vh = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)
            att = qh @ kh.transpose(0, 1, 3, 2) / np.sqrt(D)
            mask = np.triu(np.ones((T, T)), k=1).astype(bool)
            att[:, :, mask] = -1e9
            probs = softmax(att)
            c[k + "att"] = (qh, kh, vh, probs)
            oh = probs @ vh
            oh_m = oh.transpose(0, 2, 1, 3).reshape(B, T, E)
            c[k + "ohm"] = oh_m
            o = oh_m @ p[k + "wo"]
            c[k + "h"] = h
            x = x + o
            h2, c2 = layernorm_forward(x, p[k + "ln2g"], p[k + "ln2b"])
            c[k + "ln2c"] = c2
            z1 = h2 @ p[k + "w1"] + p[k + "b1"]
            a1 = gelu(z1)
            m = a1 @ p[k + "w2"] + p[k + "b2"]
            c[k + "mlp"] = (h2, z1, a1)
            x = x + m
        hf, clf = layernorm_forward(x, p["lng"], p["lnb"])
        c["lnfc"] = clf
        c["hf"] = hf
        logits = hf @ p["wh"] + p["bh"]
        return logits, c

    def loss_and_grad(self, idx, targets):
        """Cross-entropy loss + full backward. Returns (loss, grads)."""
        p, cfg = self.p, self.cfg
        B, T = idx.shape
        E, L = cfg["E"], cfg["L"]
        logits, c = self.forward(idx)
        N = B * T
        shifted = logits - logits.max(-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(-1, keepdims=True)
        flat_t = targets.reshape(N)
        loss = -np.log(probs.reshape(N, -1)[np.arange(N), flat_t] + 1e-12).mean()
        dlogits = probs.reshape(N, -1)
        dlogits[np.arange(N), flat_t] -= 1
        dlogits = (dlogits / N).reshape(B, T, -1)
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["wh"] = c["hf"].reshape(N, E).T @ dlogits.reshape(N, -1)
        g["bh"] = dlogits.sum(axis=(0, 1))
        dhf = dlogits @ p["wh"].T
        dx, dg, db = layernorm_backward(dhf, c["lnfc"])
        g["lng"], g["lnb"] = dg, db
        for i in reversed(range(L)):
            k = f"b{i}_"
            h2, z1, a1 = c[k + "mlp"]
            dm = dx
            g[k + "w2"] += a1.reshape(N, -1).T @ dm.reshape(N, E)
            g[k + "b2"] += dm.sum(axis=(0, 1))
            dz1 = gelu_backward(dm @ p[k + "w2"].T, z1)
            g[k + "w1"] += h2.reshape(N, E).T @ dz1.reshape(N, -1)
            g[k + "b1"] += dz1.sum(axis=(0, 1))
            dh2 = dz1 @ p[k + "w1"].T
            dx_mid, dg2, db2 = layernorm_backward(dh2, c[k + "ln2c"])
            g[k + "ln2g"] += dg2
            g[k + "ln2b"] += db2
            dx = dx + dx_mid
            h = c[k + "h"]
            oh_m = c[k + "ohm"]
            do = dx
            g[k + "wo"] += oh_m.reshape(N, E).T @ do.reshape(N, E)
            doh_m = do @ p[k + "wo"].T
            H, D = cfg["H"], cfg["D"]
            doh = doh_m.reshape(B, T, H, D).transpose(0, 2, 1, 3)
            qh, kh, vh, patt = c[k + "att"]
            dprobs = doh @ vh.transpose(0, 1, 3, 2)
            dvh = patt.transpose(0, 1, 3, 2) @ doh
            s = (dprobs * patt).sum(-1, keepdims=True)
            datt = patt * (dprobs - s) / np.sqrt(D)
            dqh = datt @ kh
            dkh = datt.transpose(0, 1, 3, 2) @ qh
            dq = dqh.transpose(0, 2, 1, 3).reshape(B, T, E)
            dk = dkh.transpose(0, 2, 1, 3).reshape(B, T, E)
            dv = dvh.transpose(0, 2, 1, 3).reshape(B, T, E)
            g[k + "wq"] += h.reshape(N, E).T @ dq.reshape(N, E)
            g[k + "wk"] += h.reshape(N, E).T @ dk.reshape(N, E)
            g[k + "wv"] += h.reshape(N, E).T @ dv.reshape(N, E)
            dh = dq @ p[k + "wq"].T + dk @ p[k + "wk"].T + dv @ p[k + "wv"].T
            dx_att, dg1, db1 = layernorm_backward(dh, c[k + "ln1c"])
            g[k + "ln1g"] += dg1
            g[k + "ln1b"] += db1
            dx = dx + dx_att
        np.add.at(g["wte"], c["idx"], dx)
        g["wpe"] += dx.sum(axis=0)[: p["wpe"].shape[0]]
        return float(loss), g

    def sample(self, start_ids, steps, temperature=1.0, seed=0):
        """Greedy/temperature sampling. start_ids: list[int]."""
        rng = np.random.default_rng(seed)
        T = self.cfg["T"]
        ids = list(start_ids)
        for _ in range(steps):
            ctx = np.array([ids[-T:]], dtype=np.int64)
            logits, _ = self.forward(ctx)
            z = logits[0, -1] / max(temperature, 1e-6)
            z = z - z.max()
            e = np.exp(z)
            pr = e / e.sum()
            if temperature < 1e-3:
                nxt = int(np.argmax(pr))
            else:
                nxt = int(rng.choice(len(pr), p=pr))
            ids.append(nxt)
        return ids
