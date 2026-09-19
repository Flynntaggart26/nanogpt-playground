"""Byte-Pair Encoding tokenizer — pure Python, no libraries.

Train once, reuse everywhere (Python training + JS playground share bpe.json):
  python bpe.py --merges 256        # quick demo tokenizer (~1-2 min on full corpus)
  python bpe.py --merges 1000       # fuller vocab for real BPE-model training
"""
import argparse
import json
import os
import re

from train import load_text

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = re.compile(r"(\s+|\w+|[^\w\s])")


def train_bpe(text, num_merges, verbose=True):
    words = [w for w in SPLIT.split(text) if w]
    splits = {w: list(w.encode("utf-8")) for w in set(words)}  # byte ids (int)
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    merges = []
    vocab = {i: bytes([i]) for i in range(256)}  # byte-level base: total coverage
    nxt = 256
    for m in range(num_merges):
        pair_counts = {}
        for w, c in counts.items():
            ids = splits[w]
            for a, b in zip(ids, ids[1:]):
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + c
        if not pair_counts:
            break
        best = max(pair_counts, key=pair_counts.get)
        merges.append(list(best))
        vocab[nxt] = vocab[best[0]] + vocab[best[1]]
        nxt += 1
        for w in splits:
            ids, out, i = splits[w], [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                    out.append(nxt - 1)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            splits[w] = out
        if verbose and (m + 1) % 50 == 0:
            tok = vocab[nxt - 1].decode("utf-8", "replace")
            print(f"merge {m + 1}/{num_merges}: '{tok}' (x{pair_counts[best]:,})")
    # id -> str for the web demo (byte tokens shown as raw chars where printable)
    vocab_s = {}
    for i, b in vocab.items():
        try:
            vocab_s[str(i)] = b.decode("utf-8")
        except UnicodeDecodeError:
            vocab_s[str(i)] = "�"
    return {"merges": merges, "vocab": vocab_s}


def get_encoder(bpe):
    """Precompute rank map: (a,b) -> merge order. Returns (rank, vocab_ids)."""
    rank = {tuple(m): i for i, m in enumerate(bpe["merges"])}
    return rank


def encode(text, bpe, rank=None):
    """Greedy BPE: repeatedly merge the lowest-ranked adjacent pair."""
    rank = rank or get_encoder(bpe)
    out = []
    for w in SPLIT.split(text):
        if not w:
            continue
        ids = list(w.encode("utf-8"))  # start from bytes: unknown-safe
        while len(ids) > 1:
            best, best_rank = None, None
            for a, b in zip(ids, ids[1:]):
                r = rank.get((a, b))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = (a, b), r
            if best is None:
                break
            new_id = 256 + best_rank
            ids2, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                    ids2.append(new_id)
                    i += 2
                else:
                    ids2.append(ids[i])
                    i += 1
            ids = ids2
        out.extend(ids)
    return out


def decode(ids, bpe):
    vocab = bpe["vocab"]
    raw = b"".join(
        vocab[str(i)].encode("utf-8", "replace") if int(i) >= 256
        else bytes([int(i)]) for i in ids
    )
    return raw.decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merges", type=int, default=256)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    text = load_text(a.full or True)
    print(f"corpus chars: {len(text):,} — learning {a.merges} merges…")
    bpe = train_bpe(text, a.merges)
    with open(os.path.join(HERE, "bpe.json"), "w", encoding="utf-8") as f:
        json.dump(bpe, f, ensure_ascii=False)
    # roundtrip + compression report
    rank = get_encoder(bpe)
    sample = text[:2000]
    ids = encode(sample, bpe, rank)
    rt = decode(ids, bpe)
    print(f"roundtrip exact: {rt == sample}")
    print(f"compression: {len(sample)} chars -> {len(ids)} tokens "
          f"({len(sample) / max(1, len(ids)):.2f} chars/token)")
    print("example:", [(i, bpe['vocab'][str(i)]) for i in ids[:12]])
    print("saved bpe.json")


if __name__ == "__main__":
    main()
