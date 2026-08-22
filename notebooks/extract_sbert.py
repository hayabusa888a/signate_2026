"""Sentence-Transformer (multilingual-e5-base) で文埋め込みを抽出しキャッシュ。"""
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

MODEL = "intfloat/multilingual-e5-base"
TEXT_COLS = ["今後のDX展望", "企業概要", "組織図"]

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

print("loading", MODEL, "...", flush=True)
model = SentenceTransformer(MODEL)
model.max_seq_length = 512
print("device:", model.device, flush=True)


def emb(texts):
    # e5は "passage: " プレフィックス推奨
    inp = ["passage: " + (t if isinstance(t, str) and t else " ") for t in texts]
    return model.encode(inp, batch_size=16, normalize_embeddings=True,
                        show_progress_bar=False).astype(np.float32)


for col in TEXT_COLS:
    print(f"=== {col} ===", flush=True)
    tr = emb(train[col].fillna("").tolist())
    te = emb(test[col].fillna("").tolist())
    np.save(f"../features/e5_{col}_train.npy", tr)
    np.save(f"../features/e5_{col}_test.npy", te)
    print(f"saved e5_{col} train{tr.shape} test{te.shape}", flush=True)

print("DONE e5 extraction")
