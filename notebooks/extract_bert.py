"""日本語BERTでテキスト3列の埋め込み(mean pooling, 768次元)を抽出しキャッシュ。
labelを使わない決定的処理なのでtrain/test全体で計算してOK。1回だけ実行。
"""
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

sys.stdout.reconfigure(encoding="utf-8")

MODEL = "cl-tohoku/bert-base-japanese-v3"
TEXT_COLS = ["今後のDX展望", "企業概要", "組織図"]
MAX_LEN = 256
BATCH = 16

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

print("loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("device:", device, flush=True)


@torch.no_grad()
def embed(texts):
    vecs = []
    for i in range(0, len(texts), BATCH):
        batch = [t if isinstance(t, str) and t else " " for t in texts[i:i + BATCH]]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc).last_hidden_state              # (B, L, 768)
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        summed = (out * mask).sum(1)
        cnt = mask.sum(1).clamp(min=1e-9)
        mean = (summed / cnt).cpu().numpy()               # masked mean pooling
        vecs.append(mean)
        if (i // BATCH) % 10 == 0:
            print(f"  {i}/{len(texts)}", flush=True)
    return np.vstack(vecs).astype(np.float32)


for col in TEXT_COLS:
    print(f"\n=== {col} ===", flush=True)
    tr = embed(train[col].fillna("").tolist())
    te = embed(test[col].fillna("").tolist())
    safe = col.replace("/", "_")
    np.save(f"../features/bert_{safe}_train.npy", tr)
    np.save(f"../features/bert_{safe}_test.npy", te)
    print(f"saved bert_{safe}_train.npy {tr.shape} / test {te.shape}", flush=True)

print("\nDONE bert extraction")
