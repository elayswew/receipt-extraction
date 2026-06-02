"""Models: MLP ranker, logistic baseline, and the Donut end-to-end predictor."""

import re
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.linear_model import LogisticRegression

DONUT_TASK_PROMPT = "<s_cord-v2>"


class MLPRanker(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X, y, device, epochs=50):
    """Train the MLP ranker with a positive class weight for imbalance."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    loader = data.DataLoader(data.TensorDataset(X_t, y_t), batch_size=128, shuffle=True)

    mlp = MLPRanker(input_dim=len(X[0])).to(device)
    pos = sum(y)
    pos_w = torch.tensor([(len(y) - pos) / pos] if pos > 0 else [1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    mlp.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(mlp(bx.to(device)), by.to(device))
            loss.backward()
            optimizer.step()
    return mlp


def train_logistic(X, y):
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X, y)
    return clf


def predict_mlp(cands, mlp, device):
    if not cands:
        return None
    feats = np.array([c["feats"] for c in cands], dtype=np.float32)
    mlp.eval()
    with torch.no_grad():
        scores = torch.sigmoid(mlp(torch.tensor(feats).to(device)))
        best = int(torch.argmax(scores).item())
    return cands[best]["amount_str"]


def predict_logistic(cands, clf):
    if not cands:
        return None
    feats = np.array([c["feats"] for c in cands], dtype=np.float32)
    best = int(np.argmax(clf.predict_proba(feats)[:, 1]))
    return cands[best]["amount_str"]


def predict_donut(image, processor, model, device):
    """Pretrained Donut end-to-end baseline."""
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(
        DONUT_TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            pixel_values, decoder_input_ids=decoder_input_ids, max_length=512,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    seq = processor.batch_decode(outputs.sequences)[0]
    seq = re.sub(r"<.*?>", "", seq, count=1).replace(processor.tokenizer.eos_token, "")
    try:
        return processor.token2json(seq)["total"]["total_price"]
    except Exception:
        return None
