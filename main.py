"""
End-to-end experiment on CORD-v2: compare Donut (end-to-end) against a modular
PaddleOCR + ranker pipeline (logistic baseline and MLP), and report oracle recall.

Run:  python src/main.py
"""

import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
from paddleocr import PaddleOCR
import logging

from utils import amount_to_float, get_gt_total, evaluate_metrics
from features import extract_candidates
from models import train_mlp, train_logistic, predict_mlp, predict_logistic, predict_donut

logging.getLogger("ppocr").setLevel(logging.ERROR)

DONUT_MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
MAX_TRAIN_SAMPLES = 800
MAX_TEST_SAMPLES = 100


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cord = load_dataset("naver-clova-ix/cord-v2")
    processor = DonutProcessor.from_pretrained(DONUT_MODEL_ID)
    donut = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL_ID).to(device)
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    # --- Train rankers on PaddleOCR candidates ---
    print("Building training set...")
    X, y = [], []
    for item in tqdm(cord["train"], total=MAX_TRAIN_SAMPLES):
        if len(X) > MAX_TRAIN_SAMPLES * 10:
            break
        gt_v = amount_to_float(get_gt_total(item))
        if gt_v is None:
            continue
        cands = extract_candidates(item["image"], ocr)
        if any(c["value"] and abs(c["value"] - gt_v) <= 0.01 * max(1, abs(gt_v)) for c in cands):
            for c in cands:
                X.append(c["feats"])
                y.append(1 if (c["value"] and abs(c["value"] - gt_v) <= 0.01 * max(1, abs(gt_v))) else 0)
    print(f"Collected {len(X)} candidates.")

    mlp = train_mlp(X, y, device)
    logistic = train_logistic(X, y)

    # --- Evaluate all three models ---
    print("Evaluating...")
    gts, donut_preds, lr_preds, mlp_preds = [], [], [], []
    oracle, n = 0, 0
    for item in tqdm(cord["test"], total=MAX_TEST_SAMPLES):
        if n >= MAX_TEST_SAMPLES:
            break
        gt = get_gt_total(item)
        if not gt:
            continue
        gt_v = amount_to_float(gt)
        cands = extract_candidates(item["image"], ocr)
        if gt_v and any(c["value"] and abs(c["value"] - gt_v) <= 0.01 * max(1, abs(gt_v)) for c in cands):
            oracle += 1
        donut_preds.append(predict_donut(item["image"], processor, donut, device))
        lr_preds.append(predict_logistic(cands, logistic))
        mlp_preds.append(predict_mlp(cands, mlp, device))
        gts.append(gt)
        n += 1

    print("\n" + "=" * 55)
    print(f"PaddleOCR Oracle Recall: {oracle}/{n} ({oracle / n * 100:.1f}%)")
    print("=" * 55)
    for name, preds in [("Donut", donut_preds),
                        ("PaddleOCR + Logistic", lr_preds),
                        ("PaddleOCR + MLP", mlp_preds)]:
        m = evaluate_metrics(gts, preds)
        print(f"\n{name}")
        print(f"  Strict EM : {m['strict_em']:.4f}")
        print(f"  Relaxed EM: {m['relaxed_em']:.4f}")
        print(f"  MAE       : {m['mae']:.2f}")
        print(f"  Char F1   : {m['char_f1']:.4f}")


if __name__ == "__main__":
    main()
