# 🧾 AutoDoc — Receipt Total Extraction (PaddleOCR + MLP Ranker)

A lightweight, modular alternative to end-to-end document transformers (e.g. Donut)
for extracting the **Grand Total** from receipt images. PaddleOCR generates numeric
candidates; a small MLP ranks them using an engineered feature vector that combines
semantic, numeric, and geometric signals.

Built and evaluated on the [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) dataset.

## Why this approach

End-to-end transformers like Donut reach ~92% accuracy but are GPU-heavy and costly to
deploy. This project asks whether a cheaper OCR + ranking pipeline can get close enough
to be viable for real accounting workflows, where a confidence-based human-in-the-loop
step can cover the remaining gap.

## Pipeline

1. **PaddleOCR** detects text and numbers on the receipt
2. **Regex** isolates numeric candidates (potential totals)
3. Each candidate gets a feature vector capturing:
   - **Numeric:** log-amount, decimal format, digit length
   - **Semantic:** "total" keyword, negative keyword (tax/change/subtotal)
   - **Geometric:** x/y position, distance to the nearest "total" label
4. An **MLP ranker** scores candidates; the highest score is the predicted total

## Results (CORD-v2 test set)

| Model | OCR | Ranker | Strict EM |
|-------|-----|--------|-----------|
| Donut (SOTA) | end-to-end | transformer | 92.6% |
| **Proposed** | **PaddleOCR** | **MLP** | **~64–68%** |
| Baseline | PaddleOCR | Logistic Reg. | 45.3% |
| Baseline | EasyOCR | MLP | 35.8% |

Key findings:
- **OCR quality is the bottleneck.** Oracle Recall — the upper bound where the correct
  total appears among candidates — rose from 62.1% (EasyOCR) to **85.3% (PaddleOCR)**.
- **The MLP beats the logistic baseline by ~20%**, showing non-linear ranking matters.
- **Geometric features help.** An ablation adding spatial features (x-axis alignment,
  distance to keywords) improved accuracy over a purely numeric/semantic set.

## Run

```bash
pip install -r requirements.txt
python src/main.py
```

A GPU is recommended (PaddleOCR + Donut + training). The original experiments ran on
Colab with a T4. Exact numbers vary slightly with PaddleOCR version and random
initialization.

## Project Structure

| File | Purpose |
|------|---------|
| `src/utils.py` | Amount parsing, numeric conversion, evaluation metrics |
| `src/features.py` | PaddleOCR candidate extraction + feature engineering |
| `src/models.py` | MLP ranker, logistic baseline, Donut predictor |
| `src/main.py` | End-to-end training, evaluation, three-model comparison |

## Limitations & Future Work

- Capped by OCR recall (85.3%); remaining errors are ranker selection mistakes on hard
  distractors (subtotal, cash, change).
- Future: layout-aware models (GNN over text boxes), locale-aware number normalization,
  and a confidence-threshold router for human-in-the-loop deployment.
