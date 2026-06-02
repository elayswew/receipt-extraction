"""PaddleOCR candidate extraction with an 8-dimensional feature vector."""

import re
import math
import numpy as np

from utils import amount_to_float

TOTAL_KEYWORDS = ["total", "amount due", "grand total", "balance due", "amount", "total price"]
NEGATIVE_KEYWORDS = ["tax", "change", "cash", "subtotal", "discounts", "vis", "mastercard"]
NUM_PATTERN = re.compile(r"[-+]?\d[\d,.\s]*\d")


def extract_candidates(image, ocr_engine):
    """Run PaddleOCR and build an 8-dim feature vector per numeric candidate."""
    img = np.array(image.convert("RGB"))
    h, w, _ = img.shape
    result = ocr_engine.ocr(img, cls=True)
    if not result or result[0] is None:
        return []
    lines = result[0]

    # Vertical positions of lines containing a "total" keyword
    kw_y = []
    for coords, (text, conf) in lines:
        t = str(text).strip().lower()
        ys = [p[1] for p in coords]
        if any(k in t for k in TOTAL_KEYWORDS):
            kw_y.append((min(ys) + max(ys)) / 2.0 / h)

    candidates = []
    for coords, (text, conf) in lines:
        t = str(text).strip()
        matches = list(NUM_PATTERN.finditer(t))
        if not matches:
            continue

        xs, ys = [p[0] for p in coords], [p[1] for p in coords]
        xc = (min(xs) + max(xs)) / 2.0 / w
        yc = (min(ys) + max(ys)) / 2.0 / h

        min_dist = 1.0
        for ky in kw_y:
            if abs(yc - ky) < abs(min_dist):
                min_dist = yc - ky

        contains_kw = any(k in t.lower() for k in TOTAL_KEYWORDS)
        contains_neg = any(k in t.lower() for k in NEGATIVE_KEYWORDS)

        for m in matches:
            amt = m.group(0)
            val = amount_to_float(amt)
            feats = [
                math.log1p(abs(val)) if val is not None else 0.0,  # numeric: log amount
                yc,                                          # geometric: y position
                xc,                                          # geometric: x position
                float(contains_kw),                          # semantic: total keyword
                float(contains_neg),                         # semantic: negative keyword
                float("." in amt or "," in amt),             # numeric: decimal format
                len(re.sub(r"\D", "", amt)) / 10.0,          # numeric: digit length
                min_dist,                                    # geometric: dist to total label
            ]
            candidates.append({"amount_str": amt, "value": val, "feats": feats})
    return candidates
