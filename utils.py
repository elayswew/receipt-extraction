"""Amount parsing, numeric conversion, and evaluation metrics."""

import re
import math
import json
from decimal import Decimal
from collections import Counter

MAX_ABS_AMOUNT = 1e9


def normalize_amount_str(s):
    if not s:
        return None
    s = str(s).strip().replace(" ", "")
    s = re.sub(r"[^\d,.\-]", "", s)
    if len(re.sub(r"\D", "", s)) == 0:
        return None
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", "") if len(s.split(",")[-1]) == 3 else s.replace(",", ".")
    if s.count(".") > 1:
        first = s.find(".")
        s = s[:first + 1] + s[first + 1:].replace(".", "")
    return s


def amount_to_float(s):
    s = normalize_amount_str(s)
    if not s:
        return None
    try:
        val = float(Decimal(s))
    except Exception:
        return None
    if not math.isfinite(val) or abs(val) > MAX_ABS_AMOUNT:
        return None
    return val


def get_gt_total(item):
    """Extract ground-truth total_price from a CORD-v2 item."""
    try:
        gt = item["ground_truth"]
        if isinstance(gt, str):
            gt = json.loads(gt)
        return gt["gt_parse"]["total"]["total_price"]
    except Exception:
        return None


def char_f1(gt, pred):
    if not gt or not pred:
        return 0.0
    c_gt, c_pred = Counter(str(gt)), Counter(str(pred))
    overlap = sum(min(c_gt[k], c_pred[k]) for k in c_gt)
    p, r = overlap / len(str(pred)), overlap / len(str(gt))
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def evaluate_metrics(y_true, y_pred):
    """Strict/Relaxed exact-match, MAE, RMSE, char-level F1."""
    strict, relaxed, total = 0, 0, 0
    vals_gt, vals_pred, f1s = [], [], []

    for gt, pred in zip(y_true, y_pred):
        if gt is None:
            continue
        total += 1
        f1s.append(char_f1(gt, pred) if pred else 0.0)
        v_gt = amount_to_float(gt)
        v_pred = amount_to_float(pred) if pred else None

        if pred and normalize_amount_str(gt) == normalize_amount_str(pred):
            strict += 1
            relaxed += 1
            if v_gt is not None:
                vals_gt.append(v_gt)
                vals_pred.append(v_gt)
            continue

        if v_gt is not None and v_pred is not None:
            vals_gt.append(v_gt)
            vals_pred.append(v_pred)
            if abs(v_gt - v_pred) <= 0.01 * max(1.0, abs(v_gt)):
                relaxed += 1

    mae = rmse = 0.0
    if vals_gt:
        diffs = [abs(a - b) for a, b in zip(vals_gt, vals_pred)]
        mae = sum(diffs) / len(diffs)
        rmse = math.sqrt(sum(d ** 2 for d in diffs) / len(diffs))

    return {
        "n": total,
        "strict_em": strict / total if total else 0.0,
        "relaxed_em": relaxed / total if total else 0.0,
        "mae": mae,
        "rmse": rmse,
        "char_f1": sum(f1s) / total if total else 0.0,
    }
