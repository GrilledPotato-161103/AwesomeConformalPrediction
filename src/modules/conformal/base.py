from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]

@dataclass
class ConformalConfig:
    """
    alpha: miscoverage level (e.g., 0.1 for 90% coverage)
    method:
      - "split" : classic split conformal using calibration scores
      - "naive" : alias to split (kept for extension)
    quantile_type:
      - "conservative" uses the standard split conformal index k = ceil((n+1)*(1-alpha))
    """
    alpha: float = 0.1
    method: str = "split"
    quantile_type: str = "conservative"


class Conformalizer:
    """
    Base Conformalizer implementing:
      1) Init
      2) Take prediction + ground-truth and preprocess
      3) Compute nonconformity score and save to array
      4) Quantile function
      5) Generate conformal prediction

    You must provide:
      - nonconformity_fn(pred, y) -> score (scalar or vector)
      - conformalize_fn(pred, qhat, **kwargs) -> conformal prediction object
    Optionally:
      - preprocess_fn(pred, y) -> (pred2, y2)
    """

    def __init__(
        self,
        config: ConformalConfig,
        nonconformity_fn: Callable[[Any, Any], np.ndarray],
        conformalize_fn: Callable[[Any, float], Any],
        preprocess_fn: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
    ):
        # ---- Init ----
        if not (0.0 < config.alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {config.alpha}")
        self.config = config

        self.nonconformity_fn = nonconformity_fn
        self.conformalize_fn = conformalize_fn
        self.preprocess_fn = preprocess_fn

        # calibration scores buffer
        self._scores: List[float] = []
        self._qhat: Optional[float] = None

    # ------------------------------------------------------------
    # Action 2: Take prediction + ground-truth and preprocess
    # ------------------------------------------------------------
    def ingest(self, pred: Any, y_true: Any) -> None:
        """
        Add one calibration example (pred, y_true).
        You can call this in a loop over your calibration set.
        """
        pred2, y2 = self._preprocess(pred, y_true)

        # --------------------------------------------------------
        # Action 3: compute nonconformity score and save
        # --------------------------------------------------------
        score = self._compute_score(pred2, y2)

        # Allow score to be scalar or array; store as 1D list of floats
        score_arr = np.asarray(score).reshape(-1)
        self._scores.extend([float(s) for s in score_arr])

        # Invalidate cached quantile if new scores arrive
        self._qhat = None

    def ingest_many(self, preds: Sequence[Any], y_trues: Sequence[Any]) -> None:
        if len(preds) != len(y_trues):
            raise ValueError("preds and y_trues must have same length")
        for p, y in zip(preds, y_trues):
            self.ingest(p, y)

    # ------------------------------------------------------------
    # Action 4: quantile function
    # ------------------------------------------------------------
    def quantile(self) -> float:
        """
        Compute qhat from calibration scores.
        Uses conservative split conformal quantile by default:
            k = ceil((n+1)*(1-alpha))
            qhat = k-th smallest score (1-indexed), clipped to [1,n]
        """
        if self._qhat is not None:
            return self._qhat

        scores = np.asarray(self._scores, dtype=float)
        if scores.size == 0:
            raise RuntimeError("No calibration scores found. Call ingest() first.")

        n = scores.size
        alpha = self.config.alpha

        # Conservative conformal quantile index (1-indexed)
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = int(np.clip(k, 1, n))

        # k-th smallest => np.partition at k-1
        qhat = float(np.partition(scores, k - 1)[k - 1])
        self._qhat = qhat
        return qhat

    # ------------------------------------------------------------
    # Action 5: generate conformal prediction
    # ------------------------------------------------------------
    def predict(self, pred: Any, *, qhat: Optional[float] = None, **kwargs) -> Any:
        """
        Conformalize a new model prediction `pred` into a prediction set/interval.
        - If qhat is None, uses self.quantile() from ingested calibration data.
        - kwargs are forwarded to conformalize_fn for task-specific behavior.
        """
        if qhat is None:
            qhat = self.quantile()
        return self.conformalize_fn(pred, float(qhat), **kwargs)

    # ------------------ internals ------------------
    def _preprocess(self, pred: Any, y_true: Any) -> Tuple[Any, Any]:
        if self.preprocess_fn is None:
            return pred, y_true
        return self.preprocess_fn(pred, y_true)

    def _compute_score(self, pred: Any, y_true: Any) -> np.ndarray:
        s = self.nonconformity_fn(pred, y_true)
        return np.asarray(s)


# ============================================================
# Example plug-ins (you can replace these)
# ============================================================

# --- Regression: absolute error interval ---
def reg_nonconformity(pred_y: ArrayLike, y_true: ArrayLike) -> np.ndarray:
    pred_y = np.asarray(pred_y, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return np.abs(pred_y - y_true)

def reg_conformalize(pred_y: ArrayLike, qhat: float) -> Tuple[np.ndarray, np.ndarray]:
    pred_y = np.asarray(pred_y, dtype=float)
    lo = pred_y - qhat
    hi = pred_y + qhat
    return lo, hi


# --- Classification: probability vector => set of labels above threshold ---
def cls_nonconformity(prob: ArrayLike, y_true: int) -> np.ndarray:
    """
    Score = 1 - p_true (smaller is better).
    prob: shape (C,)
    y_true: int in [0..C-1]
    """
    p = np.asarray(prob, dtype=float).reshape(-1)
    return np.array([1.0 - float(p[int(y_true)])], dtype=float)

def cls_conformalize(prob: ArrayLike, qhat: float) -> List[int]:
    """
    With score = 1 - p_true, the conformal set is:
      S = {c : 1 - p_c <= qhat}  <=>  p_c >= 1 - qhat
    """
    p = np.asarray(prob, dtype=float).reshape(-1)
    thr = 1.0 - qhat
    return [int(i) for i in np.where(p >= thr)[0]]


# ============================================================
# Minimal usage examples
# ============================================================
if __name__ == "__main__":
    # Regression
    reg = Conformalizer(
        config=ConformalConfig(alpha=0.1),
        nonconformity_fn=reg_nonconformity,
        conformalize_fn=reg_conformalize,
    )
    # calibration data
    reg.ingest(2.0, 1.5)
    reg.ingest(0.0, -0.3)
    reg.ingest(10.0, 11.2)
    q = reg.quantile()
    interval = reg.predict(5.0)
    print("REG qhat:", q, "interval for 5.0:", interval)

    # Classification
    cls = Conformalizer(
        config=ConformalConfig(alpha=0.1),
        nonconformity_fn=cls_nonconformity,
        conformalize_fn=cls_conformalize,
    )
    # calibration
    cls.ingest([0.1, 0.7, 0.2], 1)
    cls.ingest([0.6, 0.2, 0.2], 0)
    cls.ingest([0.3, 0.3, 0.4], 2)
    q = cls.quantile()
    pred_set = cls.predict([0.2, 0.5, 0.3])
    print("CLS qhat:", q, "set:", pred_set)
