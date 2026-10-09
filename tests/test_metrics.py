import numpy as np
import pytest

from multipose2 import metrics


def _labels(shape, boxes):
    m = np.zeros(shape, np.int32)
    for i, (y0, y1, x0, x1) in enumerate(boxes, 1):
        m[y0:y1, x0:x1] = i
    return m


S = (32, 32)
EMPTY = _labels(S, [])
ONE = _labels(S, [(2, 10, 2, 10)])
TWO = _labels(S, [(2, 10, 2, 10), (16, 24, 16, 24)])


def test_pooled_f1_perfect_prediction():
    truth = [TWO, ONE] + [EMPTY] * 6
    r = metrics.pooled_f1(truth, list(truth))
    assert r["f1_50"] == 1.0
    assert r["f1_50_95"] == 1.0
    assert (r["tp"][0], r["fp"][0], r["fn"][0]) == (3, 0, 0)
    assert r["n_true"] == 3 and r["n_pred"] == 3


def test_pooled_f1_counts_spurious_masks_on_empty_images():
    """The failure mode per-image averaging hides: hallucinating on background."""
    truth = [TWO, ONE] + [EMPTY] * 6
    pred = [TWO, ONE] + [ONE] * 3 + [EMPTY] * 3
    pooled = metrics.pooled_f1(truth, pred)
    assert (pooled["tp"][0], pooled["fp"][0], pooled["fn"][0]) == (3, 3, 0)
    assert pooled["f1_50"] == pytest.approx(2 * 3 / (2 * 3 + 3 + 0))
    assert pooled["precision"][0] == pytest.approx(0.5)
    assert pooled["recall"][0] == 1.0

    # the per-image mean is higher, because the 3 still-empty images score 1.0
    per_image = metrics.average_precision(truth, pred, threshold=[0.5])[0][:, 0]
    assert per_image.mean() > pooled["f1_50"] - 0.1
    assert per_image[-3:].tolist() == [1.0, 1.0, 1.0]


def test_pooled_f1_missed_cells_lower_recall_not_precision():
    truth = [TWO, ONE] + [EMPTY] * 6
    pred = [ONE, ONE] + [EMPTY] * 6
    r = metrics.pooled_f1(truth, pred)
    assert r["precision"][0] == 1.0
    assert r["recall"][0] == pytest.approx(2 / 3)
    assert (r["tp"][0], r["fp"][0], r["fn"][0]) == (2, 0, 1)


def test_pooled_f1_all_empty_scores_zero_not_one():
    """A dataset with nothing to find yields no F1, rather than a perfect one."""
    r = metrics.pooled_f1([EMPTY] * 4, [EMPTY] * 4)
    assert r["f1_50"] == 0.0
    assert r["n_true"] == 0 and r["n_pred"] == 0
    # whereas per-image averaging calls this perfect
    assert metrics.average_precision([EMPTY] * 4, [EMPTY] * 4,
                                     threshold=[0.5])[0].mean() == 1.0


def test_pooled_f1_default_thresholds_are_50_to_95():
    r = metrics.pooled_f1([ONE], [ONE])
    assert len(r["threshold"]) == 10
    assert r["threshold"][0] == pytest.approx(0.5)
    assert r["threshold"][-1] == pytest.approx(0.95)
    assert r["f1_50"] == pytest.approx(r["f1"][0])


def test_pooled_f1_accepts_a_custom_threshold():
    r = metrics.pooled_f1([TWO], [TWO], threshold=0.75)
    assert r["threshold"].tolist() == [0.75]
    assert r["f1_50"] == pytest.approx(1.0)
