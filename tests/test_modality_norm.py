import numpy as np
import pytest

from multipose2 import train
from multipose2.modality_norm import ModalityNormalizer, ModalitySpec


def _he(rng, shape=(64, 64)):
    """Dense, full-range, like a stain."""
    return rng.random((3, *shape), dtype=np.float32)


def _counts(shape=(64, 64), n=200, seed=0):
    """Sparse non-negative counts, like a transcript density map."""
    rng = np.random.default_rng(seed)
    img = np.zeros((1, *shape), np.float32)
    ys = rng.integers(0, shape[0], n)
    xs = rng.integers(0, shape[1], n)
    for y, x in zip(ys, xs):
        img[0, y, x] += 1.
    return img


def _layout(tx_mode="log1p_fixed"):
    return ModalityNormalizer.from_layout([
        {"name": "he", "n_channels": 3, "mode": "percentile"},
        {"name": "tx", "n_channels": 1, "mode": tx_mode},
    ])


def test_from_layout_assigns_channels_in_order():
    norm = _layout()
    assert [s.channels for s in norm.specs] == [(0, 1, 2), (3,)]
    assert norm.n_channels == 4


def test_overlapping_and_missing_channels_are_rejected():
    with pytest.raises(ValueError, match="claimed by both"):
        ModalityNormalizer([ModalitySpec("a", (0, 1)), ModalitySpec("b", (1, 2))])
    with pytest.raises(ValueError, match="not covered"):
        ModalityNormalizer([ModalitySpec("a", (0,)), ModalitySpec("b", (2,))])


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        ModalitySpec("a", (0,), mode="zscore")


def test_applying_an_unfitted_global_mode_raises():
    norm = _layout()
    assert not norm.is_fitted
    with pytest.raises(RuntimeError, match="fitted on the training set"):
        norm(np.zeros((4, 8, 8), np.float32))


def test_percentile_only_normalizer_needs_no_fit():
    norm = ModalityNormalizer.from_layout(
        [{"name": "he", "n_channels": 3, "mode": "percentile"}])
    assert norm.is_fitted
    out = norm(_he(np.random.default_rng(0)))
    assert out.shape == (3, 64, 64)


def test_sparse_counts_survive_log1p_fixed_but_die_under_percentile():
    """The failure this module exists to prevent.

    A sparse count channel has its 1st and 99th percentiles coincide, so
    normalize99 zeroes it outright and the modality contributes nothing.
    """
    sparse = _counts(n=15)   # 15 of 4096 px, under the 1st/99th span
    glob = ModalityNormalizer.from_layout(
        [{"name": "tx", "n_channels": 1, "mode": "percentile"}])
    assert glob(sparse)[0].std() == 0.0

    fitted = ModalityNormalizer.from_layout(
        [{"name": "tx", "n_channels": 1, "mode": "log1p_fixed"}])
    fitted.fit([sparse])
    assert fitted(sparse)[0].std() > 0.0


def test_log1p_fixed_preserves_density_differences_between_crops():
    """Per-crop normalization makes a near-empty crop look like a dense one."""
    sparse, dense = _counts(n=10, seed=1), _counts(n=2000, seed=2)

    per_crop = ModalityNormalizer.from_layout(
        [{"name": "tx", "n_channels": 1, "mode": "percentile"}])
    # both crops get stretched to the same range, so their means converge
    fitted = ModalityNormalizer.from_layout(
        [{"name": "tx", "n_channels": 1, "mode": "log1p_fixed"}])
    fitted.fit([sparse, dense])

    ratio_fitted = fitted(dense)[0].mean() / max(fitted(sparse)[0].mean(), 1e-9)
    # a global transform keeps the dense crop obviously denser
    assert ratio_fitted > 10
    # whereas the per-crop rule collapses or inverts that ordering
    assert per_crop(sparse)[0].std() == 0.0


def test_fit_is_global_not_per_image():
    """Two images normalized together must use one shared scale."""
    a, b = _counts(n=50, seed=3), _counts(n=1500, seed=4)
    norm = ModalityNormalizer.from_layout(
        [{"name": "tx", "n_channels": 1, "mode": "log1p_fixed"}])
    norm.fit([a, b])
    scale = norm.specs[0].fitted["scale"]
    # the same scale divides both, so applying it is image-independent
    assert np.allclose(norm(a)[0] * scale, np.log1p(a[0]))
    assert np.allclose(norm(b)[0] * scale, np.log1p(b[0]))


def test_flat_modality_warns_and_falls_back_to_unit_scale(caplog):
    flat = np.zeros((1, 8, 8), np.float32)
    norm = ModalityNormalizer.from_layout(
        [{"name": "tx", "n_channels": 1, "mode": "log1p_fixed"}])
    with caplog.at_level("WARNING"):
        norm.fit([flat])
    assert "flat across the whole training set" in caplog.text
    assert norm.specs[0].fitted["scale"] == 1.0


def test_identity_mode_leaves_values_alone():
    norm = ModalityNormalizer.from_layout(
        [{"name": "raw", "n_channels": 2, "mode": "identity"}])
    img = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    assert np.array_equal(norm(img), img)


def test_channels_last_round_trips():
    rng = np.random.default_rng(0)
    img_first = np.concatenate([_he(rng), _counts(n=100)], axis=0)
    norm = _layout()
    norm.fit([img_first])
    out_first = norm(img_first, channel_axis=0)
    out_last = norm(np.moveaxis(img_first, 0, -1), channel_axis=-1)
    assert out_last.shape == (64, 64, 4)
    assert np.allclose(out_first, np.moveaxis(out_last, -1, 0))


def test_wrong_channel_count_is_rejected():
    norm = _layout()
    norm.fit([np.concatenate([_he(np.random.default_rng(0)), _counts()], axis=0)])
    with pytest.raises(ValueError, match="expected 4 channels"):
        norm(np.zeros((6, 8, 8), np.float32))


def test_round_trips_through_json(tmp_path):
    norm = _layout()
    norm.fit([np.concatenate([_he(np.random.default_rng(0)), _counts()], axis=0)])
    path = norm.save(tmp_path / "norm.json")
    back = ModalityNormalizer.load(path)
    assert back.is_fitted
    assert back.to_dict() == norm.to_dict()
    img = np.concatenate([_he(np.random.default_rng(1)), _counts(seed=9)], axis=0)
    assert np.allclose(back(img), norm(img))


def test_channel_report_flags_dead_channels():
    img = np.concatenate([_he(np.random.default_rng(0)), _counts(n=15)], axis=0)
    norm = ModalityNormalizer.from_layout([
        {"name": "he", "n_channels": 3, "mode": "percentile"},
        {"name": "tx", "n_channels": 1, "mode": "percentile"},
    ])
    rows = norm.channel_report(img)
    assert [r["modality"] for r in rows] == ["he", "he", "he", "tx"]
    assert not any(r["dead"] for r in rows[:3])
    assert rows[3]["dead"] is True


def test_train_seg_rejects_an_unfitted_normalizer():
    norm = _layout()
    with pytest.raises(ValueError, match="unfitted modes"):
        train.train_seg(None, train_data=[], train_labels=[], normalize=norm)


def test_train_seg_rejects_a_bad_normalize_argument():
    with pytest.raises(ValueError, match="bool, a dict, or a ModalityNormalizer"):
        train.train_seg(None, train_data=[], train_labels=[], normalize="yes")


def test_subset_keeps_fitted_parameters_and_renumbers():
    img = np.concatenate([_he(np.random.default_rng(0)), _counts(n=300)], axis=0)
    norm = _layout()
    norm.fit([img])

    he_only = norm.subset([0, 1, 2])
    assert he_only.n_channels == 3
    assert [s.name for s in he_only.specs] == ["he"]
    assert he_only.specs[0].channels == (0, 1, 2)
    # normalizing the subset must match normalizing the full stack and slicing
    assert np.allclose(he_only(img[:3]), norm(img)[:3])

    tx_only = norm.subset([3])
    assert tx_only.n_channels == 1
    # the fitted scale is carried over, not refitted on the subset
    assert tx_only.specs[0].fitted == norm.specs[1].fitted
    assert np.allclose(tx_only(img[3:4]), norm(img)[3:4])


def test_subset_respects_the_requested_order():
    img = np.concatenate([_he(np.random.default_rng(0)), _counts(n=300)], axis=0)
    norm = _layout()
    norm.fit([img])
    reordered = norm.subset([3, 0])
    assert reordered.n_channels == 2
    assert {s.name: s.channels for s in reordered.specs} == {"tx": (0,), "he": (1,)}
    full = norm(img)
    assert np.allclose(reordered(img[[3, 0]]), full[[3, 0]])


def test_subset_of_nothing_raises():
    norm = _layout()
    norm.fit([np.concatenate([_he(np.random.default_rng(0)), _counts()], axis=0)])
    with pytest.raises(ValueError, match="no modality covers"):
        norm.subset([])
