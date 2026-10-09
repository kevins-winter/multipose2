import time
import os
import json
from contextlib import contextmanager
import numpy as np
from . import io, utils, models, dynamics
from .modality_norm import ModalityNormalizer
from .transforms import normalize_img, random_rotate_and_resize
from pathlib import Path
import torch
from torch import nn
from tqdm import trange

import logging

train_logger = logging.getLogger(__name__)

# parameter names that are frozen by the architecture, never optimized
NEVER_TRAINABLE_PARAMS = frozenset({"W2", "diam_mean", "diam_labels"})

# phases each epoch is split into for timing, and what to do about each
TIMING_PHASES = ("batch", "augment", "step", "test", "save")
TIMING_HINTS = {
    "augment": "augmentation dominates: raise batch_size and prefetch batches in worker processes",
    "batch": "batch assembly dominates: training data is on slow storage, copy it to local disk",
    "step": "the GPU step dominates: raise batch_size, and check autocast is active (a float32 dtype disables it)",
    "test": "validation dominates: lower nimg_test_per_epoch",
    "save": "checkpointing dominates: raise resume_every, or write checkpoints to local disk",
}

_AMP_DTYPE_NAMES = {
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
}


def _cuda_supports_bf16(device):
    """True only where bfloat16 is hardware-accelerated.

    torch.cuda.is_bf16_supported() defaults to including_emulation=True, so it
    returns True on pre-Ampere cards where bfloat16 is emulated in software and
    is far slower than float16 -- slower than float32 in practice. Compute
    capability 8.0 is the first with native bfloat16 tensor cores, so check that
    directly rather than trusting the convenience helper.
    """
    if device is None or device.type != "cuda":
        return False
    return torch.cuda.get_device_properties(device).major >= 8


def _resolve_amp_dtype(amp_dtype, device):
    """Choose the autocast dtype and whether gradient scaling is required.

    The autocast dtype is independent of the dtype the weights are held in:
    master weights stay float32 while the forward pass runs in reduced
    precision. Passing a float32 autocast dtype disables autocast entirely,
    which is why this never returns one.

    Args:
        amp_dtype: "auto", "bfloat16"/"bf16", "float16"/"fp16", a torch.dtype,
            or None/False/"float32" to disable autocast.
        device (torch.device): Device training will run on.

    Returns:
        tuple: (dtype or None, needs_scaler). A dtype of None disables autocast.
        Gradient scaling is needed for float16, whose range underflows, but not
        for bfloat16, which keeps float32's exponent range.
    """
    if amp_dtype in (None, False, "none", "float32", "fp32"):
        return None, False

    if amp_dtype in ("auto", True):
        if device.type == "cuda":
            dtype = (torch.bfloat16 if _cuda_supports_bf16(device)
                     else torch.float16)
        elif device.type == "cpu":
            dtype = torch.bfloat16
        else:
            # mps and other backends have incomplete autocast coverage
            train_logger.info(
                ">>> autocast disabled: no automatic dtype for device type %r",
                device.type)
            return None, False
    elif isinstance(amp_dtype, torch.dtype):
        dtype = amp_dtype
    elif isinstance(amp_dtype, str) and amp_dtype.lower() in _AMP_DTYPE_NAMES:
        dtype = _AMP_DTYPE_NAMES[amp_dtype.lower()]
    else:
        raise ValueError(
            "amp_dtype must be 'auto', 'bfloat16', 'float16', None, or a "
            f"torch.dtype, got {amp_dtype!r}"
        )

    if (dtype == torch.bfloat16 and device.type == "cuda"
            and not _cuda_supports_bf16(device)):
        train_logger.warning(
            "bfloat16 autocast requested but this GPU has no bfloat16 hardware "
            "(needs compute capability 8.0+, where bfloat16 would otherwise be "
            "emulated in software); using float16 with gradient scaling instead"
        )
        dtype = torch.float16
    return dtype, dtype == torch.float16


def _loss_fn_class(lbl, y, class_weights=None):
    """
    Calculates the loss function between true labels lbl and prediction y.

    Args:
        lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
        y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
        
    Returns:
        torch.Tensor: Loss value.

    """

    criterion3 = nn.CrossEntropyLoss(reduction="mean", weight=class_weights)
    loss3 = criterion3(y[:, :-3], lbl[:, 0].long())
    
    return loss3

def _loss_fn_seg(lbl, y, device):
    """
    Calculates the loss function between true labels lbl and prediction y.

    Args:
        lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
        y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
        device (torch.device): Device on which the tensors are located.

    Returns:
        torch.Tensor: Loss value.

    """
    criterion = nn.MSELoss(reduction="mean")
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    veci = 5. * lbl[:, -2:]
    loss = criterion(y[:, -3:-1], veci)
    loss /= 2.
    loss2 = criterion2(y[:, -1], (lbl[:, -3] > 0.5).to(y.dtype))
    loss = loss + loss2
    return loss

def _reshape_norm(data, channel_axis=None, normalize_params={"normalize": False}):
    """
    Reshapes and normalizes the input data.

    Args:
        data (list): List of input data, with channels axis first or last.
        normalize_params (dict, optional): Dictionary of normalization parameters. Defaults to {"normalize": False}.

    Returns:
        list: List of reshaped and normalized data.
    """
    data_new = []
    for td in data:
        if td.ndim == 2:
            td = td[np.newaxis, ...]
        elif td.ndim == 3:
            channel_axis0 = channel_axis if channel_axis is not None else np.array(td.shape).argmin()
            # put channel axis first
            td = np.moveaxis(td, channel_axis0, 0)
        data_new.append(td)
    data = data_new
    if normalize_params["normalize"]:
        modality_normalizer = normalize_params.get("modality_normalizer")
        if modality_normalizer is not None:
            data = [modality_normalizer(td, channel_axis=0) for td in data]
        else:
            data = [
                normalize_img(td, normalize=normalize_params, axis=0)
                for td in data
            ]
    return data

def _get_batch(inds, data=None, labels=None, files=None, labels_files=None,
               normalize_params={"normalize": False}):
    """
    Get a batch of images and labels.

    Args:
        inds (list): List of indices indicating which images and labels to retrieve.
        data (list or None): List of image data. If None, images will be loaded from files.
        labels (list or None): List of label data. If None, labels will be loaded from files.
        files (list or None): List of file paths for images.
        labels_files (list or None): List of file paths for labels.
        normalize_params (dict): Dictionary of parameters for image normalization (will be faster, if loading from files to pre-normalize).

    Returns:
        tuple: A tuple containing two lists: the batch of images and the batch of labels.
    """
    if data is None:
        lbls = None
        imgs = [io.imread(files[i]) for i in inds]
        imgs = _reshape_norm(imgs, normalize_params=normalize_params)
        if labels_files is not None:
            lbls = [io.imread(labels_files[i])[1:] for i in inds]
    else:
        imgs = [data[i] for i in inds]
        lbls = [labels[i][1:] for i in inds]
    return imgs, lbls


def _infer_nchan(data=None, files=None, channel_axis=None):
    if data is not None and len(data) > 0:
        sample = data[0]
        if sample.ndim == 2:
            return 1
        return sample.shape[0]
    if files is not None and len(files) > 0:
        sample = _reshape_norm([io.imread(files[0])], channel_axis=channel_axis)[0]
        return sample.shape[0]
    return None


def _ensure_net_input_channels(net, nchan):
    if nchan is None or not hasattr(net, "in_channels"):
        return
    if net.in_channels == nchan:
        return
    if hasattr(net, "set_input_adapter"):
        train_logger.warning(
            "training data have %d channels but model adapter expects %d; rebuilding input adapter before training",
            nchan,
            net.in_channels,
        )
        net.set_input_adapter(nchan)
        return
    raise ValueError(
        f"training data have {nchan} channels but model expects {net.in_channels}"
    )


def _last_encoder_block_prefixes(net, n_trainable_blocks):
    """Return parameter name prefixes for the final SAM encoder blocks."""
    if n_trainable_blocks < 1:
        return []
    blocks = getattr(getattr(net, "encoder", None), "blocks", None)
    if blocks is None:
        raise ValueError(
            "trainable_mode='adapter_head_last_blocks' requires net.encoder.blocks"
        )
    n_blocks = len(blocks)
    if n_trainable_blocks > n_blocks:
        train_logger.warning(
            "requested %d trainable encoder blocks, but network only has %d; using all blocks",
            n_trainable_blocks, n_blocks,
        )
        n_trainable_blocks = n_blocks
    start = n_blocks - n_trainable_blocks
    return [f"encoder.blocks.{i}." for i in range(start, n_blocks)]


def _is_never_trainable(name):
    """Parameters the architecture declares with requires_grad=False.

    W2 is a fixed identity basis for the token-to-pixel readout, and the diameter
    values are stored metadata rather than learned weights. Optimizing any of them
    (and applying weight decay to them) corrupts the model, so no trainable_mode
    may select them.
    """
    return name.rsplit(".", 1)[-1] in NEVER_TRAINABLE_PARAMS


def set_trainable_parameters(net, trainable_mode="all", n_trainable_blocks=2):
    """Select which network parameters are optimized during segmentation training."""
    valid_modes = {
        "all", "adapter_head", "adapter_head_last_blocks", "adapter_only",
        "head_only",
    }
    if trainable_mode not in valid_modes:
        raise ValueError(
            f"trainable_mode must be one of {sorted(valid_modes)}, got {trainable_mode!r}"
        )
    if n_trainable_blocks < 0:
        raise ValueError("n_trainable_blocks must be >= 0")
    for name, param in net.named_parameters():
        param.requires_grad = (trainable_mode == "all" and
                               not _is_never_trainable(name))
    if trainable_mode == "all":
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        total = sum(p.numel() for p in net.parameters())
        train_logger.info(
            ">>> trainable_mode=%s, optimizing %d / %d parameters",
            trainable_mode, trainable, total,
        )
        return

    prefixes = []
    if trainable_mode in {"adapter_head", "adapter_head_last_blocks", "adapter_only"}:
        prefixes.append("input_adapter.")
    if trainable_mode in {"adapter_head", "adapter_head_last_blocks", "head_only"}:
        prefixes.append("out.")
    if trainable_mode == "adapter_head_last_blocks":
        prefixes.extend(_last_encoder_block_prefixes(net, n_trainable_blocks))
        prefixes.append("encoder.neck.")

    for name, param in net.named_parameters():
        param.requires_grad = (any(name.startswith(prefix) for prefix in prefixes)
                               and not _is_never_trainable(name))

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    train_logger.info(
        ">>> trainable_mode=%s, n_trainable_blocks=%d, optimizing %d / %d parameters",
        trainable_mode, n_trainable_blocks, trainable, total,
    )


def _learning_rate_schedule(n_epochs, learning_rate, warmup_epochs=10,
                            decay_schedule=True):
    """Build the per-epoch learning-rate schedule used by segmentation training."""
    n_epochs = int(n_epochs)
    warmup_epochs = int(warmup_epochs)
    if n_epochs < 1:
        raise ValueError("n_epochs must be >= 1")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be >= 0")
    warmup_epochs = min(warmup_epochs, n_epochs)
    if warmup_epochs > 0:
        LR = np.linspace(0, learning_rate, warmup_epochs)
        LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - warmup_epochs)))
    else:
        LR = learning_rate * np.ones(n_epochs)
    if decay_schedule and n_epochs > 300:
        LR = LR[:-100]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif decay_schedule and n_epochs > 99:
        LR = LR[:-50]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))
    return LR


def _normalize_training_stages(training_stages, n_epochs, learning_rate,
                               trainable_mode, n_trainable_blocks,
                               warmup_epochs):
    """Return a validated list of training stage dictionaries."""
    if training_stages is None:
        return [{
            "name": "all" if trainable_mode == "all" else trainable_mode,
            "trainable_mode": trainable_mode,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "n_trainable_blocks": n_trainable_blocks,
            "warmup_epochs": warmup_epochs,
            "decay_schedule": True,
        }]
    if len(training_stages) == 0:
        raise ValueError("training_stages must contain at least one stage")

    stages = []
    for i, stage in enumerate(training_stages, start=1):
        if "n_epochs" not in stage:
            raise ValueError(f"training stage {i} is missing 'n_epochs'")
        if "learning_rate" not in stage:
            raise ValueError(f"training stage {i} is missing 'learning_rate'")
        stage_epochs = int(stage["n_epochs"])
        if stage_epochs < 1:
            raise ValueError(f"training stage {i} has n_epochs < 1")
        stage_warmup = int(stage.get("warmup_epochs", min(2, stage_epochs)))
        stages.append({
            "name": stage.get("name", f"stage{i}"),
            "trainable_mode": stage.get("trainable_mode", trainable_mode),
            "n_epochs": stage_epochs,
            "learning_rate": stage["learning_rate"],
            "n_trainable_blocks": stage.get("n_trainable_blocks", n_trainable_blocks),
            "warmup_epochs": stage_warmup,
            "decay_schedule": stage.get("decay_schedule", False),
        })
    return stages


class _PhaseTimer:
    """Accumulate wall time per named phase within one epoch."""

    def __init__(self):
        self.totals = {}

    @contextmanager
    def phase(self, name):
        t0 = time.time()
        try:
            yield
        finally:
            self.totals[name] = self.totals.get(name, 0.) + (time.time() - t0)

    def add(self, name, seconds):
        self.totals[name] = self.totals.get(name, 0.) + seconds

    def get(self, name):
        return self.totals.get(name, 0.)


def _peak_gpu_memory_gb(device):
    """Peak allocated GPU memory since the last reset, or None off CUDA."""
    if device is None or device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)


def _append_timing_record(path, record):
    """Append one JSON line per epoch, so a killed run still leaves its timings."""
    if path is None:
        return
    try:
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError as exc:
        train_logger.warning("could not write timing record to %s (%s)", path, exc)


def read_timing_records(path):
    """Read per-epoch timing records written by train_seg(timing=True).

    Args:
        path (str or Path): A ``<model_name>_timing.jsonl`` file.

    Returns:
        list of dict: One record per epoch, empty if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # a run killed mid-write can leave a partial final line
            train_logger.warning("skipping malformed timing record in %s", path)
    return records


def summarize_timing(records):
    """Summarize per-epoch timings and name the phase worth attacking first.

    Args:
        records (list of dict): Records from read_timing_records().

    Returns:
        dict or None: Totals, per-phase seconds and percentages, peak GPU memory,
        the dominant phase and a suggestion for it. None if there are no records.
    """
    if not records:
        return None
    walls = sorted(r["wall_s"] for r in records)
    total = float(sum(walls))
    phase_s = {p: float(sum(r.get(p + "_s", 0.) for r in records))
               for p in TIMING_PHASES}
    if total > 0:
        phase_pct = {p: round(100. * phase_s[p] / total, 1) for p in TIMING_PHASES}
        bottleneck = max(phase_pct, key=phase_pct.get)
    else:
        phase_pct = {p: 0. for p in TIMING_PHASES}
        bottleneck = None
    rates = [r["imgs_per_s"] for r in records if r.get("imgs_per_s")]
    peaks = [r["peak_gpu_gb"] for r in records if r.get("peak_gpu_gb")]
    return {
        "n_epochs": len(records),
        "total_s": round(total, 3),
        "total_min": round(total / 60., 2),
        "median_epoch_s": round(walls[len(walls) // 2], 3),
        "mean_imgs_per_s": round(sum(rates) / len(rates), 3) if rates else None,
        "phase_s": {p: round(v, 3) for p, v in phase_s.items()},
        "phase_pct": phase_pct,
        "peak_gpu_gb": max(peaks) if peaks else None,
        "bottleneck": bottleneck,
        "hint": TIMING_HINTS.get(bottleneck),
        "amp_dtype": records[-1].get("amp_dtype"),
    }


def _log_timing_summary(records):
    """Log where training time went, and the phase worth attacking."""
    summary = summarize_timing(records)
    if summary is None or summary["total_s"] <= 0:
        return
    train_logger.info(
        ">>> timing over %d epochs: median %.2fs/epoch, total %.1f min | "
        + ", ".join(f"{p} %.0f%%" for p in TIMING_PHASES),
        summary["n_epochs"], summary["median_epoch_s"], summary["total_min"],
        *(summary["phase_pct"][p] for p in TIMING_PHASES),
    )
    if summary["hint"]:
        train_logger.info(">>> %s", summary["hint"])


def _stage_fingerprint(stages):
    """Identity of a training schedule, so a resume cannot silently change it."""
    return [
        (s["name"], s["trainable_mode"], int(s["n_epochs"]),
         float(s["learning_rate"]), int(s["n_trainable_blocks"]),
         int(s["warmup_epochs"]), bool(s["decay_schedule"]))
        for s in stages
    ]


def _save_resume_checkpoint(path, net, optimizer, istage, stage_epoch, global_epoch,
                            train_losses, test_losses, stages):
    """Write a checkpoint that training can be resumed from.

    ``stage_epoch`` and ``global_epoch`` are the epochs that just *completed*.
    The write goes to a temporary file and is then renamed, so a run killed
    mid-write (a Colab disconnect, say) cannot leave a corrupt checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": 1,
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "istage": int(istage),
        "stage_epoch": int(stage_epoch),
        "global_epoch": int(global_epoch),
        "train_losses": np.asarray(train_losses),
        "test_losses": np.asarray(test_losses),
        "stages": _stage_fingerprint(stages),
        "in_channels": getattr(net, "in_channels", None),
        "adapter_type": getattr(net, "adapter_type", None),
        # numpy is re-seeded per epoch, so only torch's generator needs saving
        "torch_rng_state": torch.get_rng_state().cpu(),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    train_logger.info(
        "saved resume checkpoint to %s (stage %d, epoch %d)", path, istage,
        global_epoch,
    )


def _load_resume_checkpoint(path, net, stages, total_epochs):
    """Restore model weights and schedule position, or return None if absent."""
    path = Path(path)
    if not path.exists():
        train_logger.info(">>> no resume checkpoint at %s, starting from scratch", path)
        return None
    state = torch.load(path, map_location=net.device, weights_only=False)
    if state.get("stages") != _stage_fingerprint(stages):
        raise ValueError(
            f"resume checkpoint {path} was written for a different training schedule; "
            "restore the original stages, or pass resume=False to start over"
        )
    for key in ("in_channels", "adapter_type"):
        want, got = getattr(net, key, None), state.get(key)
        if got is not None and want != got:
            raise ValueError(
                f"resume checkpoint {path} has {key}={got!r} but this model has {want!r}"
            )
    net.load_state_dict(state["model_state"])
    try:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    except Exception as exc:  # non-fatal: only affects stochastic-depth draws
        train_logger.warning("could not restore torch RNG state (%s)", exc)
    train_losses = np.asarray(state["train_losses"], dtype="float64")
    test_losses = np.asarray(state["test_losses"], dtype="float64")
    if len(train_losses) != total_epochs or len(test_losses) != total_epochs:
        raise ValueError(
            f"resume checkpoint {path} has loss history of length "
            f"{len(train_losses)}, but this schedule has {total_epochs} epochs"
        )
    train_logger.info(
        ">>> resuming from %s: stage %d, %d epochs already completed",
        path, state["istage"], state["global_epoch"] + 1,
    )
    return {
        "istage": int(state["istage"]),
        "stage_epoch": int(state["stage_epoch"]) + 1,
        "global_epoch": int(state["global_epoch"]) + 1,
        "optimizer_state": state["optimizer_state"],
        "train_losses": train_losses,
        "test_losses": test_losses,
    }


def _reshape_norm_save(files, channels=None, channel_axis=None,
                       normalize_params={"normalize": False}):
    """ not currently used -- normalization happening on each batch if not load_files """
    files_new = []
    for f in trange(files):
        td = io.imread(f)
        if channels is not None:
            td = convert_image(td, channel_axis=channel_axis)
            td = td.transpose(2, 0, 1)
        if normalize_params["normalize"]:
            td = normalize_img(td, normalize=normalize_params, axis=0)
        fnew = os.path.splitext(str(f))[0] + "_cpnorm.tif"
        io.imsave(fnew, td)
        files_new.append(fnew)
    return files_new
    # else:
    #     train_files = reshape_norm_save(train_files, channels=channels,
    #                     channel_axis=channel_axis, normalize_params=normalize_params)
    # elif test_files is not None:
    #     test_files = reshape_norm_save(test_files, channels=channels,
    #                     channel_axis=channel_axis, normalize_params=normalize_params)


def _process_train_test(train_data=None, train_labels=None, train_files=None,
                        train_labels_files=None, train_probs=None, test_data=None,
                        test_labels=None, test_files=None, test_labels_files=None,
                        test_probs=None, load_files=True, min_train_masks=5,
                        compute_flows=False, normalize_params={"normalize": False}, 
                        channel_axis=None, device=None):
    """
    Process train and test data.

    Args:
        train_data (list or None): List of training data arrays.
        train_labels (list or None): List of training label arrays.
        train_files (list or None): List of training file paths.
        train_labels_files (list or None): List of training label file paths.
        train_probs (ndarray or None): Array of training probabilities.
        test_data (list or None): List of test data arrays.
        test_labels (list or None): List of test label arrays.
        test_files (list or None): List of test file paths.
        test_labels_files (list or None): List of test label file paths.
        test_probs (ndarray or None): Array of test probabilities.
        load_files (bool): Whether to load data from files.
        min_train_masks (int): Minimum number of masks required for training images.
        compute_flows (bool): Whether to compute flows.
        channels (list or None): List of channel indices to use.
        channel_axis (int or None): Axis of channel dimension.
        rgb (bool): Convert training/testing images to RGB.
        normalize_params (dict): Dictionary of normalization parameters.
        device (torch.device): Device to use for computation.

    Returns:
        tuple: A tuple containing the processed train and test data and sampling probabilities and diameters.
    """
    if device == None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else None
    
    if train_data is not None and train_labels is not None:
        # if data is loaded
        nimg = len(train_data)
        nimg_test = len(test_data) if test_data is not None else None
    else:
        # otherwise use files
        nimg = len(train_files)
        if train_labels_files is None:
            train_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in train_files
            ]
            train_labels_files = [tf for tf in train_labels_files if os.path.exists(tf)]
        if (test_data is not None or
                test_files is not None) and test_labels_files is None:
            test_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in test_files
            ]
            test_labels_files = [tf for tf in test_labels_files if os.path.exists(tf)]
        if not load_files:
            train_logger.info(">>> using files instead of loading dataset")
        else:
            # load all images
            train_logger.info(">>> loading images and labels")
            train_data = [io.imread(train_files[i]) for i in trange(nimg)]
            train_labels = [io.imread(train_labels_files[i]) for i in trange(nimg)]
        nimg_test = len(test_files) if test_files is not None else None
        if load_files and nimg_test:
            test_data = [io.imread(test_files[i]) for i in trange(nimg_test)]
            test_labels = [io.imread(test_labels_files[i]) for i in trange(nimg_test)]

    ### check that arrays are correct size
    if ((train_labels is not None and nimg != len(train_labels)) or
        (train_labels_files is not None and nimg != len(train_labels_files))):
        error_message = "train data and labels not same length"
        train_logger.critical(error_message)
        raise ValueError(error_message)
    if ((test_labels is not None and nimg_test != len(test_labels)) or
        (test_labels_files is not None and nimg_test != len(test_labels_files))):
        train_logger.warning("test data and labels not same length, not using")
        test_data, test_files = None, None
    if train_labels is not None:
        if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
            error_message = "training data or labels are not at least two-dimensional"
            train_logger.critical(error_message)
            raise ValueError(error_message)
        if train_data[0].ndim > 3:
            error_message = "training data is more than three-dimensional (should be 2D or 3D array)"
            train_logger.critical(error_message)
            raise ValueError(error_message)

    ### check that flows are computed
    if train_labels is not None:
        train_labels = dynamics.labels_to_flows(train_labels, files=train_files,
                                                device=device)
        if test_labels is not None:
            test_labels = dynamics.labels_to_flows(test_labels, files=test_files,
                                                   device=device)
    elif compute_flows:
        for k in trange(nimg):
            tl = dynamics.labels_to_flows(io.imread(train_labels_files),
                                          files=train_files, device=device)
        if test_files is not None:
            for k in trange(nimg_test):
                tl = dynamics.labels_to_flows(io.imread(test_labels_files),
                                              files=test_files, device=device)

    ### compute diameters
    nmasks = np.zeros(nimg)
    diam_train = np.zeros(nimg)
    train_logger.info(">>> computing diameters")
    for k in trange(nimg):
        tl = (train_labels[k][0]
              if train_labels is not None else io.imread(train_labels_files[k])[0])
        diam_train[k], dall = utils.diameters(tl)
        nmasks[k] = len(dall)
    diam_train[diam_train < 5] = 5.
    if test_data is not None:
        diam_test = np.array(
            [utils.diameters(test_labels[k][0])[0] for k in trange(len(test_labels))])
        diam_test[diam_test < 5] = 5.
    elif test_labels_files is not None:
        diam_test = np.array([
            utils.diameters(io.imread(test_labels_files[k])[0])[0]
            for k in trange(len(test_labels_files))
        ])
        diam_test[diam_test < 5] = 5.
    else:
        diam_test = None

    ### check to remove training images with too few masks
    if min_train_masks > 0:
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            train_logger.warning(
                f"{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set"
            )
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            if train_data is not None:
                train_data = [train_data[i] for i in ikeep]
                train_labels = [train_labels[i] for i in ikeep]
            if train_files is not None:
                train_files = [train_files[i] for i in ikeep]
            if train_labels_files is not None:
                train_labels_files = [train_labels_files[i] for i in ikeep]
            if train_probs is not None:
                train_probs = train_probs[ikeep]
            diam_train = diam_train[ikeep]
            nimg = len(train_data)

    ### normalize probabilities
    train_probs = 1. / nimg * np.ones(nimg,
                                      "float64") if train_probs is None else train_probs
    train_probs /= train_probs.sum()
    if test_files is not None or test_data is not None:
        test_probs = 1. / nimg_test * np.ones(
            nimg_test, "float64") if test_probs is None else test_probs
        test_probs /= test_probs.sum()

    ### reshape and normalize train / test data
    normed = False
    if normalize_params["normalize"]:
        train_logger.info(f">>> normalizing {normalize_params}")
    if train_data is not None:
        train_data = _reshape_norm(train_data, channel_axis=channel_axis, 
                                   normalize_params=normalize_params)
        normed = True
    if test_data is not None:
        test_data = _reshape_norm(test_data, channel_axis=channel_axis,
                                  normalize_params=normalize_params)

    return (train_data, train_labels, train_files, train_labels_files, train_probs,
            diam_train, test_data, test_labels, test_files, test_labels_files,
            test_probs, diam_test, normed)


def train_seg(net, train_data=None, train_labels=None, train_files=None,
              train_labels_files=None, train_probs=None, test_data=None,
              test_labels=None, test_files=None, test_labels_files=None,
              test_probs=None, channel_axis=None,
              load_files=True, batch_size=1, learning_rate=1e-5, SGD=False,
              n_epochs=100, weight_decay=0.1, normalize=True, compute_flows=False,
              save_path=None, save_every=100, save_each=False, nimg_per_epoch=None,
              nimg_test_per_epoch=None, rescale=False, scale_range=None, bsize=256,
              min_train_masks=5, model_name=None, class_weights=None,
              trainable_mode="all", n_trainable_blocks=2, warmup_epochs=10,
              training_stages=None, resume=False, resume_every=0,
              resume_path=None, timing=True, timing_path=None,
              amp_dtype="auto"):
    """
    Train the network with images for segmentation.

    Args:
        net (object): The network model to train. If `net` is a bfloat16 model it will be converted to float32 for training. The saved models will be in float32, but the original model will be returned as the original dtype for consistency. 
        train_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for training. Defaults to None.
        train_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for train_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        train_files (List[str], optional): List of strings - file names for images in train_data (to save flows for future runs). Defaults to None.
        train_labels_files (list or None): List of training label file paths. Defaults to None.
        train_probs (List[float], optional): List of floats - probabilities for each image to be selected during training. Defaults to None.
        test_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for testing. Defaults to None.
        test_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for test_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        test_files (List[str], optional): List of strings - file names for images in test_data (to save flows for future runs). Defaults to None.
        test_labels_files (list or None): List of test label file paths. Defaults to None.
        test_probs (List[float], optional): List of floats - probabilities for each image to be selected during testing. Defaults to None.
        load_files (bool, optional): Boolean - whether to load images and labels from files. Defaults to True.
        batch_size (int, optional): Integer - number of patches to run simultaneously on the GPU. Defaults to 1.
        learning_rate (float or List[float], optional): Float or list/np.ndarray - learning rate for training. Defaults to 1e-5.
        n_epochs (int, optional): Integer - number of times to go through the whole training set during training. Defaults to 100.
        weight_decay (float, optional): Float - weight decay for the optimizer. Defaults to 0.1.
        SGD (bool, optional): Deprecated in v4.0.1+ - AdamW always used.
        normalize (bool, dict, or ModalityNormalizer, optional): Whether and how to normalize. A ModalityNormalizer applies a per-modality transform whose global parameters were fitted on the training set, which matters when modalities differ in kind: per-crop percentiles suit morphology but destroy the absolute density that carries the signal in transcript counts. Defaults to True.
        compute_flows (bool, optional): Boolean - whether to compute flows during training. Defaults to False.
        save_path (str, optional): String - where to save the trained model. Defaults to None.
        save_every (int, optional): Integer - save the network every [save_every] epochs. Defaults to 100.
        save_each (bool, optional): Boolean - save the network to a new filename at every [save_each] epoch. Defaults to False.
        nimg_per_epoch (int, optional): Integer - minimum number of images to train on per epoch. Defaults to None.
        nimg_test_per_epoch (int, optional): Integer - minimum number of images to test on per epoch. Defaults to None.
        rescale (bool, optional): Boolean - whether or not to rescale images during training. Defaults to False.
        min_train_masks (int, optional): Integer - minimum number of masks an image must have to use in the training set. Defaults to 5.
        model_name (str, optional): String - name of the network. Defaults to None.
        trainable_mode (str, optional): Which parameters to optimize. Use "all" for full fine-tuning, "adapter_head" for the input adapter and output head, "adapter_head_last_blocks" for the input adapter, output head, encoder neck, and the final SAM encoder blocks, "adapter_only" for only the input adapter, or "head_only" for only the output head. Defaults to "all".
        n_trainable_blocks (int, optional): Number of final SAM encoder blocks to train when trainable_mode="adapter_head_last_blocks". Defaults to 2.
        warmup_epochs (int, optional): Number of epochs used to ramp the learning rate from 0 to learning_rate for non-staged training. Defaults to 10.
        training_stages (list of dict, optional): Continuous staged training schedule. Each stage can define "name", "trainable_mode", "n_epochs", "learning_rate", "n_trainable_blocks", "warmup_epochs", and "decay_schedule". Preprocessing runs once, then optimizer/trainable parameters are rebuilt at stage boundaries.
        resume (bool, optional): Load the checkpoint at resume_path, if it exists, and continue from the epoch after the one it recorded. Raises if the checkpoint was written for a different schedule or a differently shaped model. Defaults to False.
        resume_every (int, optional): Write a resume checkpoint every N epochs within each stage, and always at the end of a stage. 0 disables resume checkpointing. The checkpoint holds the full model and optimizer state, so it is roughly the model size plus two floats per trainable parameter; prefer local disk over a network drive. Defaults to 0.
        resume_path (str or Path, optional): Where the resume checkpoint is written and read. Defaults to save_path/models/<model_name>_resume.pt. Note that data preprocessing and flow computation are not cached, so they re-run on resume.
        timing (bool, optional): Log per-epoch wall time broken down into batch assembly, augmentation, the GPU step, validation and checkpointing, and append one JSON line per epoch to timing_path. The breakdown adds no CUDA synchronisation of its own, because loss.item() already synchronises at the end of each step. Defaults to True.
        timing_path (str or Path, optional): Where per-epoch timing records are appended as JSON lines. Defaults to save_path/models/<model_name>_timing.jsonl.
        amp_dtype (str or torch.dtype, optional): Autocast dtype for the forward pass, independent of the float32 master weights. "auto" picks bfloat16 on a GPU that supports it, float16 with gradient scaling otherwise, and bfloat16 on CPU. Pass "bfloat16"/"float16" to force one, or None/"float32" to train in full float32. Defaults to "auto".

    Returns:
        tuple: A tuple containing the path to the saved model weights, training losses, and test losses.
       
    """
    if SGD:
        train_logger.warning("SGD is deprecated, using AdamW instead")

    if isinstance(normalize, ModalityNormalizer):
        # a per-modality transform whose globals are already fitted on the
        # training set; carried through so _reshape_norm uses it in place of the
        # one-rule-for-every-channel default
        if not normalize.is_fitted:
            raise ValueError(
                "the ModalityNormalizer passed as normalize= has unfitted modes; "
                "call its fit() on the training images first")
        normalize_params = {"normalize": True, "modality_normalizer": normalize}
        train_logger.info(">>> normalizing per modality: %r", normalize)
    elif isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError(
            "normalize parameter must be a bool, a dict, or a ModalityNormalizer")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize

    device = net.device

    original_net_dtype = net.dtype 
    if net.dtype == torch.bfloat16:
        # NOTE: this produces a side effect of returning a network that is not of a guaranteed dtype \
        train_logger.info(">>> converting bfloat16 network to float32 for training")
        net.dtype = torch.float32

    amp_torch_dtype, amp_needs_scaler = _resolve_amp_dtype(amp_dtype, device)
    amp_enabled = amp_torch_dtype is not None
    # autocast requires a reduced-precision dtype; float32 silently disables it
    autocast_dtype = amp_torch_dtype or torch.float32
    scaler = (torch.amp.GradScaler(device.type) if amp_needs_scaler else None)
    amp_label = str(amp_torch_dtype).replace("torch.", "") if amp_enabled else "off"
    train_logger.info(
        ">>> autocast=%s, gradient scaling=%s", amp_label,
        "on" if amp_needs_scaler else "off",
    )

    scale_range = 0.5 if scale_range is None else scale_range

    out = _process_train_test(train_data=train_data, train_labels=train_labels,
                              train_files=train_files, train_labels_files=train_labels_files,
                              train_probs=train_probs,
                              test_data=test_data, test_labels=test_labels,
                              test_files=test_files, test_labels_files=test_labels_files,
                              test_probs=test_probs,
                              load_files=load_files, min_train_masks=min_train_masks,
                              compute_flows=compute_flows, channel_axis=channel_axis,
                              normalize_params=normalize_params, device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out
    # already normalized, do not normalize during training
    if normed:
        kwargs = {}
    else:
        kwargs = {"normalize_params": normalize_params, "channel_axis": channel_axis}

    input_nchan = _infer_nchan(train_data, train_files, channel_axis=channel_axis)
    _ensure_net_input_channels(net, input_nchan)
    stages = _normalize_training_stages(
        training_stages, n_epochs, learning_rate, trainable_mode,
        n_trainable_blocks, warmup_epochs,
    )
    
    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

    if class_weights is not None and isinstance(class_weights, (list, np.ndarray, tuple)):
        class_weights = torch.from_numpy(class_weights).to(device).float()
        print(class_weights)

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    total_epochs = sum(stage["n_epochs"] for stage in stages)
    train_logger.info(f">>> n_epochs={total_epochs}, n_train={nimg}, n_test={nimg_test}")
    if len(stages) == 1:
        train_logger.info(
            f">>> AdamW, learning_rate={stages[0]['learning_rate']:0.5f}, weight_decay={weight_decay:0.5f}"
        )
    else:
        train_logger.info(
            ">>> AdamW staged training, %d stages, weight_decay=%0.5f",
            len(stages), weight_decay,
        )

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    filename = save_path / "models" / model_name
    (save_path / "models").mkdir(exist_ok=True)

    train_logger.info(f">>> saving model to {filename}")

    lavg, nsum = 0, 0
    train_losses, test_losses = np.zeros(total_epochs), np.zeros(total_epochs)
    global_epoch = 0

    resume_path = (filename.with_name(filename.name + "_resume.pt")
                   if resume_path is None else Path(resume_path))
    timing_path = (filename.with_name(filename.name + "_timing.jsonl")
                   if timing_path is None else Path(timing_path)) if timing else None
    timing_records = []

    resume_istage, resume_stage_epoch, resume_optimizer_state = 1, 0, None
    if resume:
        resumed = _load_resume_checkpoint(resume_path, net, stages, total_epochs)
        if resumed is not None:
            resume_istage = resumed["istage"]
            resume_stage_epoch = resumed["stage_epoch"]
            global_epoch = resumed["global_epoch"]
            resume_optimizer_state = resumed["optimizer_state"]
            train_losses, test_losses = resumed["train_losses"], resumed["test_losses"]
            if global_epoch >= total_epochs:
                train_logger.warning(
                    "resume checkpoint %s is already at the end of the schedule "
                    "(%d/%d epochs); no training will run. Pass resume=False or "
                    "delete the checkpoint to retrain.",
                    resume_path, global_epoch, total_epochs,
                )
    if resume_every > 0:
        train_logger.info(">>> resume checkpointing every %d epochs to %s",
                          resume_every, resume_path)

    for istage, stage in enumerate(stages, start=1):
        if istage < resume_istage:
            continue
        lavg, nsum = 0, 0
        train_logger.info(
            ">>> stage %d/%d %s: trainable_mode=%s, n_epochs=%d, learning_rate=%0.6g, warmup_epochs=%d",
            istage, len(stages), stage["name"], stage["trainable_mode"],
            stage["n_epochs"], stage["learning_rate"], stage["warmup_epochs"],
        )
        set_trainable_parameters(
            net, trainable_mode=stage["trainable_mode"],
            n_trainable_blocks=stage["n_trainable_blocks"],
        )
        trainable_params = [p for p in net.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError(
                f"trainable_mode={stage['trainable_mode']!r} selected no parameters"
            )
        optimizer = torch.optim.AdamW(trainable_params, lr=stage["learning_rate"],
                                      weight_decay=weight_decay)
        if resume_optimizer_state is not None and istage == resume_istage:
            try:
                optimizer.load_state_dict(resume_optimizer_state)
                train_logger.info(">>> restored optimizer state for stage %d", istage)
            except ValueError as exc:
                train_logger.warning(
                    "could not restore optimizer state (%s); continuing with a fresh "
                    "optimizer for this stage", exc,
                )
            resume_optimizer_state = None
        LR = _learning_rate_schedule(
            stage["n_epochs"], stage["learning_rate"],
            warmup_epochs=stage["warmup_epochs"],
            decay_schedule=stage["decay_schedule"],
        )
        first_stage_epoch = resume_stage_epoch if istage == resume_istage else 0
        for stage_epoch in range(first_stage_epoch, stage["n_epochs"]):
            iepoch = global_epoch
            timer = _PhaseTimer()
            epoch_t0 = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            np.random.seed(iepoch)
            if nimg != nimg_per_epoch:
                # choose random images for epoch with probability train_probs
                rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,),
                                         p=train_probs)
            else:
                # otherwise use all images
                rperm = np.random.permutation(np.arange(0, nimg))
            for param_group in optimizer.param_groups:
                param_group["lr"] = LR[stage_epoch] # set learning rate
            net.train()
            for k in range(0, nimg_per_epoch, batch_size):
                kend = min(k + batch_size, nimg_per_epoch)
                inds = rperm[k:kend]
                with timer.phase("batch"):
                    imgs, lbls = _get_batch(inds, data=train_data, labels=train_labels,
                                            files=train_files, labels_files=train_labels_files,
                                            **kwargs)
                    diams = np.array([diam_train[i] for i in inds])
                    rsc = diams / net.diam_mean.item() if rescale else np.ones(
                        len(diams), "float32")
                # augmentations
                with timer.phase("augment"):
                    imgi, lbl = random_rotate_and_resize(imgs, Y=lbls, rescale=rsc,
                                                                    scale_range=scale_range,
                                                                    xy=(bsize, bsize))[:2]
                # network and loss optimization
                # loss.item() synchronises, so this phase measures the real GPU
                # cost without adding a sync of its own
                with timer.phase("step"):
                    X = torch.from_numpy(imgi).to(device)
                    lbl = torch.from_numpy(lbl).to(device)

                    # the losses stay inside the autocast region so autocast's
                    # own op policy promotes mse/bce/cross-entropy back to
                    # float32 rather than running them in reduced precision
                    with torch.autocast(device_type=device.type,
                                        dtype=autocast_dtype, enabled=amp_enabled):
                        y = net(X)[0]
                        loss = _loss_fn_seg(lbl, y, device)
                        if y.shape[1] > 3:
                            loss3 = _loss_fn_class(lbl, y, class_weights=class_weights)
                            loss += loss3
                    optimizer.zero_grad()
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    train_loss = loss.item()
                train_loss *= len(imgi)

                # keep track of average training loss across epochs
                lavg += train_loss
                nsum += len(imgi)
                # per epoch training loss
                train_losses[iepoch] += train_loss
            train_losses[iepoch] /= nimg_per_epoch

            should_log = iepoch == 5 or iepoch % 10 == 0 or stage_epoch == 0
            if should_log:
                lavgt = 0.
                _test_t0 = time.time()
                if test_data is not None or test_files is not None:
                    np.random.seed(42)
                    if nimg_test != nimg_test_per_epoch:
                        rperm = np.random.choice(np.arange(0, nimg_test),
                                                 size=(nimg_test_per_epoch,), p=test_probs)
                    else:
                        rperm = np.random.permutation(np.arange(0, nimg_test))
                    for ibatch in range(0, len(rperm), batch_size):
                        with torch.no_grad():
                            net.eval()
                            inds = rperm[ibatch:ibatch + batch_size]
                            imgs, lbls = _get_batch(inds, data=test_data,
                                                    labels=test_labels, files=test_files,
                                                    labels_files=test_labels_files,
                                                    **kwargs)
                            diams = np.array([diam_test[i] for i in inds])
                            rsc = diams / net.diam_mean.item() if rescale else np.ones(
                                len(diams), "float32")
                            imgi, lbl = random_rotate_and_resize(
                                imgs, Y=lbls, rescale=rsc, scale_range=scale_range,
                                xy=(bsize, bsize))[:2]
                            X = torch.from_numpy(imgi).to(device)
                            lbl = torch.from_numpy(lbl).to(device)

                            with torch.autocast(device_type=device.type,
                                                dtype=autocast_dtype,
                                                enabled=amp_enabled):
                                y = net(X)[0]
                                loss = _loss_fn_seg(lbl, y, device)
                                if y.shape[1] > 3:
                                    loss3 = _loss_fn_class(lbl, y, class_weights=class_weights)
                                    loss += loss3
                            test_loss = loss.item()
                            test_loss *= len(imgi)
                            lavgt += test_loss
                    lavgt /= len(rperm)
                    test_losses[iepoch] = lavgt
                timer.add("test", time.time() - _test_t0)
                lavg /= nsum
                train_logger.info(
                    f"{iepoch}, stage={stage['name']}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[stage_epoch]:.6f}, time {time.time()-t0:.2f}s"
                )
                lavg, nsum = 0, 0

            if iepoch == total_epochs - 1 or (iepoch % save_every == 0 and iepoch != 0):
                if save_each and iepoch != total_epochs - 1:  #separate files as model progresses
                    filename0 = str(filename) + f"_epoch_{iepoch:04d}"
                else:
                    filename0 = filename
                train_logger.info(f"saving network parameters to {filename0}")
                with timer.phase("save"):
                    net.save_model(filename0)

            if resume_every > 0 and ((stage_epoch + 1) % resume_every == 0 or
                                     stage_epoch == stage["n_epochs"] - 1):
                with timer.phase("save"):
                    _save_resume_checkpoint(
                        resume_path, net, optimizer, istage, stage_epoch, global_epoch,
                        train_losses, test_losses, stages,
                    )

            if timing:
                epoch_wall = time.time() - epoch_t0
                record = {
                    "epoch": iepoch, "stage": stage["name"], "istage": istage,
                    "stage_epoch": stage_epoch,
                    "wall_s": round(epoch_wall, 3),
                    "batch_s": round(timer.get("batch"), 3),
                    "augment_s": round(timer.get("augment"), 3),
                    "step_s": round(timer.get("step"), 3),
                    "test_s": round(timer.get("test"), 3),
                    "save_s": round(timer.get("save"), 3),
                    "nimg": int(nimg_per_epoch),
                    "batch_size": int(batch_size),
                    "imgs_per_s": round(nimg_per_epoch / epoch_wall, 3) if epoch_wall > 0 else None,
                    "lr": float(LR[stage_epoch]),
                    "train_loss": float(train_losses[iepoch]),
                    "peak_gpu_gb": _peak_gpu_memory_gb(device),
                    "amp_dtype": amp_label,
                }
                timing_records.append(record)
                _append_timing_record(timing_path, record)
                train_logger.info(
                    "%d stage=%s wall=%.2fs (batch %.2f, augment %.2f, step %.2f, "
                    "test %.2f, save %.2f) %.2f img/s%s",
                    iepoch, stage["name"], epoch_wall, record["batch_s"],
                    record["augment_s"], record["step_s"], record["test_s"],
                    record["save_s"], record["imgs_per_s"] or 0.,
                    f", peak {record['peak_gpu_gb']} GB" if record["peak_gpu_gb"] else "",
                )
            global_epoch += 1
    
    if timing:
        _log_timing_summary(timing_records)
        if timing_path is not None:
            train_logger.info(">>> per-epoch timings written to %s", timing_path)

    net.save_model(filename)
    if original_net_dtype != torch.float32:
        train_logger.info(f">>> converting network back to {original_net_dtype} after training")
        net.dtype = original_net_dtype

    return filename, train_losses, test_losses


def train_seg_staged(net, training_stages, **kwargs):
    """Train segmentation model with a continuous staged schedule.

    This is a convenience wrapper around train_seg(..., training_stages=...).
    Data preprocessing, flow generation, normalization, and channel checks run once;
    trainable parameters and the optimizer are rebuilt at each stage boundary.
    """
    return train_seg(net, training_stages=training_stages, **kwargs)
