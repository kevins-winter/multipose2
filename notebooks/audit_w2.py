"""Audit the W2 readout basis of saved checkpoints.

Transformer declares W2 as a fixed identity basis for the token-to-pixel readout
(requires_grad=False). Before commit e1b998a, set_trainable_parameters() set
requires_grad=True on every parameter when trainable_mode="all", so W2 received
gradients and weight decay and drifted away from identity. Any checkpoint trained
through an "all" stage before that fix is affected.

A drifted checkpoint is self-consistent -- W2 is saved and reloaded, so inference
matches training -- but it is off-architecture and its metrics are not comparable
with a correctly trained run. Resetting W2 to identity does not recover it,
because the output head was trained to compose with the drifted basis.

Usage:
    python notebooks/audit_w2.py <file-or-dir> [more ...]
    python notebooks/audit_w2.py ~/.cellpose/models
    python notebooks/audit_w2.py /content/drive/MyDrive/Multipose_Data --recursive

Exits non-zero if any audited checkpoint has a drifted W2.
"""
import argparse
import sys
from pathlib import Path

import torch

SKIP_SUFFIXES = {".npy", ".npz", ".txt", ".zip", ".gz", ".json", ".jsonl", ".csv",
                 ".tsv", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ipynb", ".py",
                 ".md", ".yml", ".yaml", ".cfg", ".ini", ".html", ".pdf"}


def _load_state_dict(path):
    """Return a state dict, or None if the file is not a loadable checkpoint."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # resume checkpoints carry numpy arrays and optimizer state, which
        # weights_only rejects; these are files we wrote ourselves
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            return None, f"could not load ({type(exc).__name__})"
    if isinstance(obj, dict) and "model_state" in obj:
        return obj["model_state"], "resume checkpoint"
    if isinstance(obj, dict):
        return obj, "state dict"
    return None, f"not a dict (got {type(obj).__name__})"


def audit_file(path):
    """Audit one checkpoint. Returns (verdict, detail)."""
    state, kind = _load_state_dict(path)
    if state is None:
        return "SKIP", kind

    key = "W2" if "W2" in state else ("module.W2" if "module.W2" in state else None)
    if key is None:
        return "SKIP", f"{kind}, no W2 (Cellpose 3 / non-CP4 model)"

    w2 = state[key].detach().to(torch.float32).cpu()
    if w2.ndim != 4 or w2.shape[0] != w2.shape[1] * w2.shape[2] * w2.shape[3]:
        return "SKIP", f"{kind}, unexpected W2 shape {tuple(w2.shape)}"

    n, nout, ps = w2.shape[0], w2.shape[1], w2.shape[2]
    expected = torch.eye(n).reshape(n, nout, ps, ps)
    dev = (w2 - expected).abs().max().item()
    norm_ratio = w2.norm().item() / expected.norm().item()

    detail = (f"{kind}, nout={nout} ps={ps}, max|W2-I|={dev:.3e}, "
              f"||W2||/||I||={norm_ratio:.6f}")
    if dev == 0.0:
        return "OK", detail
    if dev < 1e-6:
        return "OK", detail + "  (within float noise)"
    return "DRIFTED", detail


def iter_candidates(targets, recursive):
    for target in targets:
        p = Path(target).expanduser()
        if p.is_file():
            yield p
        elif p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            for child in sorted(it):
                if child.is_file() and child.suffix.lower() not in SKIP_SUFFIXES:
                    yield child
        else:
            print(f"  ?  {target}  (does not exist)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="checkpoint files or directories")
    ap.add_argument("--recursive", action="store_true",
                    help="descend into subdirectories")
    args = ap.parse_args(argv)

    counts = {"OK": 0, "DRIFTED": 0, "SKIP": 0}
    drifted = []
    for path in iter_candidates(args.targets, args.recursive):
        verdict, detail = audit_file(path)
        counts[verdict] += 1
        if verdict == "DRIFTED":
            drifted.append(path)
        if verdict != "SKIP":
            print(f"  {verdict:<8} {path.name:<42} {detail}")
        else:
            print(f"  {'skip':<8} {path.name:<42} {detail}")

    print(f"\n{counts['OK']} ok, {counts['DRIFTED']} drifted, {counts['SKIP']} skipped")
    if drifted:
        print("\nDrifted checkpoints were trained through an \"all\" stage before the "
              "W2 fix (commit e1b998a).\nTheir readout basis was decayed by "
              "weight_decay, so their metrics are not comparable\nwith a correctly "
              "trained run. Retrain rather than patching W2: the output head was\n"
              "trained to compose with the drifted basis, so resetting it to identity "
              "would not\nrecover the model.")
        for path in drifted:
            print(f"  {path}")
    return 1 if drifted else 0


if __name__ == "__main__":
    sys.exit(main())
