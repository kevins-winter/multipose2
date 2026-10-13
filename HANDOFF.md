# Multipose2 multimodal segmentation — session handoff

Written for a fresh Claude session picking this work up. Branch `multimodal-adapters`,
HEAD `9abd2f7`. Everything below is either labelled **measured**, **estimated**, or
**unknown**. Do not promote an estimate to a measurement, and do not treat the
unknowns as settled — two of them determine what the project can claim.

---

## 1. One-paragraph verdict

The engineering is ahead of the project's own timeline; the science has not started.
There are **zero held-out numbers for the current architecture**. Everything
quantitative here is either a defect measurement or a cost measurement. Five real
defects were found and fixed (three inherited from upstream Cellpose, one from this
branch, one introduced mid-session and then corrected). Two advisor reviews
substantially reshaped the design and corrected several overclaims. The immediate
blockers are not code: they are whether the transcript channels carry any signal at
all, and what the annotators looked at.

---

## 2. The project

Adapt Cellpose-SAM for multimodal segmentation of human sensory neuron somata in
DRG, fusing H&E morphology with Xenium transcript-derived channels (UCHL1
pan-neuronal, FABP7 satellite glia, subtype markers). Proposal: NRSC 4999 honours
thesis, two semesters, advisor Wenqin Luo, day-to-day Huasheng Yu. The proposal
commits to Cellpose-SAM specifically.

Aim 1 build the pipeline · Aim 2 compare fusion strategies · Aim 3 apply to TG.

---

## 3. Repo orientation

```bash
# tests run in a dedicated env; the default venv has no pytest and its
# numpy/cv2 are ABI-mismatched
conda create -n multipose2 python=3.10 && conda run -n multipose2 pip install -e . pytest

# the CHEAP suite: 141 tests, ~60 s on CPU. Use this one.
conda run -n multipose2 python -m pytest \
  tests/test_roles.py tests/test_modality_norm.py tests/test_metrics.py \
  tests/test_train.py tests/test_adapters.py tests/test_transforms.py \
  tests/test_utils.py -q -k "not cli and not class_train"
```

**The user is CPU-only locally and has asked not to run heavy suites.** Do not run
bare `pytest tests/` — `test_output.py` and `test_shape.py` run model eval, and
`test_class_train`/`test_cli_train` actually train. Colab T4 is the only GPU.

Notebook: `notebooks/train_multipose2_multimodal_adapters.ipynb`, 34 cells.
Sections: 0 preflight · 1 config · 2 synthesis · **2b per-modality normalization** ·
3 visualize · 4 train · 4b run manifest · 5 curves · 6 eval · **6b baselines** ·
channel diagnostics · single-modality baselines · 7 inspect · 8 W2 audit.

---

## 4. What exists

| file | what |
|---|---|
| `multipose2/roles.py` | `MultiscaleBranch`, `ModalityCorrections`, `RoleMixer`, `UnrestrictedCorrection` |
| `multipose2/modality_norm.py` | `ModalitySpec`, `ModalityNormalizer` (4 modes, globals fitted once, JSON round-trip) |
| `multipose2/metrics.py` | `pooled_f1`, `F1_THRESHOLDS` (0.50:0.05:0.95) alongside Cellpose's `average_precision` |
| `multipose2/vit_sam.py` | `forward_trunk`/`forward_head` split, `set_role_mixer`, `set_deterministic_trunk`, adapter `channel_norms`/`channel_l1` |
| `multipose2/train.py` | resume, timing, 8 trainable modes, `amp_dtype`, `role_l1`/`adapter_l1`, `_sync_stochastic_depth` |
| `notebooks/audit_w2.py` | standalone W2 drift audit, exits non-zero on drift |

`train_seg` gained: `trainable_mode`, `n_trainable_blocks`, `warmup_epochs`,
`training_stages`, `resume`, `resume_every`, `resume_path`, `timing`, `timing_path`,
`amp_dtype`, `role_l1`, `adapter_l1`.

Trainable modes: `all`, `adapter_head`, `adapter_head_last_blocks`, `adapter_only`,
`head_only`, `roles_only`, `roles_head`, `adapter_roles_head`.

---

## 5. Architecture as it stands

```
input (per-modality normalized)
  ├─► input_adapter (1x1, C→3, L1 group-lasso) ─► frozen SAM ViT-L x24 (stride 8)
  │                                              └─► head (out + fixed W2 pixel shuffle)
  │                                                    └─► base: cellprob + flows (full res)
  └─► per-modality correction branches (full res, 63 px receptive field)
        optionally FiLM-conditioned on the frozen trunk's features
        optionally mixed across modalities before the route heads
          ├─► foreground route → cellprob logit
          └─► flow route       → flow field
```

Corrections are **additive on the output**, so no gradient enters the trunk in
`roles_only`. `Transformer.forward` calls `self.role_mixer(x1, x, context=feat)`;
all three correction classes take `forward(base, inputs, context=None)`.

### The four arms (**measured** param counts, 4 modalities / 6 channels)

| arm | class / config | params | recept. field | attribution |
|---|---|---|---|---|
| 0 | `RoleMixer` — four named operators, as originally built | 7,140 | 5 px | per modality |
| 1 | `ModalityCorrections()` | 10,740 | 63 px | per modality |
| 2 | `ModalityCorrections(context_dim=256)` | 23,188 | 63 px | per modality |
| 3 | `ModalityCorrections(context_dim=256, cross_modality=True)` | 26,421 | 63 px | mix only |
| ctl | `UnrestrictedCorrection.matched_to(arm2, in_channels=6)` | 23,183 | 63 px | none |

Control matches arm 2 to **0.02%** on parameters and **exactly** on receptive field.
All arms are exact no-ops at init (coefficients zero / `softplus(-10)`), so every
comparison has a true zero point.

**Measured** behavioural separation: arm 1 is additive across modalities to
`3.6e-07`; arm 3 is not, deviating by `0.84`. Arm 2's correction changes with
context once FiLM is non-zero (it is zero-initialised, so context has no effect
until trained).

Constraints that were once fixed are now arguments:
`foreground_constraint ∈ {none, suppress_only, sign_identifiable}` (default `none`)
and `flow_parameterization ∈ {direct, potential}` (default `direct`).

---

## 6. Measured facts

### Hardware (from the preflight cell, real run)
Tesla T4, compute capability **7.5**, 14.6 GB, `/content` 70 GB free,
Python 3.13.15, torch 2.11.0+cu130, numpy 2.1.3, cv2 4.14.0.
`bf16_supported: true` — **via emulation**, which is the trap (see §8).

### Dataset (from a real training log)
- 188 fused train crops, 25 test crops
- `min_train_masks=5` dropped **139 of 188**, leaving 49
- **10 of 25** test crops have any ground-truth mask
- Roughly 400–800 annotated somata total (bounded from the above). Cellpose was
  trained on ~70,000 objects.
- Transcript renders are **JPEG**. A single-channel array saved as JPEG reads back
  as **three identical channels** — verified.

### Normalization (the one unambiguous win)
256×256 crops, transcript channel std after normalization:

| crop | per-crop percentile | globally fitted `log1p_fixed` |
|---|---|---|
| 20 transcripts | **0.00000** (channel zeroed) | 0.01747 |
| 1800 transcripts | 0.16492 | 0.16384 |

Below ~1% non-zero pixels the 1st and 99th percentiles coincide, `normalize99`'s
`x99 - x01 > 1e-3` guard fires, and the channel is zeroed. It is a cliff, not a
gradient.

Note the notebook currently assigns non-primary modalities `mode="fixed"`, not
`log1p_fixed`, because synthesis still fuses 8-bit renders rather than count maps.
Switch them to `log1p_fixed` once transcripts are re-rendered losslessly; the
cell says so in a comment. Either way the statistics are global, not per crop,
which is the property that matters.

### Training cost (T4)
| config | s/image | epoch @ n=188 | provenance |
|---|---|---|---|
| emulated bf16, batch 1 | 0.874 | 164 s | **measured** (step 164.37 s / 188) |
| `adapter_head` staged, fp16 | 0.250 | 47 s | **measured** (step 46.96 s / 188) |
| full fine-tune, fp16 | ~0.38 | ~71 s | estimated (block-count model) |
| `roles_only`, trunk forward only | ~0.128 | ~24 s | estimated (CPU ratio 1.95×, model predicts 2.0×) |
| `roles_only`, trunk cached | ~0.002–0.010 | ~1 s | **estimated, and the code path does not exist** |

Peak GPU memory: 5.814 GB at batch 1 (emulated bf16), 4.644 GB at batch 2 (fp16) —
memory went *down* while batch doubled, which is how fp16 was confirmed active.

### Identifiability checks (all run against the code)
- Sign degeneracy: negating coefficient **and** branch gave **bit-identical** output.
  Closed for signed roles by using a sigmoid branch.
- Spatial extent: one coefficient of 1.0, two branches, mean effect **50× apart**.
  Not fixable by architecture — hence `contribution_report()`.
- Suppression ceiling: worst case **−8.55 logits** at coefficient 2.13, an odds ratio
  of 1.9e-04. Never zero. It is suppression, not a gate.

### W2 readout
`W2` is an **exact pixel shuffle**: each of 192 channels maps to one output position
with weight 1.0; within an 8×8 block the base output has std 0.569. The base
**does** resolve single pixels. See §9 for the claim this refutes.

### Stochastic depth
`rdrop=0.4` leaves the frozen trunk non-deterministic in `train()` mode by max delta
**2.1157**. Now auto-disabled whenever no `encoder.` parameter is trainable.

### Prior run results — **do not quote these as performance**
From the one completed run: input adapter only, 49 oversampled crops, per-crop
normalization, broken metric, no baseline.

| | all 25 crops | non-empty GT only (n=10) |
|---|---|---|
| linear | 0.553 | 0.382 |
| msca_lite | 0.557 | 0.492 |

Decomposition: of the 0.553, **3.82 came from 10 real crops and 10.01 from 15 empty
crops scoring a perfect 1.0** — ~72% of the headline was background. Spurious masks
appeared on ~5/15 and ~6/15 empty crops.

---

## 7. Unknown, and gating

1. **Do the transcript channels carry any signal?** Notebook section 2b's channel
   report answers it in one cell and has never been run on real data. If they are
   duplicated (JPEG) and/or were being zeroed (normalization), the "6-channel
   multimodal" model has been training on H&E alone. The frozen test loss
   (0.2850 → 0.2851 over five epochs) is consistent with that.
2. **What did the annotators look at?** If they consulted transcript channels when
   deciding what counts as a neuron, a model using those channels agrees with the
   labels partly by construction. Per the second review this does *not* invalidate
   the work — reproducing multimodal expert segmentation is a legitimate goal — but
   it rules out independent biological discovery. Only the lab can answer.
3. **Split level.** Never verified at section level, let alone donor level. Leakage
   is a risk that inflates results, not proof that a result is memorization.
4. **Which Cellpose version did Yu et al. 2025 use?** The lab has six CP3 DRG
   fine-tunes from 2024 (`transDRG123`, `drgCyto3`, `drg_Neurons`, `segDRG1`,
   `TGCrop`, `TGTest`). If CP3, comparability argues for a smaller backbone.
5. Dataset size on Drive — a query was in progress and interrupted. Transcript
   `Train` jpgs sampled 7.5 KB–832 KB; the folder holds images and `_masks.png`
   together. Drive search returns ~5 files/page, so a full sum needs many calls.

---

## 8. Defects found, with origin

| defect | evidence | origin |
|---|---|---|
| `W2` optimized and weight-decayed | `requires_grad` flipped True on a fixed identity basis | **this branch**, pre-session |
| autocast silently disabled | "target dtype is not supported. Disabling autocast." | upstream Cellpose |
| emulated bf16 on a T4 | 3.35 s/image, ~13× slower than fp16 | **introduced mid-session by me**, then fixed |
| install fails on Python 3.13 | `pytest-runner` egg build; fixed via `pyproject.toml` | upstream Cellpose |
| per-crop normalization zeroes sparse channels | std exactly 0 | upstream Cellpose |
| two pre-existing test failures | 1-channel JPEG inflates to 3 channels | this branch, pre-session |
| `pip install` was a no-op on re-run | removing `--upgrade` blocked new commits | **me** |
| presence masked features, not terms | route-head bias leaked; broke additivity | **me**, caught by a test |
| control had 5 px field vs arms' 63 px | would confound structure with receptive field | **me**, caught by reading the arm table |

---

## 9. Claims that were wrong — do not reintroduce

These were asserted earlier in the session and are **false**:

1. ~~"The base is stride-8, so it cannot place a boundary more finely than 8 px."~~
   `W2` is a pixel shuffle; each pixel gets its own predicted value. The real limit
   is that all 192 values in a block are decoded **linearly from one stride-8
   feature vector**, so the base carries no *new* local detail. The prediction that
   full-resolution corrections help most at strict IoU survives, with a weaker
   justification.
2. ~~"Adding `logsigmoid` to a logit multiplies the probability."~~ It multiplies the
   **odds**.
3. ~~"The stock architecture cannot express gating even in principle."~~ The adapter
   feeds 24 nonlinear blocks. The real limits are 3-channel compression and
   interpretability.
4. ~~"An untrained mixer is bit-identical to stock Cellpose-SAM."~~ Bit-identical only
   when `enabled=False`; enabled-untrained shifts by <1e-3.
5. ~~"Flow corrections shape boundaries with no say over existence."~~ Mask
   reconstruction follows the flows, so they can merge, split or remove instances.
6. ~~"Annotation circularity invalidates the work / leakage means every number is
   memorization."~~ Over-escalated. Both bound the claim rather than voiding it.

---

## 10. Decision history — settled, do not re-litigate

- **Stay on Cellpose-SAM (CP4), not CP3, not from scratch.** From-scratch is
  impossible at ~400–800 annotated cells. CP3 was seriously considered and has real
  advantages (native `nchan`, true pyramid, full-resolution head, 6.6 M vs 304 M
  params, comparability with the lab's own models) but the decisive point is that in
  the configuration that matters H&E is exactly 3 channels, which is SAM's native
  interface. **Revisit CP3 only if** strong `flow_refine` contributions still leave
  the F1 tail flat, i.e. the ceiling is the base path.
- **No designated morphology modality.** The trunk sees all modalities through the
  learned, L1-penalized adapter; whichever channels win the 3 slots is a discovered
  result. Identity *initialisation* is a starting point, not a constraint.
- **Corrections sit at the output**, which is why `roles_only` needs no backward pass
  through the trunk (predicted 2.0×, measured 1.95× vs `adapter_head`).
- **Coefficients are a diagnostic, never evidence.** Report
  `contribution_report()` and settle behavioural questions by perturbation.
- **Normalization is global per modality, not per crop**, for transcripts. Per-crop
  statistics remove whatever varies between crops; for morphology that is nuisance
  (staining), for transcripts it is the signal (density). It is also a precondition
  for a suppression term to have a stable threshold.
- **Two separate routes (foreground, flow) are the hypothesis**; the four named
  operators were demoted to arguments after review.

---

## 11. Advisor review — what was conceded

Two detailed reviews. Round 1 rejected "discovering necessity" and interpretable
coefficients, and asked for behavioural tests plus a capacity-matched control.
Round 2 found the stride-8 error, the parity overstatement, the flow-exclusivity
error, the annotation over-escalation, and that the cached-training claim is not a
property of the checkout. Both were essentially correct and both are reflected in
the code and in the handoff deck.

Round 2's preferred direction — context-conditioned fusion with separate foreground
and flow outputs, SAM frozen — is implemented as arms 2 and 3. Their proposed
experiment (three small models, identical data and backbone inputs, plus the
unrestricted control and a fine-tuned single-modality baseline) is the plan.

### Defensible vs not

| defensible | not defensible |
|---|---|
| "Each modality's correction to foreground and to flows is measured separately" | "The model is interpretable" |
| "Removing UCHL1 costs N detections and M boundary accuracy at IoU 0.85" | "The model discovered UCHL1 is necessary" |
| "Structured corrections beat a capacity- and field-matched control by K" | "The coefficients recover marker semantics" |

Scale argument worth keeping: ~7–26 k readable parameters correct a prediction made
by ~304 M opaque ones. At best the **adjustment** is explained, not the segmentation.

---

## 12. Open work, prioritized

1. **Run section 2b's channel report on the real data.** One cell. Gates everything.
2. **Ask about the annotation protocol.** A conversation, not code.
3. **Re-render single-channel modalities losslessly** (`.tif`), re-run synthesis with
   `overwrite=True`, so transcripts stop becoming three duplicates.
4. **Audit the split**, section level then donor level.
5. **Implement the caching loop.** The trunk is deterministic now; the precompute and
   loader are not written. The risky part is the dihedral flow-vector transform —
   write its test against `random_rotate_and_resize` at θ ∈ {0, π/2, π, 3π/2}
   *before* the loop. Note `roles_only` freezes the head too, so the **final base
   prediction** can be cached (3×H×W, ~384 KB/variant fp16), not just neck features.
6. **Run the arm comparison** over one shared frozen base: train the adapter once,
   freeze, cache, then sweep arms × seeds over the role stage alone. Estimated ~25
   min one-time plus ~11 min for 6 arms × 5 seeds, versus ~18 h if each arm re-runs
   its own adapter stage.
7. **The perturbation battery**: remove/perturb each modality, measure missed
   neurons and false positives separately, read boundary quality from strict IoU.
   Keep *missing channel* (`set_presence`) distinct from *transcript dropout*
   (zeroing values) — the code distinguishes them; the analysis must too.
8. Consider a **FLOP-matched** control as well as parameter/field-matched: the
   structured arm does more compute at equal parameters.
9. Consider **pairwise branches** if arm 3 beats arm 1 — attribution for a pair is
   still readable, unlike the full mix.

---

## 13. Traps that bite silently

- **`pip install` on Colab is a no-op** unless the repo line is force-reinstalled;
  and even then, an already-imported `multipose2` survives, so **restart the
  runtime**. Notebook cell 1 now does both and says so. Cell 2 hard-fails if the
  loaded package lacks `adapter_type`/`amp_dtype`/`resume`.
- **Notebook changes need the notebook reloaded**, not just a package reinstall.
  This cost one full run: `min_train_masks` and `batch_size` live in cells, not in
  `train.py`.
- **`nimg_per_epoch` from `len(train_data)`** is pre-filter, so with
  `min_train_masks` dropping crops, `train_seg` samples **with replacement** and
  over-samples. Fixed by passing `None`; do not reintroduce.
- **"Off" differs per role.** A signed coefficient is off at `0.0`; a non-negative
  one is off at a large negative raw value (`softplus → 0`). Using `-20` for all of
  them sets `flow_center` to −20, not off. This silently corrupts an ablation.
- **`GroupNorm(min(4, width), width)`** is invalid for widths like 6, 10, 21. Use
  `_groups(width)`.
- **`torch.cuda.is_bf16_supported()`** defaults to `including_emulation=True` and
  returns True on a T4. Check `get_device_properties().major >= 8`.
- **`np.where` evaluates both branches**, so a discarded `0/0` still warns. Use
  `np.divide(..., where=...)`.
- Checkpoints trained through an `all` stage before `e1b998a` have a drifted `W2`.
  `notebooks/audit_w2.py` finds them; a drifted model is self-consistent but
  off-architecture, and cannot be repaired by resetting `W2`.

---

## 14. Artifacts

A 14-slide handoff deck exists (presentation form of the above, for the advisors) at
`claude.ai/artifact/Jg7bG85WZSjgxGAuqNKv24`. It is private until shared. Its results
slide is explicitly framed as a diagnosis of the prior run, not as performance.

---

## 15. The gating diagnostic, precisely

Two notebook cells answer unknown #1. Both already exist; neither has been run on
real data.

- **Cell 12** (section 2b) reads the channel layout off `modality_dirs` insertion
  order rather than hardcoding it, fits the normalizer on `train_data`, prints a
  per-channel raw/normalized std table, and saves `models/modality_norm.json`.
- **Cell 26** (channel diagnostics) checks the *other* failure: exact pixel-for-pixel
  duplicate channels, i.e. the JPEG inflation. `channel_report` cannot see this —
  a duplicated channel has healthy variance and reads as alive.

**Both inspect `train_data[0]` only.** That is the one real weakness: the dead-channel
failure is crop-dependent by construction (sparse crops die, dense ones do not), so a
single crop can report all-clear while a third of the training set is silently
single-modality. Before trusting either, loop the report over every crop and summarize
the *fraction* of crops in which each channel dies. That is a five-line change and it
is the first thing to do.

Keep the two questions separate when reporting: *duplicated* means `nchan` overstates
the information present, *dead* means normalization destroyed it. They have different
fixes (re-render as `.tif` vs. change the mode), and a channel can be both.

---

## 16. Commit map

25 commits, `a7e15eb..9abd2f7`. Grouped by what they were for, newest first within
each group.

**The correction architecture** (the current science)
```
9abd2f7  Match the control's receptive field, not only its parameters
5d8e388  Relax the correction design: context, receptive field, determinism
e1d4e4f  Add a capacity-matched control, and separate absence from empty signal
fd439a3  Reframe roles as output corrections, and close a sign degeneracy
3b48d6b  Penalize and stage the interpretable parameters
b2f8a97  Add per-modality role terms over the existing Cellpose outputs
fe43ee2  Split Transformer.forward into trunk and head
```
`fd439a3` onward are the two advisor reviews landing. Read `5d8e388` and `9abd2f7`
first if you want the current design rationale in the author's own words.

**Normalization**
```
acea3c4  Fit and apply the per-modality normalization in the notebook
a367d12  Normalize per modality, with global parameters fitted once
```

**Measurement**
```
372acf2  Fix two pre-existing test failures and add channel diagnostics
54616c1  Add pooled F1 and a baseline section to the notebook
cf30f57  Avoid a 0/0 warning when scoring an image with no masks
```

**Correctness fixes**
```
3d435dc  Stop oversampling when min_train_masks drops crops
e6fdd2b  Never select emulated bfloat16, and gate TF32 on hardware support
d6b0303  Make autocast actually use reduced precision
e1b998a  Never optimize architecturally frozen parameters
```
`e1b998a` is the W2 fix; `e6fdd2b` reverses a regression `d6b0303` introduced.

**Colab infrastructure**
```
6820893  Force-reinstall multipose2, and keep crops with few masks
62ecc52  Verify the loaded multipose2 is the branch build, not main
4eba438  Declare build requirements in pyproject.toml
dbd6cc3  Write a run manifest combining environment, config and timing
6dbb191  Add a W2 audit cell to the multimodal notebook
97b0add  Add a W2 drift audit for saved checkpoints
e03d03b  Add per-epoch timing breakdown to segmentation training
66c14e6  Add checkpoint resume to staged segmentation training
6367831  Pin Colab requirements and add a preflight cell
```

---

## 17. How to talk to this user

- They are a strong undergraduate who catches overclaims and asks for the mechanism.
  "Explain why global beats per crop?" was a corrective question, not a request for
  reassurance — they had asked for *per modality* and I had drifted to *per crop*.
- They want commits pushed as the work happens, not batched.
- They push back on hardcoding. Any design that names a modality by hand will be
  rejected; make it a discovered or configured property instead.
- **CPU only locally, Colab T4 for everything else.** Never propose anything that
  needs a local GPU, and never run a heavy test suite without being asked.
- Say what is measured and what is estimated, every time. Several of the corrections
  in §9 happened because an estimate was stated as a fact.
