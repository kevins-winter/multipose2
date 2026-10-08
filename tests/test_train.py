from multipose2 import io, models, train, vit_sam
from subprocess import check_output, STDOUT
import os, shutil
import torch
from pathlib import Path
import numpy as np
import json
import pytest


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def test_train_channel_guard_rebuilds_adapter():
    model = models.CellposeModel(gpu=False, nchan=3)
    train._ensure_net_input_channels(model.net, 5)
    assert model.net.in_channels == 5


def test_set_trainable_parameters_adapter_head():
    model = models.CellposeModel(gpu=False, nchan=5)
    train.set_trainable_parameters(model.net, trainable_mode="adapter_head")
    trainable_names = [
        name for name, param in model.net.named_parameters()
        if param.requires_grad
    ]
    assert trainable_names
    assert all(
        name.startswith("input_adapter.") or name.startswith("out.")
        for name in trainable_names
    )
    assert any(name.startswith("input_adapter.") for name in trainable_names)
    assert any(name.startswith("out.") for name in trainable_names)


def test_set_trainable_parameters_adapter_head_last_blocks():
    model = models.CellposeModel(gpu=False, nchan=5)
    n_blocks = len(model.net.encoder.blocks)
    train.set_trainable_parameters(
        model.net, trainable_mode="adapter_head_last_blocks",
        n_trainable_blocks=2,
    )
    trainable_names = [
        name for name, param in model.net.named_parameters()
        if param.requires_grad
    ]
    assert trainable_names
    assert any(name.startswith("input_adapter.") for name in trainable_names)
    assert any(name.startswith("out.") for name in trainable_names)
    assert any(name.startswith("encoder.neck.") for name in trainable_names)
    assert any(
        name.startswith(f"encoder.blocks.{n_blocks - 1}.")
        for name in trainable_names
    )
    assert any(
        name.startswith(f"encoder.blocks.{n_blocks - 2}.")
        for name in trainable_names
    )
    assert not any(
        name.startswith(f"encoder.blocks.{n_blocks - 3}.")
        for name in trainable_names
    )
    assert all(
        name.startswith("input_adapter.") or
        name.startswith("out.") or
        name.startswith("encoder.neck.") or
        name.startswith(f"encoder.blocks.{n_blocks - 1}.") or
        name.startswith(f"encoder.blocks.{n_blocks - 2}.")
        for name in trainable_names
    )


def test_learning_rate_schedule_short_stage_warmup():
    lr = train._learning_rate_schedule(
        n_epochs=5, learning_rate=1e-4, warmup_epochs=2,
        decay_schedule=False,
    )
    np.testing.assert_allclose(lr, [0, 1e-4, 1e-4, 1e-4, 1e-4])


def test_normalize_training_stages_defaults_to_short_warmup():
    stages = train._normalize_training_stages(
        [{"name": "tail", "trainable_mode": "adapter_head_last_blocks",
          "n_epochs": 10, "learning_rate": 5e-6}],
        n_epochs=100, learning_rate=1e-5, trainable_mode="all",
        n_trainable_blocks=2, warmup_epochs=10,
    )
    assert stages == [{
        "name": "tail",
        "trainable_mode": "adapter_head_last_blocks",
        "n_epochs": 10,
        "learning_rate": 5e-6,
        "n_trainable_blocks": 2,
        "warmup_epochs": 2,
        "decay_schedule": False,
    }]


def test_synthesize_multimodal_training_dir(tmp_path):
    he_dir = tmp_path / "H&EStain" / "Training"
    tx_dir = tmp_path / "UnremovedTranscripts" / "Training"
    out_dir = tmp_path / "Synthesized" / "Training"
    he_dir.mkdir(parents=True)
    tx_dir.mkdir(parents=True)

    io.imsave(str(he_dir / "sample_001.tif"),
              np.zeros((16, 16, 3), dtype=np.float32))
    io.imsave(str(tx_dir / "sample_001.tif"),
              np.ones((16, 16, 2), dtype=np.float32))
    io.imsave(str(he_dir / "sample_001_masks.tif"),
              np.zeros((16, 16), dtype=np.uint16))

    train_dir = io.synthesize_multimodal_training_dir(
        modality_dirs={"he": he_dir, "transcripts": tx_dir},
        output_dir=out_dir,
        label_dir=he_dir,
        mask_filter="_masks.tif",
        modality_channel_axes={"he": -1, "transcripts": -1},
    )

    images, labels, image_names, *_ = io.load_train_test_data(
        str(train_dir), mask_filter="_masks.tif"
    )
    assert len(images) == 1
    assert images[0].shape == (16, 16, 5)
    assert labels[0].shape == (16, 16)
    assert Path(image_names[0]).parent == out_dir


def test_synthesize_multimodal_training_dir_with_sample_id_regex(tmp_path):
    he_dir = tmp_path / "H&EStain" / "Training"
    tx_dir = tmp_path / "UnremovedTranscripts" / "Training"
    label_dir = tmp_path / "Labels" / "Training"
    out_dir = tmp_path / "SynthesizedRegex" / "Training"
    he_dir.mkdir(parents=True)
    tx_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    io.imsave(str(he_dir / "DRG_1_HES_0_500.tif"),
              np.zeros((16, 16, 3), dtype=np.float32))
    io.imsave(str(tx_dir / "DRG_1_TX_0_500.tif"),
              np.ones((16, 16, 2), dtype=np.float32))
    io.imsave(str(label_dir / "DRG_1_manual_0_500_masks.tif"),
              np.zeros((16, 16), dtype=np.uint16))

    train_dir = io.synthesize_multimodal_training_dir(
        modality_dirs={"he": he_dir, "transcripts": tx_dir},
        output_dir=out_dir,
        label_dir=label_dir,
        mask_filter="_masks.tif",
        modality_channel_axes={"he": -1, "transcripts": -1},
        sample_id_regex=r"(DRG_\d+)_.*_(\d+_\d+)",
    )

    images, labels, image_names, *_ = io.load_train_test_data(
        str(train_dir), mask_filter="_masks.tif"
    )
    assert len(images) == 1
    assert images[0].shape == (16, 16, 5)
    assert labels[0].shape == (16, 16)
    assert Path(image_names[0]).name == "DRG_1_0_500.tif"


def test_synthesize_multimodal_training_dir_ignores_secondary_masks(tmp_path):
    he_dir = tmp_path / "H&EStain" / "Train"
    tx_dir = tmp_path / "UnremovedTranscripts" / "Train"
    out_dir = tmp_path / "Synthesized" / "Train"
    he_dir.mkdir(parents=True)
    tx_dir.mkdir(parents=True)

    io.imsave(str(he_dir / "DRG_1_HES_0_500.jpg"),
              np.zeros((16, 16, 3), dtype=np.uint8))
    io.imsave(str(he_dir / "DRG_1_HES_0_500_masks.png"),
              np.zeros((16, 16), dtype=np.uint8))
    io.imsave(str(tx_dir / "DRG_1_SegTrans_0_500.jpg"),
              np.ones((16, 16, 1), dtype=np.uint8))
    io.imsave(str(tx_dir / "DRG_1_SegTrans_0_500_masks.png"),
              np.zeros((16, 16), dtype=np.uint8))

    train_dir = io.synthesize_multimodal_training_dir(
        modality_dirs={"he": he_dir, "transcripts": tx_dir},
        output_dir=out_dir,
        label_dir=he_dir,
        mask_filter="_masks.png",
        modality_channel_axes={"he": -1, "transcripts": -1},
        sample_id_regex=r"(DRG_\d+)_.*_(\d+_\d+)",
    )

    images, labels, *_ = io.load_train_test_data(
        str(train_dir), mask_filter="_masks.png"
    )
    assert len(images) == 1
    assert images[0].shape == (16, 16, 4)
    assert labels[0].shape == (16, 16)


def test_synthesize_multimodal_training_dir_accepts_mask_filter_with_extension(tmp_path):
    he_dir = tmp_path / "H&EStain" / "Train"
    tx_dir = tmp_path / "UnremovedTranscripts" / "Train"
    out_dir = tmp_path / "SynthesizedExtension" / "Train"
    he_dir.mkdir(parents=True)
    tx_dir.mkdir(parents=True)

    io.imsave(str(he_dir / "DRG_1_HES_0_500.jpg"),
              np.zeros((16, 16, 3), dtype=np.uint8))
    io.imsave(str(tx_dir / "DRG_1_SegTrans_0_500.jpg"),
              np.ones((16, 16, 1), dtype=np.uint8))
    io.imsave(str(he_dir / "DRG_1_HES_0_500_masks.png"),
              np.zeros((16, 16), dtype=np.uint8))

    train_dir = io.synthesize_multimodal_training_dir(
        modality_dirs={"he": he_dir, "transcripts": tx_dir},
        output_dir=out_dir,
        label_dir=he_dir,
        mask_filter="_masks.png",
        modality_channel_axes={"he": -1, "transcripts": -1},
        sample_id_regex=r"(DRG_\d+)_.*_(\d+_\d+)",
    )

    images, labels, image_names, *_ = io.load_train_test_data(
        str(train_dir), mask_filter="_masks.png"
    )
    assert len(images) == 1
    assert images[0].shape == (16, 16, 4)
    assert labels[0].shape == (16, 16)
    assert Path(image_names[0]).with_suffix("").name == "DRG_1_0_500"


def test_class_train(data_dir):
    train_dir = str(data_dir.joinpath('2D').joinpath('train'))
    model_dir = str(data_dir.joinpath('2D').joinpath('train').joinpath('models'))
    shutil.rmtree(model_dir, ignore_errors=True)
    output = io.load_train_test_data(train_dir, mask_filter='_cyto_masks')
    images, labels, image_names, test_images, test_labels, image_names_test = output
    use_gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=use_gpu)
    cpmodel_path = train.train_seg(model.net, images, labels, train_files=image_names,
                                   test_data=test_images, test_labels=test_labels,
                                   test_files=image_names_test,
                                   save_path=train_dir, n_epochs=3)[0]
    io.add_model(cpmodel_path)
    io.remove_model(cpmodel_path, delete=True)
    print('>>>> model trained and saved to %s' % cpmodel_path)


def test_cli_train(data_dir):
    # import sys
    # path_root = Path(__file__).parents[1]
    # sys.path.append(str(path_root))
    # print(Path(__file__).parents[0],Path(__file__).parents[1],Path(__file__).parents[2])
    train_dir = str(data_dir.joinpath('2D').joinpath('train'))
    model_dir = str(data_dir.joinpath('2D').joinpath('train').joinpath('models'))
    shutil.rmtree(model_dir, ignore_errors=True)
    use_gpu = torch.cuda.is_available()
    gpu_str = "--use_gpu" if use_gpu else ""
    cmd = 'python -m multipose2 %s --train --n_epochs 3 --dir %s --mask_filter _cyto_masks --pretrained_model None' % (gpu_str, train_dir)
    try:
        cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
    except Exception as e:
        print(e)
        raise ValueError(e)


def test_cli_make_train(data_dir):
    script_name = Path().resolve() / 'multipose2/gui/make_train.py'
    image_path = data_dir / '3D/gray_3D.tif'

    cmd = f'python {script_name} --image_path {image_path}'
    res = check_output(cmd, stderr=STDOUT, shell=True)

    # there should be 30 slices: 
    files = [f for f in (data_dir / '3D/train/').iterdir() if 'gray_3D' in f.name]
    assert 30 == len(files)

    shutil.rmtree((data_dir / '3D/train'))


def test_never_trainable_params_stay_frozen_in_every_mode():
    net = vit_sam.Transformer(in_channels=5, bsize=64)
    params = dict(net.named_parameters())
    for name in train.NEVER_TRAINABLE_PARAMS:
        assert name in params, f"{name} is not a parameter of Transformer"
    for mode in ("all", "adapter_head", "adapter_head_last_blocks",
                 "adapter_only", "head_only"):
        train.set_trainable_parameters(net, trainable_mode=mode)
        for name in train.NEVER_TRAINABLE_PARAMS:
            assert not params[name].requires_grad, (
                f"{name} became trainable in trainable_mode={mode!r}"
            )


def test_trainable_mode_all_still_trains_encoder_and_adapter():
    net = vit_sam.Transformer(in_channels=5, bsize=64)
    train.set_trainable_parameters(net, trainable_mode="all")
    assert net.encoder.blocks[0].attn.qkv.weight.requires_grad
    assert net.encoder.pos_embed.requires_grad
    assert net.input_adapter.proj.weight.requires_grad
    assert net.out.weight.requires_grad


class _TinyNet(torch.nn.Module):
    """Minimal stand-in exposing what the resume helpers touch."""

    def __init__(self, in_channels=5, adapter_type="linear"):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.in_channels = in_channels
        self.adapter_type = adapter_type

    def forward(self, x):
        return self.lin(x)

    @property
    def device(self):
        return next(self.parameters()).device


def _tiny_stages():
    return train._normalize_training_stages(
        [{"name": "s1", "trainable_mode": "all", "n_epochs": 3, "learning_rate": 1e-5},
         {"name": "s2", "trainable_mode": "all", "n_epochs": 2, "learning_rate": 1e-6}],
        n_epochs=5, learning_rate=1e-5, trainable_mode="all",
        n_trainable_blocks=2, warmup_epochs=0,
    )


def _save_tiny(tmp_path, net, stages, istage, stage_epoch, global_epoch,
               train_losses=None, test_losses=None):
    opt = torch.optim.AdamW(net.parameters(), lr=1e-5)
    net(torch.zeros(2, 4)).sum().backward()
    opt.step()
    path = tmp_path / "run_resume.pt"
    train._save_resume_checkpoint(
        path, net, opt, istage, stage_epoch, global_epoch,
        np.zeros(5) if train_losses is None else train_losses,
        np.zeros(5) if test_losses is None else test_losses,
        stages,
    )
    return path


def test_resume_checkpoint_roundtrip_restores_position_and_weights(tmp_path):
    stages = _tiny_stages()
    net = _TinyNet()
    losses = np.arange(5, dtype="float64")
    path = _save_tiny(tmp_path, net, stages, istage=1, stage_epoch=1, global_epoch=1,
                      train_losses=losses)
    saved_weight = net.lin.weight.detach().clone()

    fresh = _TinyNet()
    with torch.no_grad():
        fresh.lin.weight.fill_(0.0)
    resumed = train._load_resume_checkpoint(path, fresh, stages, total_epochs=5)

    # the saved epoch is the one that completed, so resume starts at the next
    assert resumed["istage"] == 1
    assert resumed["stage_epoch"] == 2
    assert resumed["global_epoch"] == 2
    assert np.array_equal(resumed["train_losses"], losses)
    assert torch.equal(fresh.lin.weight, saved_weight)
    assert resumed["optimizer_state"]["state"]


def test_resume_at_stage_end_rolls_over_to_next_stage(tmp_path):
    stages = _tiny_stages()
    net = _TinyNet()
    # last epoch of stage 1 (n_epochs=3) completed
    path = _save_tiny(tmp_path, net, stages, istage=1, stage_epoch=2, global_epoch=2)
    resumed = train._load_resume_checkpoint(path, _TinyNet(), stages, total_epochs=5)
    # stage_epoch == n_epochs makes range(first, n_epochs) empty, so stage 1 is
    # skipped and stage 2 starts at 0
    assert resumed["stage_epoch"] == stages[0]["n_epochs"]
    assert range(resumed["stage_epoch"], stages[0]["n_epochs"]) == range(3, 3)
    assert resumed["global_epoch"] == 3


def test_resume_checkpoint_rejects_changed_schedule(tmp_path):
    stages = _tiny_stages()
    path = _save_tiny(tmp_path, _TinyNet(), stages, 1, 0, 0)
    changed = train._normalize_training_stages(
        [{"name": "s1", "trainable_mode": "all", "n_epochs": 4, "learning_rate": 1e-5},
         {"name": "s2", "trainable_mode": "all", "n_epochs": 2, "learning_rate": 1e-6}],
        n_epochs=6, learning_rate=1e-5, trainable_mode="all",
        n_trainable_blocks=2, warmup_epochs=0,
    )
    with pytest.raises(ValueError, match="different training schedule"):
        train._load_resume_checkpoint(path, _TinyNet(), changed, total_epochs=6)


def test_resume_checkpoint_rejects_mismatched_model_shape(tmp_path):
    stages = _tiny_stages()
    path = _save_tiny(tmp_path, _TinyNet(in_channels=5), stages, 1, 0, 0)
    with pytest.raises(ValueError, match="in_channels"):
        train._load_resume_checkpoint(path, _TinyNet(in_channels=8), stages,
                                      total_epochs=5)


def test_resume_checkpoint_write_is_atomic_and_absent_returns_none(tmp_path):
    stages = _tiny_stages()
    path = _save_tiny(tmp_path, _TinyNet(), stages, 1, 0, 0)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert train._load_resume_checkpoint(tmp_path / "nope.pt", _TinyNet(), stages, 5) is None


class _Interrupted(RuntimeError):
    """Stands in for a Colab disconnect part-way through training."""


class _FakeSegNet(torch.nn.Module):
    """Tiny stand-in for Transformer: same interface train_seg needs, trivial compute."""

    def __init__(self, nchan=2, fail_after_steps=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(nchan, 3, 1)
        self.in_channels = nchan
        self.adapter_type = "linear"
        self.diam_mean = torch.nn.Parameter(torch.tensor([30.]), requires_grad=False)
        self.diam_labels = torch.nn.Parameter(torch.tensor([30.]), requires_grad=False)
        self.W2 = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.steps = 0
        self.fail_after_steps = fail_after_steps

    def forward(self, x):
        if self.training:
            self.steps += 1
            if self.fail_after_steps is not None and self.steps > self.fail_after_steps:
                raise _Interrupted(f"killed at step {self.steps}")
        return self.conv(x), torch.zeros((x.shape[0], 256))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return torch.float32

    @dtype.setter
    def dtype(self, value):
        pass

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)


def _fake_training_data(n=4, size=64, nchan=2):
    rng = np.random.default_rng(0)
    data = [rng.random((nchan, size, size), dtype="float32") for _ in range(n)]
    labels = []
    for _ in range(n):
        lbl = np.zeros((size, size), dtype="uint16")
        lbl[8:24, 8:24] = 1
        lbl[40:56, 40:56] = 2
        labels.append(lbl)
    return data, labels


_RESUME_STAGES = [
    {"name": "s1", "trainable_mode": "all", "n_epochs": 2, "learning_rate": 1e-4,
     "warmup_epochs": 0},
    {"name": "s2", "trainable_mode": "all", "n_epochs": 2, "learning_rate": 1e-5,
     "warmup_epochs": 0},
]


def _run(net, tmp_path, resume, resume_path):
    data, labels = _fake_training_data()
    return train.train_seg(
        net, train_data=data, train_labels=labels, channel_axis=0,
        training_stages=[dict(s) for s in _RESUME_STAGES],
        batch_size=2, bsize=32, normalize=False, min_train_masks=0,
        save_path=str(tmp_path), model_name="t", rescale=False,
        resume=resume, resume_every=1, resume_path=str(resume_path),
    )


def test_train_seg_resume_skips_completed_epochs(tmp_path):
    resume_path = tmp_path / "t_resume.pt"

    # die part-way through epoch 1, after epoch 0 has been checkpointed
    interrupted = _FakeSegNet(fail_after_steps=3)
    with pytest.raises(_Interrupted):
        _run(interrupted, tmp_path, resume=True, resume_path=resume_path)
    assert resume_path.exists()
    ckpt = torch.load(resume_path, weights_only=False)
    assert ckpt["global_epoch"] == 0
    loss_before = float(ckpt["train_losses"][0])
    assert loss_before > 0

    # resume with a fresh net and finish the schedule
    resumed = _FakeSegNet()
    _, train_losses, _ = _run(resumed, tmp_path, resume=True, resume_path=resume_path)

    assert len(train_losses) == 4
    # the completed epoch's loss came back from the checkpoint, not from rerunning it
    assert float(train_losses[0]) == loss_before
    # every later epoch ran, including across the stage boundary at epoch 2
    assert np.all(train_losses[1:] > 0)
    # 3 remaining epochs x 2 batches, and no epoch replayed
    assert resumed.steps == 6


def test_train_seg_without_resume_runs_every_epoch(tmp_path):
    resume_path = tmp_path / "fresh_resume.pt"
    net = _FakeSegNet()
    _, train_losses, _ = _run(net, tmp_path, resume=False, resume_path=resume_path)
    assert np.all(train_losses > 0)
    assert net.steps == 8
    assert resume_path.exists()


def test_timing_records_one_line_per_epoch_with_sane_phases(tmp_path):
    net = _FakeSegNet()
    _run(net, tmp_path, resume=False, resume_path=tmp_path / "unused_resume.pt")

    timing_file = tmp_path / "models" / "t_timing.jsonl"
    assert timing_file.exists()
    records = [json.loads(line) for line in timing_file.read_text().splitlines()]

    assert [r["epoch"] for r in records] == [0, 1, 2, 3]
    assert [r["stage"] for r in records] == ["s1", "s1", "s2", "s2"]
    for r in records:
        for key in ("wall_s", "batch_s", "augment_s", "step_s", "test_s", "save_s",
                    "imgs_per_s", "lr", "train_loss", "nimg", "batch_size"):
            assert key in r, key
        assert all(r[k] >= 0 for k in ("batch_s", "augment_s", "step_s", "test_s", "save_s"))
        # the phases are disjoint sub-intervals of the epoch
        assert r["batch_s"] + r["augment_s"] + r["step_s"] <= r["wall_s"] + 1e-6
        assert r["nimg"] == 4 and r["batch_size"] == 2
        assert r["peak_gpu_gb"] is None  # CPU run


def test_timing_appends_across_a_resume(tmp_path):
    resume_path = tmp_path / "t_resume.pt"
    timing_file = tmp_path / "models" / "t_timing.jsonl"

    interrupted = _FakeSegNet(fail_after_steps=3)
    with pytest.raises(_Interrupted):
        _run(interrupted, tmp_path, resume=True, resume_path=resume_path)
    before = len(timing_file.read_text().splitlines())
    assert before == 1

    _run(_FakeSegNet(), tmp_path, resume=True, resume_path=resume_path)
    records = [json.loads(line) for line in timing_file.read_text().splitlines()]
    # the pre-interruption epoch is kept, and the resumed epochs are appended
    assert len(records) == 4
    assert [r["epoch"] for r in records] == [0, 1, 2, 3]


def test_timing_can_be_disabled(tmp_path):
    data, labels = _fake_training_data()
    train.train_seg(
        _FakeSegNet(), train_data=data, train_labels=labels, channel_axis=0,
        training_stages=[dict(_RESUME_STAGES[0])], batch_size=2, bsize=32,
        normalize=False, min_train_masks=0, save_path=str(tmp_path),
        model_name="notiming", rescale=False, timing=False,
    )
    assert not (tmp_path / "models" / "notiming_timing.jsonl").exists()


def test_read_and_summarize_timing_roundtrip(tmp_path):
    net = _FakeSegNet()
    _run(net, tmp_path, resume=False, resume_path=tmp_path / "unused.pt")

    records = train.read_timing_records(tmp_path / "models" / "t_timing.jsonl")
    assert len(records) == 4

    summary = train.summarize_timing(records)
    assert summary["n_epochs"] == 4
    assert summary["total_s"] > 0
    assert summary["median_epoch_s"] > 0
    assert set(summary["phase_pct"]) == set(train.TIMING_PHASES)
    # shares are percentages of total wall time, so they cannot exceed 100
    assert 0 <= sum(summary["phase_pct"].values()) <= 100.5
    assert summary["bottleneck"] in train.TIMING_PHASES
    assert summary["hint"] == train.TIMING_HINTS[summary["bottleneck"]]
    assert summary["peak_gpu_gb"] is None  # CPU run


def test_summarize_timing_handles_empty_and_missing():
    assert train.summarize_timing([]) is None
    assert train.read_timing_records("/nonexistent/timing.jsonl") == []


def test_read_timing_records_skips_truncated_final_line(tmp_path):
    path = tmp_path / "timing.jsonl"
    good = {"epoch": 0, "wall_s": 1.0, "step_s": 0.5}
    path.write_text(json.dumps(good) + "\n" + '{"epoch": 1, "wall_s":')
    records = train.read_timing_records(path)
    assert records == [good]


def test_summarize_timing_picks_the_dominant_phase():
    records = [
        {"wall_s": 10.0, "batch_s": 1.0, "augment_s": 7.0, "step_s": 1.0,
         "imgs_per_s": 2.0, "peak_gpu_gb": 3.5},
        {"wall_s": 10.0, "batch_s": 1.0, "augment_s": 7.0, "step_s": 1.0,
         "imgs_per_s": 4.0, "peak_gpu_gb": 4.25},
    ]
    summary = train.summarize_timing(records)
    assert summary["bottleneck"] == "augment"
    assert summary["phase_pct"]["augment"] == 70.0
    assert summary["mean_imgs_per_s"] == 3.0
    assert summary["peak_gpu_gb"] == 4.25
    assert "prefetch" in summary["hint"]
