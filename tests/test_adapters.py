import torch

from conftest import MockTransformer
from multipose2 import vit_sam


def test_linear_adapter_shapes():
    for nchan in (1, 3, 5, 8):
        adapter = vit_sam.LinearInputAdapter(nchan)
        x = torch.zeros((2, nchan, 32, 32))
        assert adapter(x).shape == (2, 3, 32, 32)


def test_msca_lite_adapter_shape():
    adapter = vit_sam.MSCALiteInputAdapter(5)
    x = torch.zeros((2, 5, 32, 32))
    assert adapter(x).shape == (2, 3, 32, 32)


def test_old_checkpoint_without_adapter_loads(tmp_path):
    src = MockTransformer(1, in_channels=3)
    state = {
        k: v for k, v in src.state_dict().items()
        if not k.startswith("input_adapter.") and not k.startswith("_adapter_")
    }
    path = tmp_path / "old_cpsam_like.pt"
    torch.save(state, path)

    for adapter_type in ("linear", "msca_lite"):
        dst = MockTransformer(1, in_channels=8, adapter_type=adapter_type)
        dst.load_model(path, device=dst.device)
        assert dst.in_channels == 8
        assert dst.adapter_type == adapter_type


def test_multichannel_checkpoint_loads_with_matching_and_mismatched_nchan(tmp_path):
    src = MockTransformer(1, in_channels=5, adapter_type="linear")
    path = tmp_path / "linear_5chan.pt"
    src.save_model(path)

    matching = MockTransformer(1, in_channels=5, adapter_type="linear")
    matching.load_model(path, device=matching.device)
    assert matching.in_channels == 5

    mismatched = MockTransformer(1, in_channels=8, adapter_type="linear")
    mismatched.load_model(path, device=mismatched.device)
    assert mismatched.in_channels == 8


def _reference_forward(net, x):
    """Inline copy of the pre-split forward pass, kept as a regression reference."""
    x = net.input_adapter(x)
    x = net.encoder.patch_embed(x)
    if net.encoder.pos_embed is not None:
        x = x + net.encoder.pos_embed
    for blk in net.encoder.blocks:
        x = blk(x)
    x = net.encoder.neck(x.permute(0, 3, 1, 2))
    x1 = net.out(x)
    x1 = torch.nn.functional.conv_transpose2d(x1, net.W2, stride=net.ps, padding=0)
    return x1, torch.zeros((x.shape[0], 256), device=x.device)


def test_forward_matches_pre_split_reference():
    net = vit_sam.Transformer(in_channels=3, bsize=64)
    net.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        got, got_style = net(x)
        want, want_style = _reference_forward(net, x)
    assert torch.equal(got, want)
    assert got_style.shape == want_style.shape


def test_trunk_head_split_composes_to_forward():
    net = vit_sam.Transformer(in_channels=5, bsize=64)
    net.eval()
    x = torch.randn(2, 5, 64, 64)
    with torch.no_grad():
        feat = net.forward_trunk(net.input_adapter(x))
        assert feat.shape == (2, 256, 8, 8)
        assert torch.equal(net.forward_head(feat), net(x)[0])
