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
