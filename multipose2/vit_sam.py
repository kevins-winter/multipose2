"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import torch
import logging
from segment_anything import sam_model_registry
torch.backends.cuda.matmul.allow_tf32 = True
from torch import nn 
import torch.nn.functional as F

vit_logger = logging.getLogger(__name__)

ADAPTER_TYPES = {"linear": 0, "msca_lite": 1}
ADAPTER_TYPE_IDS = {v: k for k, v in ADAPTER_TYPES.items()}


class LinearInputAdapter(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 3, kernel_size=1)
        self._init_identity()

    def _init_identity(self):
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            ncopy = min(self.proj.in_channels, 3)
            for c in range(ncopy):
                self.proj.weight[c, c, 0, 0] = 1.0

    def forward(self, x):
        return self.proj(x)


class MSCALiteInputAdapter(nn.Module):
    """Small multiscale spatial/channel attention adapter before SAM."""

    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or min(max(in_channels * 2, 8), 32)
        self.stem = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                               padding=1, groups=hidden_channels)
        self.conv5 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5,
                               padding=2, groups=hidden_channels)
        self.conv7 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=7,
                               padding=3, groups=hidden_channels)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 3, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(hidden_channels * 3, 3, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.stem(x), inplace=True)
        feats = torch.cat((self.conv3(x), self.conv5(x), self.conv7(x)), dim=1)
        feats = feats * self.channel_gate(feats)
        return self.out(feats)


def _make_input_adapter(in_channels, adapter_type):
    if adapter_type == "linear":
        return LinearInputAdapter(in_channels)
    if adapter_type == "msca_lite":
        return MSCALiteInputAdapter(in_channels)
    raise ValueError(
        f"Unknown adapter_type {adapter_type!r}; expected one of {sorted(ADAPTER_TYPES)}"
    )


class Transformer(nn.Module):
    def __init__(self, backbone="vit_l", ps=8, nout=3, bsize=256, rdrop=0.4,
                  checkpoint=None, dtype=torch.float32, in_channels=3,
                  adapter_type="linear"):
        super(Transformer, self).__init__()

        # instantiate the vit model, default to not loading SAM
        # checkpoint = sam_vit_l_0b3195.pth is standard pretrained SAM
        self.encoder = sam_model_registry[backbone](checkpoint).image_encoder
        w = self.encoder.patch_embed.proj.weight.detach()
        embed_dim = w.shape[0]
        self._set_input_adapter(in_channels, adapter_type)
        
        # change token size to ps x ps
        self.ps = ps
        self.encoder.patch_embed.proj = nn.Conv2d(3, embed_dim, stride=ps, kernel_size=ps)
        self.encoder.patch_embed.proj.weight.data = w[:,:,::16//ps,::16//ps]
        
        # adjust position embeddings for new bsize and new token size
        ds = (1024 // 16) // (bsize // ps)
        self.encoder.pos_embed = nn.Parameter(self.encoder.pos_embed[:,::ds,::ds], requires_grad=True)

        # readout weights for nout output channels
        # if nout is changed, weights will not load correctly from pretrained Cellpose-SAM
        self.nout = nout
        self.out = nn.Conv2d(256, self.nout * ps**2, kernel_size=1)

        # W2 reshapes token space to pixel space, not trainable
        self.W2 = nn.Parameter(torch.eye(self.nout * ps**2).reshape(self.nout*ps**2, self.nout, ps, ps), 
                               requires_grad=False)
        
        # fraction of layers to drop at random during training
        self.rdrop = rdrop

        # average diameter of ROIs from training images from fine-tuning 
        self.diam_labels = nn.Parameter(torch.tensor([30.]), requires_grad=False)
        # average diameter of ROIs during main training
        self.diam_mean = nn.Parameter(torch.tensor([30.]), requires_grad=False)
        
        # set attention to global in every layer
        for blk in self.encoder.blocks:
            blk.window_size = 0

        self._dtype = dtype
        if dtype != torch.float32:
            self.dtype = dtype

    def _set_input_adapter(self, in_channels, adapter_type=None):
        adapter_type = adapter_type or getattr(self, "adapter_type", "linear")
        if adapter_type not in ADAPTER_TYPES:
            raise ValueError(
                f"Unknown adapter_type {adapter_type!r}; expected one of {sorted(ADAPTER_TYPES)}"
            )
        self.in_channels = int(in_channels)
        self.adapter_type = adapter_type
        self.input_adapter = _make_input_adapter(self.in_channels, self.adapter_type)
        self.register_buffer(
            "_adapter_in_channels",
            torch.tensor([self.in_channels], dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "_adapter_type_id",
            torch.tensor([ADAPTER_TYPES[self.adapter_type]], dtype=torch.int64),
            persistent=True,
        )

    def set_input_adapter(self, in_channels, adapter_type=None):
        device = self.device
        dtype = self.dtype
        self._set_input_adapter(in_channels, adapter_type)
        self.input_adapter.to(device=device, dtype=dtype)
        self._adapter_in_channels = self._adapter_in_channels.to(device)
        self._adapter_type_id = self._adapter_type_id.to(device)

    def forward(self, x):      
        # same progression as SAM until readout
        x = self.input_adapter(x)
        x = self.encoder.patch_embed(x)
        
        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed
        
        if self.training and self.rdrop > 0:
            nlay = len(self.encoder.blocks)
            rdrop = (torch.rand((len(x), nlay), device=x.device) < 
                     torch.linspace(0, self.rdrop, nlay, device=x.device)).to(x.dtype)
            for i, blk in enumerate(self.encoder.blocks):            
                mask = rdrop[:,i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x = x * mask + blk(x) * (1-mask)
        else:
            for blk in self.encoder.blocks:
                x = blk(x)

        x = self.encoder.neck(x.permute(0, 3, 1, 2))

        # readout is changed here
        x1 = self.out(x)
        x1 = F.conv_transpose2d(x1, self.W2, stride = self.ps, padding = 0)
        
        # maintain the second output of feature size 256 for backwards compatibility
           
        return x1, torch.zeros((x.shape[0], 256), device=x.device)
    
    def load_model(self, PATH, device, strict = False):
        state_dict = torch.load(PATH, map_location = device, weights_only=True)
        keys = [k for k in state_dict.keys()]

        # loudly fail on attempt to load not cp4 model: 
        w2_data = state_dict.get('W2', None)
        if w2_data == None:
            raise ValueError('This model does not appear to be a CP4 model. CP3 models are not compatible with CP4.')

        checkpoint_adapter_id = state_dict.get("_adapter_type_id", None)
        skip_adapter_weights = False
        if checkpoint_adapter_id is not None:
            checkpoint_adapter_type = ADAPTER_TYPE_IDS.get(int(checkpoint_adapter_id.item()))
            if checkpoint_adapter_type != self.adapter_type:
                skip_adapter_weights = True
                vit_logger.warning(
                    "Checkpoint %s was saved with adapter_type=%s; current model uses adapter_type=%s. Adapter weights will be initialized for the current adapter.",
                    PATH,
                    checkpoint_adapter_type,
                    self.adapter_type,
                )

        # models are always saved as float32
        if keys[0][:7] == "module.":
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v
            state_dict = new_state_dict
        state_dict, skipped = self._filter_incompatible_adapter_keys(
            state_dict, skip_adapter_weights=skip_adapter_weights
        )
        incompatible = self.load_state_dict(state_dict, strict = strict)

        missing_adapter = sorted(
            key for key in incompatible.missing_keys if key.startswith("input_adapter.")
        )
        if missing_adapter or skipped:
            vit_logger.warning(
                "Checkpoint %s does not contain compatible adapter weights (%s); the input adapter will use initialized weights.",
                PATH,
                ", ".join(sorted(set(missing_adapter + skipped))),
            )
        if incompatible.unexpected_keys:
            vit_logger.warning(
                "Unexpected checkpoint keys while loading %s: %s",
                PATH,
                ", ".join(incompatible.unexpected_keys),
            )

        if self.dtype != torch.float32:
            self = self.to(self.dtype)

    def _filter_incompatible_adapter_keys(self, state_dict, skip_adapter_weights=False):
        current = self.state_dict()
        filtered, skipped = {}, []
        for key, value in state_dict.items():
            if skip_adapter_weights and key.startswith("input_adapter."):
                skipped.append(key)
                continue
            if key.startswith("input_adapter.") and key in current:
                if current[key].shape != value.shape:
                    skipped.append(key)
                    continue
            if key in {"_adapter_in_channels", "_adapter_type_id"} and key in current:
                if current[key].shape != value.shape or not torch.equal(
                    current[key].cpu(), value.cpu()
                ):
                    skipped.append(key)
                    continue
            filtered[key] = value
        return filtered, skipped

    @property
    def dtype(self):
        """
        Get the data type of the model.

        Returns:
            torch.dtype: The data type of the model.
        """
        return self._dtype
    
    @dtype.setter
    def dtype(self, value):
        """
        Set the data type of the model.

        Args:
            value (torch.dtype): The data type to set for the model.
        """
        if self._dtype != value:
            self.to(value)
            self._dtype = value
    
    @property
    def device(self):
        """
        Get the device of the model.

        Returns:
            torch.device: The device of the model.
        """
        return next(self.parameters()).device

    def save_model(self, filename):
        """
        Save the model to a file.

        Args:
            filename (str): The path to the file where the model will be saved.
        """
        torch.save(self.state_dict(), filename)



class CPnetBioImageIO(Transformer):
    """
    A subclass of the CP-SAM model compatible with the BioImage.IO Spec.

    This subclass addresses the limitation of CPnet's incompatibility with the BioImage.IO Spec,
    allowing the CPnet model to use the weights uploaded to the BioImage.IO Model Zoo.
    """

    def forward(self, x):
        """
        Perform a forward pass of the CPnet model and return unpacked tensors.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the output tensor, style tensor, and downsampled tensors.
        """
        output_tensor, style_tensor, downsampled_tensors = super().forward(x)
        return output_tensor, style_tensor, *downsampled_tensors
    

    def load_model(self, filename, device=None):
        """
        Load the model from a file.

        Args:
            filename (str): The path to the file where the model is saved.
            device (torch.device, optional): The device to load the model on. Defaults to None.
        """
        if (device is not None) and (device.type != "cpu"):
            state_dict = torch.load(filename, map_location=device, weights_only=True)
        else:
            self.__init__(self.nout)
            state_dict = torch.load(filename, map_location=torch.device("cpu"), 
                                    weights_only=True)

        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict):
        """
        Load the state dictionary into the model.

        This method overrides the default `load_state_dict` to handle Cellpose's custom
        loading mechanism and ensures compatibility with BioImage.IO Core.

        Args:
            state_dict (Mapping[str, Any]): A state dictionary to load into the model
        """
        if state_dict["output.2.weight"].shape[0] != self.nout:
            for name in self.state_dict():
                if "output" not in name:
                    self.state_dict()[name].copy_(state_dict[name])
        else:
            super().load_state_dict(
                {name: param for name, param in state_dict.items()},
                strict=False)


    
