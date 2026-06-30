"""Model factory: encoder+decoder combinations with freezing logic."""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import timm
import pathlib


# ---------- Encoder registry ----------

ENCODER_REGISTRY = {
    # ImageNet CNN family
    "resnet34": {"family": "ImageNet CNN", "smp_name": "resnet34", "pretrained": "imagenet"},
    "resnet50": {"family": "ImageNet CNN", "smp_name": "resnet50", "pretrained": "imagenet"},
    "resnet50_random": {"family": "ImageNet CNN", "smp_name": "resnet50", "pretrained": None},
    "efficientnet-b3": {"family": "ImageNet CNN", "smp_name": "efficientnet-b3", "pretrained": "imagenet"},
    "efficientnet-b5": {"family": "ImageNet CNN", "smp_name": "efficientnet-b5", "pretrained": "imagenet"},
    "tu-efficientnetv2_s": {"family": "ImageNet CNN", "smp_name": "tu-tf_efficientnetv2_s.in1k", "pretrained": "imagenet"},
    "tu-efficientnetv2_m": {"family": "ImageNet CNN", "smp_name": "tu-tf_efficientnetv2_m.in1k", "pretrained": "imagenet"},
    "tu-efficientnetv2_l": {"family": "ImageNet CNN", "smp_name": "tu-tf_efficientnetv2_l.in1k", "pretrained": "imagenet"},
    # ConvNeXt family
    "tu-convnext_tiny": {"family": "ConvNeXt", "smp_name": "tu-convnext_tiny", "pretrained": "imagenet"},
    "tu-convnext_small": {"family": "ConvNeXt", "smp_name": "tu-convnext_small", "pretrained": "imagenet"},
    # Foundation ViT family
    "phikon-v2": {"family": "Histology ViT", "type": "foundation", "hf_repo": "owkin/phikon-v2"},
    "uni": {"family": "Histology ViT", "type": "foundation", "hf_repo": "MahmoodLab/UNI"},
    "conch": {"family": "Histology ViT", "type": "foundation", "hf_repo": "MahmoodLab/CONCH"},
    "h-optimus-0": {"family": "Histology ViT", "type": "foundation", "hf_repo": "bioptimus/H-optimus-0"},
    "virchow2": {"family": "Histology ViT", "type": "foundation", "hf_repo": "paige-ai/Virchow2"},
}

# Decoder registry
DECODER_REGISTRY = {
    "Unet": smp.Unet,
    "UnetPlusPlus": smp.UnetPlusPlus,
    "MAnet": smp.MAnet,
    "FPN": smp.FPN,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
}


def _get_hf_token():
    """Read HuggingFace token."""
    token_path = pathlib.Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    import os
    return os.environ.get("HF_TOKEN")


class ViTUNetAdapter(nn.Module):
    """Adapter to connect ViT patch-token outputs to a UNet-style decoder."""

    def __init__(self, vit_model, embed_dim, patch_size, img_size=512, num_classes=3):
        super().__init__()
        self.vit = vit_model
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size

        # Multi-scale projection layers
        self.proj_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(embed_dim, 512, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(embed_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(embed_dim, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(embed_dim, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)),
        ])

        # Decoder
        self.up1 = self._up_block(512, 256)
        self.up2 = self._up_block(256 + 256, 128)
        self.up3 = self._up_block(128 + 128, 64)
        self.up4 = self._up_block(64 + 64, 32)
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head = nn.Conv2d(32, num_classes, 1)

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _extract_features(self, x):
        """Extract patch tokens from ViT."""
        features = self.vit.forward_features(x)
        if isinstance(features, dict):
            tokens = features.get("x", features.get("last_hidden_state", None))
            if tokens is None:
                tokens = list(features.values())[0]
        elif isinstance(features, (tuple, list)):
            tokens = features[0]
        else:
            tokens = features

        # Remove CLS/register tokens
        expected = self.grid_size ** 2
        if tokens.shape[1] > expected:
            # Try removing first token (CLS), check if remainder matches
            if tokens.shape[1] - 1 == expected:
                tokens = tokens[:, 1:, :]
            else:
                # Take last N tokens (patch tokens typically come last)
                tokens = tokens[:, -expected:, :]

        B = tokens.shape[0]
        spatial = tokens.permute(0, 2, 1).reshape(B, self.embed_dim, self.grid_size, self.grid_size)
        return spatial

    def forward(self, x):
        # If input doesn't match expected size, resize
        input_h, input_w = x.shape[2], x.shape[3]
        if input_h != self.img_size or input_w != self.img_size:
            x = nn.functional.interpolate(x, size=(self.img_size, self.img_size),
                                          mode="bilinear", align_corners=False)

        spatial = self._extract_features(x)

        f1 = self.proj_layers[0](spatial)
        f2 = self.proj_layers[1](spatial)
        f3 = self.proj_layers[2](spatial)
        f4 = self.proj_layers[3](spatial)

        x = self.up1(f1)
        x = torch.cat([x, nn.functional.interpolate(f2, size=x.shape[2:], mode="bilinear", align_corners=False)], dim=1)
        x = self.up2(x)
        x = torch.cat([x, nn.functional.interpolate(f3, size=x.shape[2:], mode="bilinear", align_corners=False)], dim=1)
        x = self.up3(x)
        x = torch.cat([x, nn.functional.interpolate(f4, size=x.shape[2:], mode="bilinear", align_corners=False)], dim=1)
        x = self.up4(x)
        x = self.final_up(x)

        # Always interpolate to 512×512 for consistent output
        if x.shape[2:] != (512, 512):
            x = nn.functional.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)

        return self.head(x)


def _get_patch_size(vit):
    """Extract patch size from a timm ViT model."""
    ps = vit.patch_embed.patch_size
    return ps[0] if hasattr(ps, '__len__') else ps


def _load_foundation_model(encoder_name, img_size=512, num_classes=3, freeze=True):
    """Load a foundation ViT model and wrap with UNet adapter."""
    token = _get_hf_token()

    try:
        if encoder_name == "phikon-v2":
            # Phikon-v2 is a DINOv2 model — load via transformers
            from transformers import AutoModel
            hf_model = AutoModel.from_pretrained("owkin/phikon-v2", token=token, trust_remote_code=True)
            # Wrap transformers model to have forward_features interface
            class _PhikonWrapper(nn.Module):
                def __init__(self, hf_model):
                    super().__init__()
                    self.model = hf_model
                def forward_features(self, x):
                    out = self.model(x)
                    return out.last_hidden_state  # (B, 1+N, D) with CLS
            vit = _PhikonWrapper(hf_model)
            embed_dim = hf_model.config.hidden_size
            patch_size = hf_model.config.patch_size

        elif encoder_name == "uni":
            vit = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True,
                                     img_size=img_size, init_values=1e-5, dynamic_img_size=True)
            embed_dim = vit.embed_dim
            patch_size = _get_patch_size(vit)

        elif encoder_name == "conch":
            # CONCH vision encoder is ViT-B/16 (768 dim), native 448×448 (28×28 patches)
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(repo_id="MahmoodLab/CONCH",
                                         filename="pytorch_model.bin", token=token)
            # Load at native 448 resolution so pos_embed matches, then use dynamic_img_size for 512
            vit = timm.create_model("vit_base_patch16_224", pretrained=False,
                                     img_size=448, dynamic_img_size=True)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            vision_state = {}
            for k, v in state.items():
                if k.startswith("visual.trunk."):
                    vision_state[k.replace("visual.trunk.", "")] = v
            msg = vit.load_state_dict(vision_state, strict=False)
            print(f"  CONCH loaded: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
            embed_dim = vit.embed_dim
            patch_size = 16

        elif encoder_name == "h-optimus-0":
            # H-Optimus-0 is ViT-g/14 — use 518 (37×14) to be divisible by patch_size=14
            vit_img_size = 518  # 37 * 14
            vit = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True,
                                     img_size=vit_img_size, init_values=1e-5, dynamic_img_size=True)
            embed_dim = vit.embed_dim
            patch_size = _get_patch_size(vit)
            img_size = vit_img_size

        elif encoder_name == "virchow2":
            # Virchow2 is ViT-H/14 with SwiGLU MLP
            # Load at native 224 so pos_embed matches, then use dynamic_img_size for 518
            from timm.layers import SwiGLUPacked
            from huggingface_hub import hf_hub_download
            vit_img_size = 518  # 37 * 14
            vit = timm.create_model("vit_huge_patch14_224", pretrained=False,
                                     img_size=224, init_values=1e-5, num_classes=0,
                                     reg_tokens=4, mlp_ratio=5.3375,
                                     global_pool="", dynamic_img_size=True,
                                     mlp_layer=SwiGLUPacked)
            ckpt = hf_hub_download("paige-ai/Virchow2", "pytorch_model.bin", token=token)
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            vit.load_state_dict(state, strict=True)
            embed_dim = vit.embed_dim
            patch_size = _get_patch_size(vit)
            img_size = vit_img_size

        else:
            raise ValueError(f"Unknown foundation model: {encoder_name}")

    except Exception as e:
        print(f"  Failed to load {encoder_name}: {e}")
        return None

    if freeze:
        for param in vit.parameters():
            param.requires_grad = False

    model = ViTUNetAdapter(vit, embed_dim, patch_size, img_size=img_size, num_classes=num_classes)
    return model


def create_model(encoder_name, decoder_name="Unet", num_classes=3, freeze_encoder=True,
                 pretrained=True, img_size=512):
    """Create a segmentation model."""
    info = ENCODER_REGISTRY[encoder_name]

    # Foundation model path
    if info.get("type") == "foundation":
        model = _load_foundation_model(encoder_name, img_size=img_size,
                                        num_classes=num_classes, freeze=freeze_encoder)
        if model is None:
            raise RuntimeError(f"Could not load foundation model: {encoder_name}")
        return model

    # SMP model path
    smp_name = info["smp_name"]
    weights = info["pretrained"] if pretrained else None
    if encoder_name == "resnet50_random":
        weights = None

    decoder_cls = DECODER_REGISTRY[decoder_name]

    model = decoder_cls(
        encoder_name=smp_name,
        encoder_weights=weights,
        in_channels=3,
        classes=num_classes,
    )

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    return model


def count_parameters(model):
    """Return (total_params_M, trainable_params_M)."""
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return round(total, 2), round(trainable, 2)
