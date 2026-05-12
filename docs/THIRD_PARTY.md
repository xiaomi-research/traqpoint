# Third-party dependencies

This repository does **not** redistribute third-party source code or model weights.
Please download them from official sources and accept their licenses.

## One-click setup (source code)

From the repo root:

```bash
bash scripts/setup_third_party.sh
```

This clones pinned commits into `./third_party/`.

### Pinned versions

- LightGlue: `eb42fee2d71449efb0aa5c10549752b5d75384d8` (from https://github.com/cvg/LightGlue)
- DINOv3: `ffb4bb89c6558ca3244655c25a3955d01788b732` (from https://github.com/facebookresearch/dinov3)
- Hierarchical-Localization: `c13273bd0ecc2917a35910fd843712a1c6243193` (from https://github.com/cvg/Hierarchical-Localization)

## Required weights

Place the following files under `./third_party/`:

- `dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth`
- `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`
- (optional) `dinov2_vitb14_pretrain.pth`
- (optional) `aliked-n16.pth`

Exact usage depends on your config/model. If you change filenames/paths, update the corresponding code (e.g. `traqpoint/models/backbone_dinov3_conv.py`).
