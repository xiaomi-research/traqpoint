# third_party (git submodules)

Third-party dependencies are managed as **git submodules**, pinned to specific commits for reproducibility.

## Setup

```bash
git submodule update --init --recursive
```

Or equivalently:

```bash
bash scripts/setup_third_party.sh
```

## Submodules

| Directory | Repository | License |
|-----------|-----------|---------|
| `LightGlue/` | [cvg/LightGlue](https://github.com/cvg/LightGlue) | Apache-2.0 |
| `Hierarchical-Localization/` | [cvg/Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization) | Apache-2.0 |
| `facebookresearch_dinov3_main/` | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | Apache-2.0 |

## Weights (not versioned)

Pretrained weight files (`*.pth`, `*.pt`, `*.safetensors`) are git-ignored and must be downloaded manually. See `docs/THIRD_PARTY.md` for the required weight files and download links.
