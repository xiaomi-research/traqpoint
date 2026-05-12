#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Pinned commits (keep these stable for reproducibility)
LIGHTGLUE_COMMIT="eb42fee2d71449efb0aa5c10549752b5d75384d8"
HLOC_COMMIT="c13273bd0ecc2917a35910fd843712a1c6243193"
DINOv3_COMMIT="ffb4bb89c6558ca3244655c25a3955d01788b732"

git submodule update --init --recursive

echo "[third_party] Checking out pinned commits..."

git -C third_party/LightGlue checkout "$LIGHTGLUE_COMMIT"
git -C third_party/Hierarchical-Localization checkout "$HLOC_COMMIT"
git -C third_party/facebookresearch_dinov3_main checkout "$DINOv3_COMMIT"

cat <<'EOF'

[third_party] Submodules initialized and pinned to specific commits.

Next steps (weights):
  Place required .pth weight files under ./third_party/ as described in docs/THIRD_PARTY.md.

EOF
