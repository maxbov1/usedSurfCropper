# Optional SAM 2 backend

The app now has a lazy local SAM 2 backend. It does not download weights or
start model code until both environment variables are set:

```bash
export USEDSURF_SAM2_CHECKPOINT=/absolute/path/to/sam2.1_hiera_tiny.pt
export USEDSURF_SAM2_CONFIG=/absolute/path/to/sam2/configs/sam2.1/sam2.1_hiera_t.yaml
```

Use a separate Python environment for the model because PyTorch is a large
dependency. Follow Meta's official SAM 2 installation instructions, then on
Apple Silicon skip CUDA extension compilation:

```bash
git clone https://github.com/facebookresearch/sam2.git /path/to/sam2
cd /path/to/sam2
SAM2_BUILD_CUDA=0 pip install -e .
```

The backend uses MPS when available and otherwise CPU. It sends SAM 2 an
upright-board box prompt, validates the returned mask, applies the configured
5% padding, and rejects full-frame masks. It never replaces the manual review
path. SAM 2 is Apache-2.0 licensed; check the upstream repository for current
checkpoint names and download instructions.

The first quality experiment should compare SAM 2 masks against the crop
rectangles you annotate. Do not treat the website-reference images as labels:
they are not pixel-aligned originals.
