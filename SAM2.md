# Optional SAM 2 backend

The app has a lazy local SAM 2 backend. It does not download weights or start
model code until both environment variables are set:

```bash
export USEDSURF_SAM2_CHECKPOINT=/absolute/path/to/sam2.1_hiera_tiny.pt
export USEDSURF_SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml
```

`USEDSURF_SAM2_CHECKPOINT` is an absolute filesystem path. `USEDSURF_SAM2_CONFIG`
is **not** a filesystem path — SAM 2 loads it through Hydra, which resolves it
relative to the installed `sam2` package's own config directory. Passing an
absolute path here fails with "Cannot find primary config"; the value above
(`configs/sam2.1/sam2.1_hiera_t.yaml`) is correct regardless of where the repo
is cloned.

## One-time setup (verified on Apple Silicon)

Use a separate Python environment for this, since PyTorch is a large
dependency. This repo already has one at `.venv-sam2` with the rest of the
app's requirements installed alongside it, so `.venv-sam2` can run the whole
app, not just SAM 2.

1. **Get the real checkpoint.** The file matters: a bad download can silently
   save an error page instead of weights (this repo's `sam2.1_hiera_tiny.pt`
   was a 111-byte S3 `AccessDenied` response at one point, not a checkpoint —
   always sanity-check the file size, ~149MB for the tiny checkpoint).

   ```bash
   curl -L -o sam2.1_hiera_tiny.pt \
     https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
   ```

   The checkpoint is intentionally **not** committed to git (see `.gitignore`)
   — a ~150MB binary doesn't belong in history. Re-run this step after a fresh
   clone, or on any machine that needs it.

2. **Install the `sam2` package** (CUDA extensions skipped on Apple Silicon):

   ```bash
   git clone https://github.com/facebookresearch/sam2.git /path/to/sam2-src
   SAM2_BUILD_CUDA=0 .venv-sam2/bin/pip install -e /path/to/sam2-src
   .venv-sam2/bin/pip install "lap>=0.5.12"   # ultralytics dep, not pulled in automatically
   ```

3. **Set the env vars and run.** `start.sh` auto-selects `.venv-sam2` instead
   of `.venv` whenever `USEDSURF_SAM2_CHECKPOINT` is set, so no manual `source
   .venv-sam2/bin/activate` is needed — just export the two variables before
   running it:

   ```bash
   export USEDSURF_SAM2_CHECKPOINT=/absolute/path/to/sam2.1_hiera_tiny.pt
   export USEDSURF_SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml
   ./start.sh
   ```

   (If `.venv` doesn't have torch/sam2 installed and you forget to set
   `USEDSURF_SAM2_CHECKPOINT`, `start.sh` picks `.venv` and the app runs fine
   — just without SAM 2, falling back to the OpenCV-only board detector with
   no error. Nothing breaks either way; you just don't get the benefit below.)

## What it's used for

Two independent call sites, both optional/lazy — with no checkpoint
configured, both are no-ops and behavior is identical to before SAM 2 existed:

- **Grouping (`organizer/web.py`, the classification loop that assigns
  `shot_type` and board-identity signatures):** `board_region_mask()` gets a
  precise board-only mask, used two ways:
  - `board_identity_signature()`/`board_local_signature()` blank non-board
    pixels before computing the color/edge histogram, so the comparison is
    actually "does this look like the same board," not "does this look like
    the same wall."
  - `board_mask_width_fraction()` corrects `full_board` → `side_profile`
    when OpenCV's own candidate box was wrong (measured the wall, not the
    board) but SAM 2's real board footprint is narrow. This is the fix for a
    real production bug: two standing side-rail shots were misclassified as
    `full_board`, entered global visual matching, and got pulled into an
    unrelated board's group.
- **Final listing crop (`scripts/run_batch_crop.py` → `organizer/crop.py`'s
  `suggested_crop()`):** uses SAM 2 as a fallback board-boundary source when
  OpenCV's own detector finds nothing usable. `run_batch_crop.py` also trusts
  the photo's already-corrected `shot_type` (from grouping, or a human
  review correction) over re-deriving it from a fresh, uncorrected
  `classify_shot()` call.

## Performance (tiny checkpoint, M-series Mac, MPS)

Measured against this repo's real 129-photo input set: ~0.7s/photo after a
one-time ~6s model warmup on the first call. A full batch of ~120 full-board/
side-profile photos (cards and detail shots skip SAM 2) took ~3.5 minutes
total. Found 3 genuine shot-type misclassifications in that set, all
visually confirmed as real standing side-rail shots.

## Notes

The backend uses MPS when available and otherwise CPU. It sends SAM 2 a
board box prompt, validates the returned mask, applies the configured 5%
padding, and rejects full-frame masks. It never replaces the manual review
path. SAM 2 is Apache-2.0 licensed; check the upstream repository for
current checkpoint names and download instructions, since URLs do change.

The first quality experiment should compare SAM 2 masks against the crop
rectangles you annotate. Do not treat the website-reference images as labels:
they are not pixel-aligned originals.
