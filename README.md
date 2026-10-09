# UsedSurf inventory organizer

Local macOS proof of concept for organizing surfboard inventory photos. It keeps originals in `input/`, stores review state in SQLite, and writes approved galleries to `output/shaper-model-sku/`.

## Run it

```bash
cd /Users/maxboving/projects/usedSurf
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./start.sh
```

`start.sh` runs a local health check first and then starts the app. Run
`./doctor.sh` by itself when diagnosing the installation. The app is
self-contained: SQLite state lives in `data/usedsurf.sqlite3`, and each crop,
retest, or cleanup operation is recorded in the `runs` and `artifacts` tables.

The cropper optionally uses Ultralytics YOLO11 Nano (`yolo11n.pt`) as a local,
coarse `surfboard` detector. The first crop run downloads the 5.4 MB weights
into this project; later runs are local. On Apple Silicon it selects PyTorch
MPS when available and otherwise uses CPU. YOLO only narrows the OpenCV search
region; OpenCV still determines the final contour and padding. YOLO11 is
available under Ultralytics' AGPL-3.0 license (or a separate Enterprise
license), so review that license before commercial distribution.

## Build the macOS app

The distributable is a double-clickable `UsedSurf.app`. The bundle contains
the application, YOLO weights, and a small crop-regression fixture pack. User
photos, SQLite state, archives, corrections, and downloaded model copies live
outside the bundle at `~/Library/Application Support/UsedSurf/`, so installing
an update does not overwrite inventory work.

To build locally on a Mac:

```bash
python3 -m venv .venv-build
source .venv-build/bin/activate
pip install -r requirements.txt -r requirements-build.txt
bash scripts/build_mac_app.sh
open dist/UsedSurf.app
```

GitHub Actions builds separate Intel and Apple Silicon ZIPs on version tags.
After the repository is configured, set `USED_SURF_GITHUB_REPO=owner/repo`
when launching locally if you want Advanced Debug to show GitHub HEAD update
status. The packaged app performs that check automatically; it is best effort
and never blocks startup. It reports update availability; it does not silently
replace the app bundle.

For a non-technical Mac user, the one-command installer clones the repository,
installs the build dependencies, builds the native app, and places the
clickable application in `/Applications`:

```bash
curl -fsSL https://raw.githubusercontent.com/maxbov1/usedSurfCropper/main/scripts/install_mac.sh | bash
```

The installer may ask for the Mac administrator password when replacing an
existing `/Applications/UsedSurf.app`. It does not put the build environment or
source checkout in `/Applications`; user photos, review state, and corrections
remain in `~/Library/Application Support/UsedSurf/`.

Open <http://127.0.0.1:5051>, upload a JPEG batch, and click **Continue**. Grouping uses capture time as the primary order, inventory cards as board start markers, and a lightweight OpenCV appearance signature to split an obvious visual change in a long cardless run. Ambiguous card conflicts stay in the review UI. Port 5051 is the default local port. For long local batches, use `scripts/run_local.sh` instead; it runs the app under macOS `caffeinate` so sleep does not suspend the local server. Use the project environment so local Tesseract OCR is available. There is deliberately no folder watcher: processing begins only after that explicit action.

Optional OCR/HEIC capabilities on Apple Silicon:

```bash
brew install tesseract
pip install pytesseract pillow-heif
```

OpenCV is required for the current silhouette baseline and is Apache-2.0. Tesseract is Apache-2.0; pillow-heif is BSD-3-Clause and uses the LGPL-licensed libheif library. The app calls these locally only. Without OCR/HEIC extras, JPEG/PNG review and export still work, while OCR and HEIC decoding are marked unavailable rather than guessed.

## Run the crop fixture runner

```bash
python3 scripts/run_crop_fixtures.py
```

This processes the downloaded sample pack without changing it and writes `data/crop-eval/latest.json`, annotated originals with red rectangles under `data/crop-eval/previews/<board>/`, and the actual proposed crop JPEGs under `data/crop-eval/crops/<board>/`. It is a behavioral runner, not an accuracy benchmark yet: annotate expected rectangles in the fixtures before calculating pass/fail metrics.

## Build a shot-type labeling set

UsedSurf listing photos can be downloaded locally for classifier training. Product gallery order is preserved, and each case keeps its product URL and gallery position. Inventory cards are not expected in this web set.

```bash
.venv-sam2/bin/python scripts/import_usedsurf.py --pages 6 --max-boards 100
```

Then open `http://127.0.0.1:5051/labeler` and label each image with Full board, Side profile, or Fin/detail. Keyboard shortcuts are `1`, `2`, `3`, `S` for skip, and `U` for undo. Labels live in `data/shot-labels/manifest.json`, separate from production grouping and crop corrections.

## Train the shot classifier

```bash
.venv-sam2/bin/python scripts/train_shot_classifier.py --epochs 60
```

This uses a local torchvision MobileNetV3-Small ImageNet embedding and trains only a small linear layer. By default it uses `label_source=human` records only and writes `data/shot-labels/classifier-human-v1.json` and `.pt`; the original gallery-order guesses remain separate. The split is by product listing, so photos from one board never appear in both train and test. To intentionally run the weak labels, use `--label-source gallery_order`. The downloaded MobileNetV3-Small weights are about 10 MB and run on CPU; they are cached under `~/.cache/torch/`.

## Notes

- Capture order uses EXIF `DateTimeOriginal`/`DateTimeDigitized` first. File creation/copy time is never used as capture time.
- All crop coordinates are pixels in the orientation-normalized image: `(x, y, width, height)`, origin top-left.
- Card photos are retained in the manifest but excluded from gallery exports.
- Approved output is written to a temporary sibling folder and renamed into place. Existing approved folders are never overwritten; reruns reuse saved corrections.
- With no sample images in this repository, no detection accuracy is claimed. Put a representative shoot in `input/` and record corrections in the review UI before relying on grouping.
- Cleanup moves originals into `data/archive/` and creates smaller, verified
  derivatives plus a `retention-manifest.json`. It does not delete originals.
  To inspect what is eligible for removal, run:

  ```bash
  python3 scripts/compact_archive.py data/archive/cleanup-YYYYMMDD-HHMMSS
  ```

  Only after reviewing the manifest should `--apply` be used. Card and
  human-reviewed sources are retained as valuable training/reference data.

## Validation checklist

Use a whole shoot as one evaluation set. Shuffle arrival order, remove one EXIF timestamp, include an unreadable card, a side profile with fins, and a detail shot. Confirm the review queue surfaces those cases, the fins remain inside the crop, details keep their original framing, and rerunning preserves edits and leaves `input/` byte-for-byte unchanged.

For a dependency-light local regression run, use:

```bash
python3 -m unittest discover -s tests -q
```

On some macOS Python installations, pytest's terminal-capture dependency imports
the system `readline` module and can crash before collection. That is an
environment-level native crash, so the local validation path uses unittest
until the Python environment is repaired.

To safely reduce generated crop history while retaining the current review run,
true-test references, annotations, calibration, reports, logs, and reviewed
artifacts:

```bash
python3 scripts/compact_crop_history.py
python3 scripts/compact_crop_history.py --apply
```

The first command is a dry run. The second removes only old generated proposal
and overlay images and writes a cleanup manifest under `data/crop-eval/`.
