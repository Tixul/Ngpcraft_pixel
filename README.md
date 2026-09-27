<img width="1826" height="1138" alt="Capture d&#39;écran 2026-09-27 203024" src="https://github.com/user-attachments/assets/d9ef601b-06a5-4b50-bb76-5972eb621e4b" />
# NgpCraft Pixel

**Convert photos and images into Neo Geo Pocket Color-ready sprites and backgrounds.**

A Qt desktop tool that takes arbitrary source images (photos, artwork, references) and produces PNGs that respect the strict NGPC hardware constraints: 4-bit RGB palettes, 3 colors per sprite/tile, 16-palette budget, 8×8 tile alignment. Output is ready to feed directly into the standard `ngpc_sprite_export.py` / `ngpc_tilemap.py` toolchain.

Part of the NgpCraft ecosystem — sits upstream of the compiler/linker/asset tools.

---

## Why

NGPC homebrew needs sprites and tilemaps in an awkward format:

- 4-bit per channel RGB (`0x0BGR`, 4096 possible colors)
- Each sprite/tile uses at most **3 visible colors** + 1 transparent index
- At most **16 palettes** of 4 colors each, shared across a layer
- Sprites: up to 160×152 px. Backgrounds: up to 256×256 px (one scroll plane), 8×8 tile grid, ≤512 unique tiles in VRAM

Preparing a clean quantized PNG by hand in Aseprite or Photoshop is tedious, and batch tools (Pyxelate, pixelit) don't respect the NGPC palette budget or tile layout. NgpCraft Pixel automates that last mile: load a reference, tweak a few sliders, export a PNG that the toolchain will accept without complaint.

---

## Features

### Sprite mode
- Target any dimension from 8×8 up to 160×152, in multiples of 8
- Presets for 16×16 and 32×32
- 3-color (native) or 6-color (dual-layer, for `--layer2` in `ngpc_sprite_export.py`) output
- **Automatic cutout** via OpenCV GrabCut, seeded from a center rectangle
- **Interactive matting**: brush strokes to mark "keep" / "remove" areas — re-runs GrabCut with your hints in mask-init mode
- **Live palette panel** showing the chosen 3 or 6 colors, each labeled with its NGPC RGB444 hex (`0x0F00` for pure blue, etc.)
- **Pixel-level retouching** directly on the preview: click a palette swatch, click pixels in the zoomed preview. Alt+click acts as an eyedropper.
- Option to keep manual retouches persistent across reprocess (slider tweaks no longer wipe pixel edits)
- **Dual-layer preview panel** in 6-color mode — see Layer A and Layer B side by side before export
- Export to PNG, or to two PNGs (layer A + layer B) for dual-layer builds
- **"Copy NGPC palette"** toolbar action — copies the palette in `--fixed-palette` format (`0000,XXXX,YYYY,ZZZZ`) ready to paste into `ngpc_sprite_export.py` invocations

### Character preset (one-click)
- Toolbar **"Preset Character"** button applies a bundle tuned for character sprites:
  - Cutout + anti-halo + auto-crop + bilateral smoothing
  - Quantization switched to **k-means LAB** (better skin tones)
  - Sharpening and dithering disabled (they add noise on faces)
  - **Silhouette outline** on (see below)
  - Symmetric mirror left off by default — the user toggles it when the pose is front/back symmetric
- Individual post-processing toggles remain available to fine-tune:
  - **Silhouette outline** — morphological 1-px internal erosion ring is forced onto the darkest palette color. Preserves sprite size (no growth), pushes the character's silhouette so it reads at tiny sizes.
  - **Symmetric mirror** — auto-detects which half (left or right) has more opaque pixels and mirrors it onto the other half. Great for standing poses, catches any remaining asymmetry from imperfect cutouts.

### Background mode (tilemap)
- Target up to 256×256 px = 32×32 tiles = one full scroll plane
- Tile-aware quantization: each 8×8 tile is constrained to ≤3 opaque colors
- **Palette budget slider (1–16)**: greedy LAB-distance merge until the global palette count fits
- Live stats: unique tiles / total tiles (with VRAM ≤512 check), palettes used vs budget, saturated tile count
- One-click **"Copy `ngpc_tilemap.py` command"** assembles the CLI invocation with the correct `--max-palettes` and `--black-is-transparent` flags
- Output PNG is directly compatible with `ngpc_tilemap.py` — no manual palette setup needed

### Quantization quality
- **k-means clustering in CIE Lab space** for perceptually better palette choices (vs. plain RGB median-cut)
- **Palette snap** to the NGPC RGB444 grid (4096 candidates) with duplicate avoidance for palette diversity
- **Pixel assignment in Lab** for cleaner anti-aliased edges, not just RGB distance
- Floyd-Steinberg dithering toggle, with opaque-region bbox confinement to avoid error bleeding into transparent areas

### Image prep pipeline
- Contrast / brightness / saturation sliders
- Bilateral filter preprocessing (edge-preserving smoothing) for photos — recommended
- Unsharp mask for contour accentuation
- **Anti-halo erosion** (1 px) after cutout, to kill mixed-color pixels around the subject edge
- **Auto-crop to subject** + aspect-ratio padding — drastically increases effective resolution when your subject occupies a fraction of the source (12× more opaque pixels on a 60×60 subject in a 240×240 frame)
- **Multi-pass Lanczos downscale** — halves iteratively until within 2× of target for better detail preservation on large sources

### Performance
- **Pre-resize sources larger than 1600 px** at load time — a 4000×3000 photo goes through the pipeline in ~125 ms instead of ~835 ms
- **Staged cache** with per-stage invalidation: toggling the quantization method re-runs only the final stage (~0 ms on cache hit). Full stats:
  - Cold run: 75 ms
  - Toggle dither / change method / swap color count: 0–4 ms
  - Change target size: ~72 ms (stages B+C)
  - Change brightness: ~105 ms (full pipeline)
  - Re-run identical params: 0 ms
- **QTimer debounce** (120 ms) coalesces rapid UI events into a single pipeline pass
- **Cached preview pixmap** invalidated only on actual change
- **Warm-up** of the Lab grid at startup (zero first-run stutter on k-means)

---

## Installation

### Requirements
- Python 3.10+
- PySide6 6.5+
- Pillow 10+
- NumPy 1.24+
- opencv-contrib-python 4.8+ (required for GrabCut cutout and anti-halo; otherwise those features degrade gracefully)

### Quick start (Windows)
```bat
run.bat
```
The script creates a local `.venv`, installs dependencies, and launches the app.

### Manual (any platform)
```bash
python -m venv .venv
.venv\Scripts\activate        # or: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

---

## Usage

1. **Open image** — drop any JPG/PNG/BMP/WebP. Sources over 1600 px are downscaled on the fly.
2. **Pick a mode** — Sprite for characters / props / UI elements; Background for scenes and levels.
3. **Frame the subject (sprite mode)** — drag a rectangle on the source view to crop. Right-click to reset.
4. **Refine the cutout (sprite mode, optional)** — press **K** for the green "keep" brush or **R** for the red "remove" brush, paint on the source, release. GrabCut re-runs with your hints. Press **C** to go back to crop mode.
5. **Tweak** — contrast, brightness, saturation, dithering, quantization method. The pipeline re-runs live.
6. **Retouch the output** — in the preview, click a palette swatch and paint pixels. Alt+click on any preview pixel to pick its color. Ctrl+Z undoes both preview retouches and detourage brush strokes.
7. **Export** — PNG that's ready to feed to `ngpc_sprite_export.py` or `ngpc_tilemap.py`.

### One-click character workflow

For a character reference photo:

1. Open the image
2. Click **Preset Character** — sets the full processing chain for you
3. If your character is front/back-facing and symmetric, tick **Sprite symétrique**
4. Pick a palette swatch, Alt+click on a pixel you want to recolor, retouch by painting
5. Export — the output is a 16×16 or 32×32 PNG ready for `ngpc_sprite_export.py`

### Exporting for the toolchain

**Single sprite:**
```bash
python tools/ngpc_sprite_export.py assets/enemy_16x16_3c.png \
  -o GraphX/enemy_mspr.c -n enemy --header \
  --frame-w 16 --frame-h 16 --tile-base 128 --pal-base 1
```

**Dual-layer sprite** (export 2 layers from the tool, feed both to the exporter):
```bash
python tools/ngpc_sprite_export.py assets/boss_layerA.png \
  --layer2 assets/boss_layerB.png \
  -o GraphX/boss_mspr.c -n boss --header \
  --frame-w 32 --frame-h 32 --tile-base 256 --pal-base 0
```

**Background tilemap** (the tool copies this exact command to your clipboard via the "Copy command" button):
```bash
python tools/ngpc_tilemap.py assets/level1_160x152_bg.png \
  -o GraphX/level1.c -n level1 --header \
  --max-palettes 8
```

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| **C** | Crop tool (default) |
| **K** | "Keep" brush (mark foreground) |
| **R** | "Remove" brush (mark background) |
| **E** | Eraser (clear hints) |
| **F5** | Reprocess |
| **Ctrl+Z** | Undo preview retouche or detourage stroke |
| **Ctrl+Wheel** | Zoom source view |
| **Double-click source** | Fit-to-view |
| **Alt+Click preview** | Color eyedropper |

---

## NGPC hardware constraints enforced

| Constraint | Sprite mode | BG mode |
|------------|-------------|---------|
| Color format | RGB444 (4 bit/channel) | RGB444 |
| Palette layout | index 0 = transparent, 1–3 visible | same |
| Colors per tile/sprite | 3 + transparent | 3 + transparent |
| Palette budget | 16 sprite palettes total | 16 per scroll plane |
| Max dimensions | 160×152 px | 256×256 px (one plane), 20×19 visible |
| Tile alignment | N/A | 8×8 pixels, hard-aligned |
| Unique tiles limit | N/A | ≤ 512 (VRAM) — reported in stats |
| Dual-layer option | Yes (`--layer2`) | Phase 2 roadmap |

---

## Technical notes

- **Quantization** lives in `pipeline.py`. `quantize_ngpc` dispatches between median-cut (fast, PIL-driven) and k-means in Lab space (slower, better perceptual). Both paths snap the final palette to the RGB444 grid and reassign each pixel in Lab distance.
- **Tile-aware quantization** (`tile_aware_quantize`) handles BG mode in 5 steps: global candidate pool (~N×3 colors), source remap, per-tile top-3 extraction, greedy palette merging until within budget, per-tile final re-quantization. The greedy merge uses an upper-triangular distance matrix updated incrementally so each merge is O(N) rather than O(N²).
- **Silhouette outline** (`apply_silhouette_outline`) runs a 1-pixel erosion of the alpha mask and paints the resulting 1-pixel ring with the darkest palette color (by luma 0.299/0.587/0.114 weighting), preserving sprite dimensions.
- **Symmetric mirror** (`apply_symmetric_mirror`) picks the half with more opaque pixels and mirrors it onto the other half — handy for cleaning up imperfect cutouts.
- **Staged pipeline** (`StagedPipeline`) caches three stages — preprocessing + cutout, auto-crop + downscale, quantization + post-process — keyed on tuples of parameters and explicit `source_version` / `user_mask_version` counters (so non-hashable PIL images and NumPy masks can still participate in the cache). Post-process (outline, mirror) is gated to sprite mode because it would violate the 3-colors-per-tile BG constraint.
- **Mask seeding for GrabCut** uses `GC_INIT_WITH_MASK` when the user has painted hints, falling back to `GC_INIT_WITH_RECT` for automatic runs.

---

## Project layout

```
NgpCraft_pixel/
├── main.py              # PySide6 UI entry point
├── pipeline.py          # Image processing pipeline + tile-aware quantization
├── requirements.txt
└── run.bat              # Windows launcher (creates venv + installs deps)
```

Supporting files in the broader NgpCraft toolchain (not shipped with this tool but referenced by the exports):
- `ngpc_sprite_export.py` — PNG to C metasprite exporter, supports `--layer2`
- `ngpc_tilemap.py` — PNG to C tilemap exporter
- `ngpc_sprite_bundle.py` — batch sprite export with shared palettes

---

## Related NgpCraft projects

- **NgpCraft toolchain** — assembler, linker, and C compiler targeting TLCS-900H for the NGPC, avoiding the legacy Toshiba tools
- **NgpCraft engine** — the game engine used by Stargunner and other test projects
- **NgpCraft base template** — C project scaffolding with `ngpc_gfx`, `ngpc_soam`, `ngpc_timing`, etc.

---

## License

License not yet specified. No LICENSE file is included.

## Contributing

Bug reports and PRs welcome. The tool is evolving quickly — describe the steps to reproduce issues and the expected result. Remove personal paths and image metadata before sharing diagnostics or samples.


## Public repository contents and privacy

This repository includes the application source, dependency list, Windows launcher,
and maintenance scripts (`diagnose.bat`, `repair.bat`, `install_ml.bat`).
The NGPC exporter tools mentioned above belong to the separate toolchain.

Local test images, generated exports, virtual environments, Python caches, model
weights, and development notes are intentionally excluded. Use your own images;
no sample image is required to start the application. Store private inputs in
`image test/` or `private/`, and generated files in `exports/` (ignored by Git).

`requirements.txt` currently installs the ML dependencies as well as the core
libraries. Model weights are downloaded separately when requested by ML features;
they are not included in this repository. Third-party libraries and model weights
have their own licenses; review those before redistribution.

The diagnostic script prints local Python paths. Redact usernames, local paths,
and other personal information before posting its output in a public issue.


## Automated desktop releases (GitHub Actions)

Push this repository, including `.github/workflows/release.yml`, to GitHub.
Every pushed tag beginning with `v` builds the application on four native runners:

| Download | Platform | Launch after extracting the entire archive |
| --- | --- | --- |
| `NgpCraftPixel-windows-x64.zip` | Windows x64 | `NgpCraftPixel/NgpCraftPixel.exe` |
| `NgpCraftPixel-linux-x64.tar.gz` | Linux x64 (Ubuntu 22.04 or compatible newer distribution) | `NgpCraftPixel/NgpCraftPixel` |
| `NgpCraftPixel-macos-x64.zip` | macOS Intel | `NgpCraftPixel.app` |
| `NgpCraftPixel-macos-arm64.zip` | macOS Apple Silicon | `NgpCraftPixel.app` |

Python is bundled; users do not need to install it. Keep the whole extracted
folder together. Linux still requires system graphics libraries and a desktop
session. macOS builds target macOS 15 or newer. Binaries have no publisher signing
certificate or Apple notarization, so OS security checks may require approval.

To publish a version after committing and pushing the changes:

```bash
git tag -a v1.0.0 -m "NgpCraft Pixel 1.0.0"
git push origin v1.0.0
```

All four builds must pass their frozen-application smoke tests before the workflow
creates the GitHub release and uploads the archives and SHA-256 checksum files.
Tags containing a hyphen, such as `v1.1.0-beta.1`, create prereleases.
Re-running a tag build replaces its assets. Use a new tag for a new version.
The workflow uses GitHub's automatic token; no personal token is required.
GitHub Actions must be enabled and repository policies must allow release writes.

For a test build without publishing, use **Actions > Build desktop releases >
Run workflow** on a branch. Download the resulting artifacts from that run.

ML libraries are included; model weights are not. Model downloads still happen
on demand. Updating Python packages from inside a frozen application is disabled;
install a newer application release instead. ML inference with downloaded models
is not covered by the automated smoke test.

To build locally on the desired operating system in a clean Python 3.11 environment:

```bash
python -m pip install -r requirements-build.txt
python packaging/build.py
```

Builds use PyInstaller's native platform packaging:
[PyInstaller documentation](https://www.pyinstaller.org/en/stable/usage.html).


### Build dependency policy and troubleshooting

`requirements-build.txt` pins the numerical/ML packages to a compatible Python
3.11 stack shared by the four platforms. Numba 0.62.1 and llvmlite 0.45.1 have
Intel macOS wheels; newer Numba releases no longer provide official Intel macOS
support. CI requires binary wheels so it never attempts an LLVM source build.
See the [Numba support policy](https://numba.readthedocs.io/en/stable/user/installing.html).

The standalone bundle explicitly includes `pymatting` distribution metadata,
which its import-time version lookup requires. Both source and frozen smoke tests
check ML imports without downloading weights. Failed builds save their traceback
and PyInstaller warnings as `diagnostics-*` artifacts, separate from release assets.

After changing the workflow or build scripts, push the new commit and run the
workflow on that branch, or push a new version tag. Re-running an old tag uses
its original commit and will not pick up fixes from a newer branch commit.
