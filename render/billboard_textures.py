"""Drop-in LED billboard creatives for Times Square.

Every image in ``osm3d/assets/billboards`` is loaded in filename order into one
shared 16:9 texture array.  The neon shader picks a layer per LED screen from
the screen's deterministic seed, so replacing or adding a file only requires a
restart.  Images of another aspect are padded to 16:9 rather than cropped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "billboards"
TEXTURE_WIDTH, TEXTURE_HEIGHT = 1024, 576
TEXTURE_ASPECT = TEXTURE_WIDTH / TEXTURE_HEIGHT
MAX_LAYERS = 16
SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def texture_paths(directory: Path | str = ASSET_DIR) -> list[Path]:
    """Billboard files in stable draw order; bad directory means no billboards."""
    try:
        entries = sorted(Path(directory).iterdir(), key=lambda path: path.name.lower())
    except OSError:
        return []
    return [path for path in entries
            if path.is_file() and path.suffix.lower() in SUFFIXES][:MAX_LAYERS]


def _pillow_frame(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        image = ImageOps.pad(image, (TEXTURE_WIDTH, TEXTURE_HEIGHT),
                             method=Image.Resampling.LANCZOS,
                             color=(0, 0, 0, 255), centering=(.5, .5))
        return np.array(image, np.uint8)


def _pygame_frame(path: Path) -> np.ndarray:
    import pygame
    source = pygame.image.load(str(path))
    width, height = source.get_size()
    scale = min(TEXTURE_WIDTH / max(width, 1), TEXTURE_HEIGHT / max(height, 1))
    size = max(1, round(width * scale)), max(1, round(height * scale))
    # smoothscale needs a 32-bit source; the scratch surface also flattens
    # palette and alpha modes without requiring an initialized display.
    scratch = pygame.Surface((width, height), pygame.SRCALPHA)
    scratch.blit(source, (0, 0))
    scaled = pygame.transform.smoothscale(scratch, size)
    canvas = pygame.Surface((TEXTURE_WIDTH, TEXTURE_HEIGHT), pygame.SRCALPHA)
    canvas.fill((0, 0, 0, 255))
    canvas.blit(scaled, ((TEXTURE_WIDTH - size[0]) // 2, (TEXTURE_HEIGHT - size[1]) // 2))
    data = pygame.image.tostring(canvas, "RGBA", False)
    return np.frombuffer(data, np.uint8).reshape(TEXTURE_HEIGHT, TEXTURE_WIDTH, 4).copy()


def load_billboard_images(directory: Path | str = ASSET_DIR):
    """Return ``(frames, paths)`` as RGBA8 ``(n, h, w, 4)`` plus the files used.

    A single unreadable replacement must not remove the rest of the ad rotation,
    so files that cannot be decoded are skipped.  ``frames`` is ``None`` when
    nothing could be loaded; the shader then keeps its procedural screen look.
    """
    paths = texture_paths(directory)
    if not paths:
        return None, []

    decoders = []
    try:
        import PIL  # noqa: F401
        decoders.append(_pillow_frame)
    except Exception:
        pass
    try:
        import pygame  # noqa: F401
        decoders.append(_pygame_frame)
    except Exception:
        pass
    if not decoders:
        print("[billboards] Pillow/pygame unavailable, using procedural screens")
        return None, []

    frames, loaded = [], []
    for path in paths:
        error = None
        for decode in decoders:
            try:
                frames.append(decode(path))
                loaded.append(path)
                break
            except Exception as exc:
                error = exc
        else:
            print(f"[billboards] skipped {path.name}: {error}")
    if not frames:
        return None, []
    return np.ascontiguousarray(np.stack(frames), np.uint8), loaded
