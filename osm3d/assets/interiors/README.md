# Photographic room textures

Historical assets. The active renderer uses solid, ray-intersected rooms in
`osm3d/shaders/interiors.py` and does not load these photographs.

`rooms.npz` bundles eight photographs from [Poly Haven](https://polyhaven.com),
whose assets are released under [CC0](https://polyhaven.com/license).

| Layer | Use | Photograph | Photographer |
| --- | --- | --- | --- |
| 0 | Café | [Comfy Café](https://polyhaven.com/a/comfy_cafe) | Sergej Majboroda |
| 1 | Display retail | [Decor Shop](https://polyhaven.com/a/decor_shop) | Sergej Majboroda |
| 2 | Market / stocked shop | [Witsand Woolshop](https://polyhaven.com/a/witsand_woolshop) | Jenelle van Heerden |
| 3 | Arcade / concourse | [Cinema Lobby](https://polyhaven.com/a/cinema_lobby) | Sergej Majboroda |
| 4 | Lobby | [St Fagans Interior](https://polyhaven.com/a/st_fagans_interior) | Morgan Evans |
| 5 | Office | [Poly Haven Studio](https://polyhaven.com/a/poly_haven_studio) | Greg Zaal |
| 6 | Living room | [Brown Photostudio 02](https://polyhaven.com/a/brown_photostudio_02) | Sergej Majboroda |
| 7 | Bedroom | [Hotel Room](https://polyhaven.com/a/hotel_room) | Greg Zaal |

These are representative interiors, selected independently of real business names.
Full download URLs, authors, source hashes, orientation and preparation settings
are recorded in [`sources.json`](sources.json).

## Preparation and format

The 4k RGBE source photographs are exposure-normalized, highlight-compressed and
slightly desaturated. Furniture, materials and contact shading remain photographic.
Selected bright fixture cores form a separate emission mask; photographed windows
are excluded. The office's exterior foliage is replaced with neutral blinds behind
its desks, and the timber ceiling is retouched with tiled panels and luminaires.

The former runtime asset is an eight-layer **1024×1024 octahedral direction map**, with
eleven baked mip levels. RGB is sRGB-encoded photographic shading; alpha is linear
fixture coverage. RGB is filtered in linear light and fixture masks are area-filtered
independently. `GL_SRGB8_ALPHA8` converts RGB back to linear light during sampling.

The former shared array occupied **44,739,232 bytes (42.7 MiB)** of texture storage,
on texture unit **11**. The NPZ is approximately **23.9 MiB** and remains available
to the offline comparison tools and asset tests.

## Rebuild

```powershell
python tools/bake_interior_textures.py --resolution 4k --download
```

Cached sources default to `%LOCALAPPDATA%\Temp\opencode\gaia_interior_sources`.
Use `--sources PATH` to choose another cache and omit `--download` for an offline
rebuild. `--out PATH` selects another output directory. NumPy and the standard
library are sufficient; no image-decoding dependency is required.
