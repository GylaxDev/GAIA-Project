# Times Square billboards

The LED screens on the buildings around Times Square draw these creatives.

## Replacing an ad

Drop a new image in this folder and restart the game. Files load in filename
order, so keep the `NN_name.png` prefix to control the ad rotation order. Any
number of files works (up to 16).

- 16:9 is the native shape. Other aspect ratios are padded with black bars, not
  cropped, so the whole ad stays visible.
- 1024 x 576 is the texture size. Larger images are downscaled; keep important
  text away from the outer 5% so it stays legible when a narrow screen scales
  the whole ad down.
- PNG, JPEG, WebP and BMP are accepted. PNG with no alpha is smallest on disk.
- Use bright, high-contrast art: the shader applies night-time LED radiance and
  bloom, so white text and saturated colour read best.

`tools/bake_billboard_textures.py` regenerates the six starter creatives.
