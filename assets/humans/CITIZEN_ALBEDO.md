# Citizen NPC identity albedo (local prototype)

> **Superseded for runtime NPCs:** pedestrians now use the citizen meshes themselves; see
> [CITIZEN_MODELS.md](CITIZEN_MODELS.md). The albedo layers below are only used by the fallback path
> (when `human_citizens.npz` is absent). The licensing section still applies to both.

The supplied citizen pack contains 14 unrigged, low-poly characters and 28
1024px diffuse textures. The rendered source heads have usable photographic
facial UVs, but include painted eyes, hair, collars, highlights and shadows.
There are no source normal or roughness maps.

## Integration and preservation

`tools/bake_citizen_albedo.py` projects the actual textured source faces into a
front view, aligns pupil/nose/mouth/chin landmarks measured with OpenCV YuNet,
and transfers a feathered facial patch into the **existing detailed head UVs**.
Connected bangs, painted eyeballs, outer head/hair silhouette, scalp and collar
are masked out. Two bright side-bang patches have additional inspected exclusion
masks. Photo luminance is compressed to 40% while 35% of source chroma is
retained, so lips, brows, freckles and stubble survive without reproducing the
source's lighting. The photographed nose relief is neutralized to the same
citizen skin tone, so the detailed nose geometry shades itself instead of
receiving a second pair of painted nostrils. Cheek pigment is normalized so the
runtime skin palette still matches the neck, hands and retained body. The upper
fade now reaches the geometric hairline (1.734-1.746 m), avoiding a pale strip
of original Aiden forehead above the transferred patch.

The new `human_npc_albedo.npz` contains **only 14 RGBA albedo layers and metadata**:
8 male identities followed by 6 female identities. Alpha is transfer coverage,
not transparency of the head. Selection uses the existing normalized instance
seed and variant, so it is stable across animation and LODs. Where the patch is
absent, the existing procedural pigment/brows/stubble treatment remains.
Photographic illumination and some front-projection distortion remain; this is
not de-lit scan-quality albedo or a promise to reproduce the citizen geometry.

`human_faces.npz`, `human_eyes.npz`, `human_layers.npz`, and `human_models.npz`
remain **byte-for-byte unchanged**. The source pack is not installed as runtime
geometry or clothing. Aiden retains layer 0 and his fixed head identity; the
existing normal/specular-to-roughness maps remain layers 1 and 2. Independent
eyes and animated lids, scalp shells, accessories and garments are retained.

`face_textures()` still returns the original immutable three-map array.
`render_face_textures()` appends NPC albedo layers for renderer/preview upload
on the existing texture unit 15; no additional sampler is used. A missing NPC
archive falls back to the old treatment. A changed detailed-head hash or identity
order is rejected and requires rebaking **only NPC albedo**. Face/eye fitting and
their existing dependency hashes are not changed. Fourteen extra mipmapped 1K
RGBA layers cost approximately 75 MiB of additional GPU texture storage.

## NPC proportions and shading

The detailed head remains Aiden's. NPC-only runtime corrections keep the
supplied identities near average-human proportions without replacing or
rewriting the shared head:

- `NPC_FACE_VARIATION = .25` (formerly `1.8`). Across 257 seeds, random lateral
  deformation stays below 4% of the vertex offset. The previous `1.8` gain
  allowed almost 20% jaw-width and 14% mouth-width changes.
- A shared crown/forehead compression (`NPC_FOREHEAD_BROW = .029`,
  `NPC_FOREHEAD_SCALE = .60`) shortens only geometry above the NPC brow. The
  measured crown moves from 1.7900 m to 1.7572 m (male) and 1.7810 m to
  1.7482 m (female). The player, eye pivots and lower face are untouched.
- Female lip tint uses the fitted eye height instead of the old body landmark,
  removing the previous ~7 mm vertical offset.
- NPC normal-map strength is fixed at 0.35 (player 0.80), and NPC irises use
  subdued brown tones. Eye geometry, eye textures, animated lids and the
  normal/roughness maps remain the existing detailed assets.

The taller detailed forehead, the shared Aiden jaw/nose volume and the
procedural hairline remain visible limitations of reusing this head. Source
hair is deliberately not transferred as geometry or painted bangs.

## Licensing: not cleared for shipping

The supplied ZIP `citizen-npc-pack-1-character-lowpoly-mobile.zip` contains only
`source/Citizen NPC.fbx` and 28 PNG textures: **no license or attribution file**.

A matching [Sketchfab listing by 00amza](https://sketchfab.com/3d-models/citizen-npc-pack-1-character-lowpoly-mobile-711ee48bcf80426bbf98ad01e1b9da1c)
was checked on 2026-11-16. It advertises **CC Attribution / CC BY 4.0**, linking
to <https://creativecommons.org/licenses/by/4.0/>. The title/archive name and
`Citizen 00amza_*` mesh names match. This is useful provenance evidence, **not
verification that the uploader owns the underlying models or photographed
likenesses, or that this particular archive is cleared**.

Before publishing: verify the download's provenance and redistribution rights,
retain the applicable license and author/source attribution, and identify the
derived texture changes. Suggested attribution *if verified*: “Citizen NPC Pack
1 Character Lowpoly Mobile — 00amza, Sketchfab, CC BY 4.0; facial albedo remapped
and masked for GAIA.” Link the listing and license with that credit. Separately,
the detailed shared head and Aiden maps are still reference-derived Ubisoft
assets with no supplied redistribution permission; see
[AIDEN_REFERENCE.md](AIDEN_REFERENCE.md). The CC listing does not clear those.

## Rebuild and inspect

```powershell
python tools/preview_citizen_source.py
python tools/bake_citizen_albedo.py
python tools/preview_humans.py previews/humans/natural_citizen_sheet.png --citizen-sheet
python tools/preview_humans.py previews/humans/natural_npc_identities.png --identity-sheet
python tools/preview_humans.py previews/humans/natural_npc_lod1.png --identity-sheet --lod 1
python tools/preview_humans.py previews/humans/natural_npc_views.png --face --variant male --seed .71
python tools/preview_humans.py previews/humans/natural_player_preserved.png --face --variant player
python -m unittest discover -s tests -p 'test_human*.py'
python -m unittest discover -s tests -p 'test_shader_portability.py'
python tools/check_human_animation.py
```

Regression checks cover original asset hashes, unchanged player/detail maps,
all identity layers, painted-eye/scalp/neck masks, neutralized nose pigment,
source-hair masking, optional asset fallback and stale-bake rejection. Aiden's
close-up render was compared against the pre-change render and is pixel-identical.

Validation: 37 human tests and the shader portability test pass. Main, MRT and
shadow programs compile with GL error 0 on the RTX 3050. The existing GPU eye
animation check passes all 1,539 samples (maximum blink error 0.000124).
