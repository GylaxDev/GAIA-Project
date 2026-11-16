# Reference-fitted faces

The user supplied `WD_AP.7z` (Watch_Dogs_AidenPearce XNALara model). Its scoped,
unmodified extraction (`reference_aiden/Watch_Dogs_AidenPearce/`) is **not
bundled** with this project; re-baking the reference-fitted faces requires
restoring that folder from the original archive. Watch Dogs and Aiden Pearce
belong to Ubisoft. The archive provides **no redistribution license**; these
local reference-derived assets are not cleared for publishing or shipping. No
license is inferred from the user's attachment.

## Integration

`tools/aiden_source.py` reads the ASCII model offline using NumPy only.
`tools/bake_human_faces.py` extracts the authored head, preserves its facial UVs,
maps +Z-forward/+Y-up into GAIA's +X-forward/+Z-up frame, and fits eye, mouth,
chin and crown landmarks. It bakes `human_faces.npz` without rewriting the
archived `human_models.npz` or independent `human_layers.npz`.

The player uses the reference head and diffuse/normal/specular maps, with a
fixed geometric identity and green-tinted detailed imported eyes. This is a
reference-derived likeness, not a pixel-identical Watch Dogs render: the rig,
head proportions, cap, clothing, lighting and renderer differ.

Male and female NPCs receive the same detailed head topology and normal /
roughness detail. Female proportions soften/narrow the jaw and nose. Stable
seeds vary the NPC features only inside a tightly bounded range
(`NPC_FACE_VARIATION = .25`, below 4% lateral deformation across dense seed
coverage), and an NPC-only crown/forehead compression brings the detailed
head's tall upper face closer to the citizen pack. NPC skin palettes, lip tint,
freckles, eyebrows, stubble choices and iris colours remain independent of
Aiden's diffuse texture. Clean-shaven NPCs do not inherit his painted beard.

NPC albedo now optionally uses masked photographic identity patches from the
supplied citizen pack, while retaining this detailed head and its normal /
roughness maps. See [CITIZEN_ALBEDO.md](CITIZEN_ALBEDO.md) for the separate bake,
preservation guarantees, limitations and licensing check. The earlier procedural
treatment remains the fallback outside the patch or when that archive is absent.

The head is cut at the neck, and its short transition fits each body LOD's neck
rim. Non-head skin, rig attributes, hair/props and independent garments stay
unchanged except for clearance adjustments to the separate cap and existing
scalp-following hair shells. The new head
is rigidly skinned to the gameplay head joint; facial expressions beyond blink
and gaze are not imported. The same head and textured eyes are kept at both
LODs, along with the near scalp-following hair shells (higher far/shadow
geometry cost, but no disappearing eye sockets or coarse scalp intersections).

The eye baker refits the existing attributed eye source and projects face UVs
onto its animated lid patches. The reference socket's interior is masked behind
the fixed opening, without erasing the animated lids. Face, eye and garment
SHA-256 checks prevent stale assemblies. Runtime uses NumPy arrays and three
mipmapped texture arrays (detail 13, eyes 14, face 15), not XPS/TGA parsing.

## Rebuild and inspect

```powershell
python tools/bake_human_faces.py
python tools/bake_human_eyes.py
python tools/bake_citizen_albedo.py
python tools/preview_humans.py previews/humans/natural_player_preserved.png --face --variant player
python tools/preview_humans.py previews/humans/natural_npc_identities.png --identity-sheet
python tools/preview_humans.py previews/humans/natural_npc_lod1.png --identity-sheet --lod 1
python -m unittest discover -s tests -p 'test_human*.py'
python tools/check_human_animation.py
```

After rebuilding garments, rebuild faces **then** eyes, then NPC albedo.
Remaining inherited clothing seams/coat-tail motion are not fixed by this
facial change. NPC hair uses the existing shells rather than a newly authored
hair-card system; the shared Aiden jaw/nose volume and hairline remain visible
limitations of this approach.

Validation: 37 human tests and the shader portability test pass. Main, MRT and
shadow programs compile on the RTX 3050 with GL error 0. The GPU animation
check covers 1,539 samples; maximum blink error versus the CPU is 0.000124,
with yaw/pitch errors below 0.000006. Player/NPC close-ups, complete blinks,
walking and LOD-1 heads were inspected. Aiden's close-up is pixel-identical to
the preserved pre-change render. A quantized integer hash replaces the old
floating-point sine hash to keep GPU and CPU facial timing consistent.
