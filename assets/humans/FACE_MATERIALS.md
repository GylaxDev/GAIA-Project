# Facial proportions and human materials

The active heads now use the supplied Aiden reference's detailed topology;
see [AIDEN_REFERENCE.md](AIDEN_REFERENCE.md) for fitting, shared NPC detail,
rebuild order, and the archive's missing redistribution license. The earlier
anatomy correction below still fits the retained body/accessory assembly.
The separate citizen NPC albedo transfer and its licensing limitations are
documented in [CITIZEN_ALBEDO.md](CITIZEN_ALBEDO.md); it does not replace this head.

`osm3d/human_anatomy.py` applies a smooth, non-destructive head correction to
both body LODs. The original `human_models.npz`, UVs, joint indices and skin
weights are preserved. Changes include approximately 65 mm eye spacing,
smaller sockets, a restrained nose/lip span, less inflated cheeks, a balanced
chin and a lower hairline. Shading normals are welded across painted facial
and UV seams; the old blocky eyebrow paint is replaced with tapered brows.

Seeded geometric identities and the independent clothing assembly are
documented in [LAYERED_HUMANS.md](LAYERED_HUMANS.md). Clothing UVs/materials
belong to separate garments, not painted regions on the body.

`tools/bake_human_eyes.py` fits the supplied eye model to these corrected heads:
27 mm globes, approximately 12 mm irises and thin, face-following eyelids.
Rebuild faces then eyes after changing the body/layers or anatomy revision.

The pedestrian shader adds subtle pigmentation, cheek/nose/ear blood tint,
filtered pores and sparse freckles, softer natural lip coloration, GGX
material highlights, variable skin roughness and a restrained secondary skin
oil lobe. Subsurface shading is an inexpensive surface approximation, not
volumetric skin simulation. Cloth, leather, metal and glass retain distinct
roughness; wetness lowers roughness without returning to Phong highlights.

Inspect with:

```powershell
python tools/preview_humans.py previews/humans/face_materials.png --face
python tools/preview_humans.py previews/humans/face_materials_blink.png --blink-sheet --variant female
python -m unittest discover -s tests -p 'test_human*.py'
```
