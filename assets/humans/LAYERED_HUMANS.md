# Separate clothing and procedural faces

The original `human_models.npz` remains an immutable FBX-derived archive.
`human_layers.npz` is built by `python tools/bake_human_layers.py` (NumPy only).
It reuses the archive's rig and UVs, restores its former clothing regions to
skin, and removes the known clothing offsets below the face. Shirts, sleeves,
trousers, shoes, knitwear, coats, collars, lapels, caps and coverings are separate
skinned surfaces. The body stays underneath and is never recolored as clothing.
The layer archive retains the original facial surface. The active runtime face
assembly replaces only its head with the supplied reference-fitted head; see
[AIDEN_REFERENCE.md](AIDEN_REFERENCE.md). Clothing remains independent.

The 24-float vertex layout is unchanged. The last component tags the surface:
`0` body, `-1` garment, `-2` hair/prop, `1` realistic eye, `2..3` eyelid.
Garments inherit up to four joint influences, interpolated at cut hems. Sleeve
and trouser geometry is hidden for short styles instead of shaded as skin.
Both detailed and distance meshes use this assembly.

The player has an open brown leather coat, raised collar, a light ribbed knit
layer, a fitted dark cap and stubble. The face is uncovered by default; set
`Walker.face_covering = True` to enable the separately fitted covering.

`osm3d/human_faces.py` varies jaw breadth, head width, cheek relief, nose, chin,
lip span and socket position from the existing instance seed. It is geometric
variation on the supplied base head, not a second generated head laid over it.
The mapping is shared by main/MRT/shadow shaders and the CPU pose reference.
Its inverse-transpose Jacobian corrects normals. Eye apertures translate
rigidly so iris dimensions, blinking and coordinated gaze remain intact.
No per-character texture or additional instance data is needed. The player now
keeps the authored reference likeness instead of seeded geometric variation.

Inspection:

```powershell
python tools/preview_humans.py previews/humans/layered_player.png --variant player
python tools/preview_humans.py previews/humans/layered_player_mask.png --variant player --mask --face
python tools/preview_humans.py previews/humans/procedural_faces.png --identity-sheet
python tools/preview_humans.py previews/humans/layered_cast_lod1.png --lod 1
python -m unittest discover -s tests -p 'test_human*.py'
```

The preview's `--seed`, `--style` and `--skin-only` options inspect identity,
garment selection and the unpainted underlying body independently. These are
fitted skinned garments, not a cloth-physics simulation.

Known visual limits: small lower-leg/footwear seam intersections and inherited
hair-shell styling remain. Coarse bodies retain the archive's angular anatomy
(the reference-fitted faces stay detailed); coat-tail movement uses the existing gameplay animation, not cloth
collision or simulation.
