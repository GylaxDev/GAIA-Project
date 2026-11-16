# Citizen NPC meshes (replace the Aiden-based NPC bodies)

Pedestrians now use the 14 supplied citizen characters as their actual bodies, heads, hair
and clothing. They are rigged to the existing 15-joint gameplay skeleton, so the whole gait
system (two-bone leg IK, arm swing, umbrella / phone / coffee / pocket poses, head gaze,
skirt and coat follow-through) drives them unchanged. The player keeps the layered
Aiden body and detailed head.

## What changed

| | before | now |
|---|---|---|
| NPC mesh | shared Aiden head + procedural garments, 127k tris at LOD 0 | citizen mesh, 1.4-2.6k tris (+3.4k hidden prop tris) |
| Identity | seeded face morph, hair/clothing from palettes | one of 14 real outfits/faces chosen by the instance seed (8 male, 6 female) |
| Face | transferred photo patch on a shared head | the citizen's own head and photo-textured face |
| GPU texture | 14 x 1K NPC albedo layers (about 75 MiB) | 28 x 512 layers (about 37 MiB); the old layers are no longer uploaded |

Files: `human_citizens.npz` (baked meshes + textures), `tools/bake_citizen_models.py`
(rebuild), `tools/preview_citizens.py` (CPU preview), `tests/test_human_citizens.py`.
`human_models.npz`, `human_layers.npz`, `human_faces.npz`, `human_eyes.npz` and
`human_npc_albedo.npz` are untouched. If `human_citizens.npz` is missing the old NPCs are used.
The source inputs for re-baking (`reference_citizen/citizen_source.npz` and its
textures) are **not bundled**; restore them from the supplied citizen pack before
running the bake.

## How they are built

1. Weld the FBX corners, rotate from the pack's -y facing to +x, stand on z = 0, scale to 1.78 m.
2. Re-pose the A-pose arms (12-44 degrees out) so they hang beside the body like the shader's rest skeleton.
3. Skin by copying joint influences from the nearest bare-body vertices of the existing male/female
   template. Arms and body are transferred separately, because hanging hands sit next to the thighs
   and would otherwise steal their weights. Skirt and coat hems that stand away from the legs follow
   the coat joints instead, like the old garments.
4. Carry over the umbrella, phones, backpack, shoulder bag, coffee cup and watch from the old template.
   They still switch on by the same accessory flags.
5. Vertices use material `M_CITIZEN` (35); the blink slot holds the texture layer (2k body, 2k+1 outfit).
6. Relax the hands (`tools/relax_citizen_hands.py`, also run by the bake). The pack's hands have the fingers
   fanned wide in the front/back plane and, with no finger joints, ride the forearm joint, so in the gait the
   whole fan swept forward like a reaching hand. Fingers are gathered (about 45 % narrower at the tips) and
   curled 2 cm toward the palm; only skin vertices of the hands move, weights/UVs/indices are unchanged.
   For an already baked asset run `python tools/relax_citizen_hands.py` once (it records `hands_relaxed` in the
   metadata and skips itself the second time).

Renderer: one mesh slot per citizen plus the player (`human_assets.mesh_slots`); draw ranges are packed
by (LOD, slot). Citizens share one VAO between LODs. The citizen texture array is bound on unit 12
around the pedestrian draw and the previous binding (water ripples) is restored afterwards.

## Known limits

* 14 identities. A crowd repeats outfits; per-instance brightness and size/width variation only.
  Palette-driven clothing colours, hair styles, caps, glasses, hoods and beards no longer apply to NPCs.
* The pack has painted eyes: NPCs no longer blink or have independent gaze (head turns still work).
* No normal or roughness maps in the pack; materials use a fixed roughness plus the existing crease folds.
* Textures are 512 px. Set `TEXTURE_SIZE = 1024` in the bake tool for the full-resolution pack
  (about 150 MiB of GPU memory).
* Skin weights are copied from a different body, so skirts and baggy trousers can show slight
  stretching at extreme strides; long coats are not simulated.
* Arm gait (shaders/pedestrian.py `solveGait`): the elbow used to sit at a near-constant 40-46 degrees in a walk
  with the forearms held out in front (a "zombie" look). It is now relaxed in a walk (about 26 degrees when
  the arm is back to 33 degrees forward at sf .35), folds with the swing, still reaches 77-90 degrees for a run, and
  the forearms angle slightly toward the midline. Shoulder swing forward is a little shorter (0.65 instead of 0.75).
* `--pose walk` in the preview is a CPU port of the shader gait (foot trajectory, two-bone IK, pelvis, arm swing,
  coat follow), with skeleton lengths derived from the preview's joint heights and plain-walk stride defaults
  (`--stride`, `--sf`). It tracks the game closely, not exactly.
* GLSL and GPU behaviour were checked by assembling the shader strings and by CPU preview/tests only;
  compile the programs and run the app before shipping.

## Licensing: not cleared for shipping

Same status as [CITIZEN_ALBEDO.md](CITIZEN_ALBEDO.md): the ZIP contains no licence file. The matching
Sketchfab listing by 00amza advertises CC BY 4.0, which is provenance evidence, not verification of
ownership or of the photographed likenesses. Verify, keep the licence and credit, and identify that
the meshes were re-posed, rigged and the textures resampled. Suggested credit *if verified*: "Citizen
NPC Pack 1 Character Lowpoly Mobile - 00amza, Sketchfab, CC BY 4.0; re-posed, rigged and resampled for GAIA."

## Rebuild and inspect

```powershell
python tools/bake_citizen_models.py
python tools/preview_citizens.py previews/humans/citizens_rest.png
python tools/preview_citizens.py previews/humans/citizens_walk.png --pose walk
python tools/preview_citizens.py previews/humans/citizen_3_cycle.png --pose walk --slot 3 --yaws 90 --phases 0 .125 .25 .375 .5 .625 .75 .875
python tools/preview_citizens.py previews/humans/citizen_3.png --slot 3 --yaws 0 45 90 180
python -m unittest discover -s tests -p 'test_human_citizens.py'
```
