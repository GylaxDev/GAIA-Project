# Recorded V8 engine

These PCM16/44.1 kHz mono assets use all three user-supplied recordings:

- `V8 Engine Startup and Revving  HQ Sound Effects - Everyday Cinematic Sounds.mp3`:
  idle, low exhaust and loaded rev texture.
- `car startup.mp3`: the starter, with its long quiet tail removed.
- `freesound_community-engine-47745.mp3`: a steady ~2640 RPM running-engine layer,
  used for smooth higher revs instead of repeatedly pitching a fading rev take.

They contain recorded engine texture, not synthetic instrument tones. `v8.json`
records source filenames, ranges and whole-cycle counts. Earlier startup assets
are retained, but the active starter is now `car_startup.wav`.

The bake removes the recording's leading silence, tracks combustion timing,
removes weak release cycles, levels the loops smoothly, and joins them over
phase-matched engine cycles. At runtime,
one continuous crank clock drives every layer; the automatic gearbox determines
RPM and the throttle controls exhaust brightness. Loops are band-limited for
their playback range, including when the mixer runs below 44.1 kHz. Adjacent
recordings are phase-aligned and blended with smooth equal-power fades.

Startup and idle share the audio sample clock, so a delayed render frame cannot
leave a silent gap when the starter finishes. Windows playback uses a native
WinMM PCM reservoir (normally 240 ms, adaptive up to 400 ms), not pygame's
Python-dependent short-buffer completion callbacks. RPM trajectory prediction
keeps revving continuous despite that safety margin. Other platforms and dummy
devices use longer pygame buffers. `OSM3D_ENGINE_BACKEND=mixer` forces that fallback.
Native device failures also fall back to the mixer without killing the worker.

Rebuild with the existing numpy/pygame dependencies:

```powershell
python tools/bake_engine_audio.py 'C:\path\to\V8 Engine Startup and Revving  HQ Sound Effects - Everyday Cinematic Sounds.mp3' --startup 'C:\path\to\car startup.mp3' --engine 'C:\path\to\freesound_community-engine-47745.mp3'
```

Source attribution: **Everyday Cinematic Sounds** and **Freesound Community**, as
identified by the supplied filenames; the starter's creator was not identified.
No redistribution licences were supplied with the recordings. Confirm the
source owners' licences/permission before distributing these derived assets.
Missing assets use an uneven-bank combustion/exhaust fallback instead.

## Brake and crash

`Car brake and crash Sound Effect - Audio Vampire - Foley Sounds Effects.mp3` (user supplied)
is cut into two PCM16/44.1 kHz mono assets:

- `car_skid.wav`: the steady brake squeal (0.85-2.45 s of the source), joined into a
  seamless 1.42 s loop with an equal-power cross-fade. Plays on the slide channel.
- `car_crash.wav`: the impact and debris tail (2.55 s to the end). Used for medium and
  hard collisions; light bumps stay synthesized.

`audio.py` mixes a street-canyon echo (slap-backs plus a dark tail) into these and the other
car one-shots at start-up. `ECHO` at the top of `audio.py` scales it (0 = dry, 2 = twice as wet).

## In-car wind and suspension

- `freesound_community-car-strong-wind-noise-58443.mp3` (user supplied) -> `car_wind.wav`: an
  11 s steady stretch of the 2-minute recording, cross-faded into a seamless 10 s loop. It
  replaces the synthesized wind and is scaled with speed.
- `All Car Suspension Sounds Fortnite - Sound Effects for editing - Sound Library.mp3`
  (user supplied) -> `car_suspension_0/1/2.wav`: its three separate clips, from a short
  creak to a deep heavy hit. They play when a wheel's vertical speed spikes (bump, kerb,
  landing). Bumps under 2 m/s are silent; clip 0 plays to ~3.2 m/s, clip 1 to ~4.8 m/s, clip 2
  above that, with the volume rising steeply with the size of the hit.
- `SKID_PITCH` in `audio.py` raises the recorded skid loop (1.5 = a fifth higher).
