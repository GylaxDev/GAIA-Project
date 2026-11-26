# Offline naked-eye stars

`stars.csv` is a reduced **HYG v4.1** catalog by **David Nash / Astronomy Nexus**,
combining Hipparcos, Yale Bright Star and Gliese data. It is distributed under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).
The same license applies to this derived data, not automatically to the renderer's code.

Source: [HYG database](https://github.com/astronexus/HYG-Database), pinned commit
`3bf37f4b2d5460e1278286320d1d62fab9b493c1`, `hyg/CURRENT/hygdata_v41.csv`.
Original [field documentation and license](https://github.com/astronexus/HYG-Database/blob/3bf37f4b2d5460e1278286320d1d62fab9b493c1/hyg/README.md).

Modifications: remove the Sun, keep visual magnitude **≤ 6.5**, reduce columns,
and sort by magnitude then HYG ID. RA is in hours, declination in degrees,
positions/equinox are **J2000.0**, and proper motions are in milliarcseconds/year.
HIP proper motion in RA includes `cos(dec)`. Missing B−V indices use a neutral
color at rendering time; catalog coordinates/magnitudes are not synthesized.
`catalog.json` records source and derived SHA-256 identities.

The application never downloads this asset. Reproduce it in another directory:

```powershell
python tools/bake_star_catalog.py --out previews/catalog_reproduction
```

The optional constellation guides are manually specified common Western
asterisms connecting catalog stars. They are recognition aids, not the official
IAU constellation boundaries. The ordinary night sky has **no drawn lines**.
