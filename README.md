# Space Debris Finder

This small tool fetches public TLE catalogs, computes current sub-satellite locations using Skyfield, performs simple cleaning (dedupe/filter), and can cluster nearby debris using DBSCAN.

Usage


#start debris_3d_anim.html

1. Install dependencies (prefer a virtualenv):

```bash
pip install -r requirements.txt
```

2. Run the script:

```bash
python debris_finder.py
```

Outputs

- `debris_catalog_cleaned.csv`: cleaned catalog with `lat_deg`, `lon_deg`, `alt_m`.
- `debris_map.png`: scatter of sub-satellite points (colored by cluster if used).

Notes

- The script uses public Celestrak TLE URLs by default; you can pass other TLE URLs with `--urls`.
- This is a data-processing and analysis tool — physically "cleaning" space debris is out of scope.
