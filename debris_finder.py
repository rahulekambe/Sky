#!/usr/bin/env python3
"""
Fetch TLEs from public catalogs, compute current locations using Skyfield,
clean the catalog (dedupe/filter), and optionally cluster debris with DBSCAN.

Outputs a cleaned CSV and an optional scatter plot of sub-satellite points.
"""
import argparse
import io
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import DBSCAN
from skyfield.api import EarthSatellite, Loader


DEFAULT_URLS = [
    "https://celestrak.org/NORAD/elements/visual.txt",
    "https://celestrak.org/NORAD/elements/debris.txt",
    "https://celestrak.org/NORAD/elements/launch.txt",
]


def fetch_tle_text(url):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.text


def parse_tles(text):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    tles = []
    i = 0
    while i + 2 < len(lines):
        # Many catalogs use 3-line blocks: name, line1, line2
        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]
        if line1.startswith('1 ') and line2.startswith('2 '):
            tles.append((name, line1, line2))
            i += 3
        else:
            # Skip one line to try to realign
            i += 1
    return tles


def compute_positions(tles, ts, load):
    rows = []
    for name, l1, l2 in tles:
        try:
            sat = EarthSatellite(l1, l2, name)
            t = ts.now()
            geoc = sat.at(t)
            sub = geoc.subpoint()
            lat = sub.latitude.degrees
            lon = sub.longitude.degrees
            alt_m = sub.elevation.m
            rows.append({
                'name': name,
                'line1': l1,
                'line2': l2,
                'lat_deg': lat,
                'lon_deg': lon,
                'alt_m': alt_m,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def clean_catalog(df):
    before = len(df)
    df = df.drop_duplicates(subset=['line1', 'line2'])
    df = df.dropna(subset=['lat_deg', 'lon_deg'])
    # Filter to reasonable Low/LEO+MEO+GEO range (20 km - 200000 km)
    df = df[(df['alt_m'] > 20000) & (df['alt_m'] < 200000000)]
    after = len(df)
    return df, before, after


def cluster_positions(df, eps_km=50, min_samples=3):
    # Convert lat/lon to radians and run DBSCAN with haversine distance.
    coords = np.radians(df[['lat_deg', 'lon_deg']].values)
    kms_per_radian = 6371.0088
    eps = eps_km / kms_per_radian
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine').fit(coords)
    df['cluster'] = db.labels_
    return df


def plot_map(df, out_path):
    plt.figure(figsize=(10, 5))
    sc = plt.scatter(df['lon_deg'], df['lat_deg'], c=df.get('cluster', -1), cmap='tab20', s=8)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Space Debris Sub-Satellite Points (colored by cluster)')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Find and clean space debris locations')
    parser.add_argument('--urls', nargs='*', help='TLE source URLs', default=DEFAULT_URLS)
    parser.add_argument('--out', help='Output CSV path', default='debris_catalog_cleaned.csv')
    parser.add_argument('--plot', help='Output PNG path (scatter)', default='debris_map.png')
    parser.add_argument('--cluster', action='store_true', help='Run DBSCAN clustering')
    parser.add_argument('--eps-km', type=float, default=50.0, help='DBSCAN eps in km')
    parser.add_argument('--min-samples', type=int, default=3, help='DBSCAN min_samples')
    args = parser.parse_args()

    load = Loader('./.skyfield-cache')
    ts = load.timescale()

    all_tles = []
    for url in args.urls:
        try:
            print(f'Fetching {url}...')
            txt = fetch_tle_text(url)
            tles = parse_tles(txt)
            print(f'  parsed {len(tles)} TLEs')
            all_tles.extend(tles)
        except Exception as e:
            print(f'  failed to fetch/parse {url}: {e}', file=sys.stderr)

    if not all_tles:
        print('No TLEs found; exiting.', file=sys.stderr)
        sys.exit(1)

    df = compute_positions(all_tles, ts, load)
    df, before, after = clean_catalog(df)
    print(f'Cleaned catalog: {before} -> {after} entries')

    if args.cluster and len(df) > 0:
        df = cluster_positions(df, eps_km=args.eps_km, min_samples=args.min_samples)

    df.to_csv(args.out, index=False)
    print(f'Wrote cleaned catalog to {args.out}')

    if len(df) > 0:
        try:
            plot_map(df, args.plot)
            print(f'Wrote map to {args.plot}')
        except Exception as e:
            print(f'Failed to draw map: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
