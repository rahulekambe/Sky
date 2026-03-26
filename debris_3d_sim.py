#!/usr/bin/env python3
"""Generate an interactive 3D Earth with debris plotted in orbit.

Reads `debris_catalog_cleaned.csv` (expects `lat_deg`, `lon_deg`, `alt_m`, optional `cluster`) and
writes `debris_3d.html` (interactive Plotly file).
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go


EARTH_RADIUS_KM = 6371.0088


def sph_to_cart(lat_deg, lon_deg, radius_km):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = radius_km * np.cos(lat) * np.cos(lon)
    y = radius_km * np.cos(lat) * np.sin(lon)
    z = radius_km * np.sin(lat)
    return x, y, z


def make_earth_surface(res=50, radius=EARTH_RADIUS_KM):
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(-np.pi / 2, np.pi / 2, res // 2)
    u, v = np.meshgrid(u, v)
    x = radius * np.cos(v) * np.cos(u)
    y = radius * np.cos(v) * np.sin(u)
    z = radius * np.sin(v)
    return x, y, z


def main(csv_path='debris_catalog_cleaned.csv', out_html='debris_3d.html'):
    if not os.path.exists(csv_path):
        print(f'CSV not found: {csv_path}', file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    # Ensure required columns exist
    for c in ('lat_deg', 'lon_deg', 'alt_m'):
        if c not in df.columns:
            print(f'Missing column {c} in {csv_path}', file=sys.stderr)
            sys.exit(1)

    # Convert alt to km and compute cartesian coords
    df['alt_km'] = df['alt_m'] / 1000.0
    df['rad_km'] = EARTH_RADIUS_KM + df['alt_km']
    xs, ys, zs = sph_to_cart(df['lat_deg'].values, df['lon_deg'].values, df['rad_km'].values)
    df['x'] = xs
    df['y'] = ys
    df['z'] = zs

    # Earth surface
    ex, ey, ez = make_earth_surface(res=160)

    # Use surface elevation as surfacecolor for subtle shading and add lighting
    surface = go.Surface(x=ex, y=ey, z=ez, surfacecolor=ez, showscale=False,
                         colorscale='Blues', lighting=dict(ambient=0.7, diffuse=0.6, roughness=0.9),
                         lightposition=dict(x=100, y=0, z=0), opacity=0.95)

    # Debris scatter: color by altitude (km), size inversely proportional to altitude
    alt_km = df['alt_km'].values
    # scale sizes so LEO points appear larger
    sizes = np.clip(12 * (1.0 / (np.sqrt(alt_km + 1) / 5.0)), 3, 14)
    marker = dict(size=sizes, color=alt_km, colorscale='Viridis', colorbar=dict(title='alt (km)'),
                  opacity=0.9)

    scatter = go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], mode='markers', marker=marker,
                           text=df.apply(lambda r: f"{r.get('name','')}: {r['alt_km']:.1f} km", axis=1),
                           hoverinfo='text')

    # Altitude lines from Earth's surface up to the debris point (one long trace with NaN separators)
    x_lines = []
    y_lines = []
    z_lines = []
    for _, r in df.iterrows():
        sx, sy, sz = sph_to_cart(r['lat_deg'], r['lon_deg'], EARTH_RADIUS_KM)
        x_lines.extend([sx, r['x'], None])
        y_lines.extend([sy, r['y'], None])
        z_lines.extend([sz, r['z'], None])

    orbit_lines = go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines',
                               line=dict(color='white', width=1), opacity=0.35, hoverinfo='none')

    layout = go.Layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        scene=dict(aspectmode='data',
                   xaxis=dict(showbackground=False, visible=False),
                   yaxis=dict(showbackground=False, visible=False),
                   zaxis=dict(showbackground=False, visible=False),
                   camera=dict(eye=dict(x=1.7, y=1.2, z=0.8))),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text='Space Debris 3D Simulation', x=0.5, xanchor='center', font=dict(color='white'))
    )

    fig = go.Figure(data=[surface, orbit_lines, scatter], layout=layout)

    fig.write_html(out_html, auto_open=False)
    print(f'Wrote {out_html}')


if __name__ == '__main__':
    main()
